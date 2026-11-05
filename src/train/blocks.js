// TF.js layers for the Transformer blocks that TF.js does not ship. Each one keeps the same
// weights as the PyTorch / Keras export, so the parameter counts on the canvas match all three.
import * as tf from "@tensorflow/tfjs";

const first = (inputs) => (Array.isArray(inputs) ? inputs[0] : inputs);

// GELU, the exact (erf) form PyTorch's nn.GELU uses.
const gelu = (x) => tf.mul(tf.mul(x, 0.5), tf.add(1, tf.erf(tf.div(x, Math.SQRT2))));

// Sine/cosine position table, [n, dim] (Vaswani et al. 2017).
function sincos(n, dim) {
  const out = new Float32Array(n * dim);
  for (let pos = 0; pos < n; pos++) {
    for (let i = 0; i < dim; i += 2) {
      const angle = pos * Math.exp((-Math.log(10000) * i) / dim);
      out[pos * dim + i] = Math.sin(angle);
      if (i + 1 < dim) out[pos * dim + i + 1] = Math.cos(angle);
    }
  }
  return out;
}

export class PositionEmbedding extends tf.layers.Layer {
  static className = "TBPositionEmbedding";
  constructor(config = {}) {
    super(config);
    this.learned = config.learned ?? true;
    this.cls = config.cls ?? false;
  }
  build(inputShape) {
    const [, t, d] = inputShape;
    this.n = t + (this.cls ? 1 : 0);
    this.d = d;
    if (this.cls) this.clsToken = this.addWeight("cls_token", [1, 1, d], "float32", tf.initializers.zeros());
    if (this.learned) this.pos = this.addWeight("pos", [1, this.n, d], "float32", tf.initializers.randomNormal({ stddev: 0.02 }));
    else this.table = sincos(this.n, d);
    this.built = true;
  }
  computeOutputShape(s) {
    return [s[0], s[1] + (this.cls ? 1 : 0), s[2]];
  }
  call(inputs) {
    return tf.tidy(() => {
      let x = first(inputs);
      if (this.cls) x = tf.concat([tf.tile(this.clsToken.read(), [x.shape[0], 1, 1]), x], 1);
      return tf.add(x, this.learned ? this.pos.read() : tf.tensor3d(this.table, [1, this.n, this.d]));
    });
  }
  getConfig() {
    return { ...super.getConfig(), learned: this.learned, cls: this.cls };
  }
}

// Multi-head self-attention with Q, K, V, O projections: 4·D·D + 4·D values, like nn.MultiheadAttention.
export class SelfAttention extends tf.layers.Layer {
  static className = "TBSelfAttention";
  constructor(config = {}) {
    super(config);
    this.heads = config.heads ?? 4;
    this.causal = config.causal ?? false;
  }
  build(inputShape) {
    const d = inputShape[inputShape.length - 1];
    if (d % this.heads) throw new Error(`토큰 칸 수 ${d}가 헤드 수 ${this.heads}로 나눠떨어지지 않아요`);
    this.w = {};
    for (const k of ["q", "k", "v", "o"]) {
      this.w[k] = [
        this.addWeight(`${k}_kernel`, [d, d], "float32", tf.initializers.glorotUniform({})),
        this.addWeight(`${k}_bias`, [d], "float32", tf.initializers.zeros()),
      ];
    }
    this.built = true;
  }
  computeOutputShape(s) {
    return s;
  }
  call(inputs) {
    return tf.tidy(() => {
      const x = first(inputs);
      const [, t, d] = x.shape;
      const h = this.heads;
      const dh = d / h;
      const flat = x.reshape([-1, d]);
      const proj = (y, k) => tf.add(tf.matMul(y, this.w[k][0].read()), this.w[k][1].read());
      const split = (y) => y.reshape([-1, t, h, dh]).transpose([0, 2, 1, 3]); // [B, heads, T, dh]
      const q = split(proj(flat, "q"));
      const k = split(proj(flat, "k"));
      const v = split(proj(flat, "v"));
      let scores = tf.mul(tf.matMul(q, k, false, true), 1 / Math.sqrt(dh)); // [B, heads, T, T]
      if (this.causal) {
        const allowed = tf.linalg.bandPart(tf.ones([t, t]), -1, 0); // the token itself and the ones before it
        scores = tf.add(scores, tf.mul(tf.sub(1, allowed), -1e9));
      }
      const mixed = tf.matMul(tf.softmax(scores), v).transpose([0, 2, 1, 3]).reshape([-1, d]);
      return proj(mixed, "o").reshape([-1, t, d]);
    });
  }
  getConfig() {
    return { ...super.getConfig(), heads: this.heads, causal: this.causal };
  }
}

// [T, D] → [D]: the mean over tokens, or the CLS token in front.
export class TokenPool extends tf.layers.Layer {
  static className = "TBTokenPool";
  constructor(config = {}) {
    super(config);
    this.mode = config.mode ?? "mean";
  }
  computeOutputShape(s) {
    return [s[0], s[2]];
  }
  call(inputs) {
    return tf.tidy(() => {
      const x = first(inputs);
      return this.mode === "cls" ? x.slice([0, 0, 0], [-1, 1, -1]).reshape([-1, x.shape[2]]) : x.mean(1);
    });
  }
  getConfig() {
    return { ...super.getConfig(), mode: this.mode };
  }
}

// Root-mean-square norm over the last axis with a learned scale (D values), like nn.RMSNorm.
export class RMSNorm extends tf.layers.Layer {
  static className = "TBRMSNorm";
  constructor(config = {}) {
    super(config);
    this.eps = config.eps ?? 1e-6;
  }
  build(inputShape) {
    this.weight = this.addWeight("weight", [inputShape[inputShape.length - 1]], "float32", tf.initializers.ones());
    this.built = true;
  }
  computeOutputShape(s) {
    return s;
  }
  call(inputs) {
    return tf.tidy(() => {
      const x = first(inputs);
      return tf.mul(tf.mul(x, tf.rsqrt(tf.add(tf.mean(tf.square(x), -1, true), this.eps))), this.weight.read());
    });
  }
  getConfig() {
    return { ...super.getConfig(), eps: this.eps };
  }
}

// Mixture of experts, dense for teaching: every expert runs, then the gate's top-k are mixed.
export class MoEFFN extends tf.layers.Layer {
  static className = "TBMoEFFN";
  constructor(config = {}) {
    super(config);
    this.experts = config.experts ?? 4;
    this.topk = config.topk ?? 2;
    this.hidden = config.hidden ?? 128;
    // Which experts are picked is a yes/no choice: no gradient flows through the picking itself.
    const k = this.topk;
    this.pick = tf.customGrad((logits) => {
      const kth = tf.topk(logits, k).values.slice([0, k - 1], [-1, 1]);
      return { value: tf.cast(tf.greaterEqual(logits, kth), "float32"), gradFunc: () => tf.zerosLike(logits) };
    });
  }
  build(inputShape) {
    const d = inputShape[inputShape.length - 1];
    const h = this.hidden;
    const w = (name, shape, init) => this.addWeight(name, shape, "float32", init);
    this.gate = [w("gate_kernel", [d, this.experts], tf.initializers.glorotUniform({})), w("gate_bias", [this.experts], tf.initializers.zeros())];
    this.ffn = Array.from({ length: this.experts }, (_, e) => [
      w(`expert${e}_w1`, [d, h], tf.initializers.glorotUniform({})),
      w(`expert${e}_b1`, [h], tf.initializers.zeros()),
      w(`expert${e}_w2`, [h, d], tf.initializers.glorotUniform({})),
      w(`expert${e}_b2`, [d], tf.initializers.zeros()),
    ]);
    this.built = true;
  }
  computeOutputShape(s) {
    return s;
  }
  call(inputs) {
    return tf.tidy(() => {
      const x = first(inputs);
      const d = x.shape[x.shape.length - 1];
      const flat = x.reshape([-1, d]);
      const logits = tf.add(tf.matMul(flat, this.gate[0].read()), this.gate[1].read()); // [N, E]
      const mask = this.pick(logits);
      const weights = tf.softmax(tf.add(logits, tf.mul(tf.sub(1, mask), -1e9))); // 0 for experts not picked
      const outs = this.ffn.map(([w1, b1, w2, b2]) => tf.add(tf.matMul(gelu(tf.add(tf.matMul(flat, w1.read()), b1.read())), w2.read()), b2.read()));
      const mixed = tf.sum(tf.mul(tf.stack(outs, 2), weights.expandDims(1)), 2); // [N, D]
      return mixed.reshape(x.shape);
    });
  }
  getConfig() {
    return { ...super.getConfig(), experts: this.experts, topk: this.topk, hidden: this.hidden };
  }
}

for (const C of [PositionEmbedding, SelfAttention, TokenPool, RMSNorm, MoEFFN]) tf.serialization.registerClass(C);

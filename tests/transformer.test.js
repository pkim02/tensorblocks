// Transformer blocks: shapes and parameter counts follow PyTorch's own formulas, the TF.js
// model built for in-browser training has exactly the values the canvas counts, and the
// exported code carries the helper classes it needs.
import assert from "node:assert/strict";
import { test } from "node:test";
import * as tf from "@tensorflow/tfjs";
import { buildTemplate, analyze, TEMPLATES } from "../src/model/graph.js";
import { useApp } from "../src/store.js";
import { LAYERS, PALETTE, fmtShapeLong } from "../src/model/layers.js";
import { buildModel } from "../src/train/train.js";
import { pytorchCode, kerasCode } from "../src/export/code.js";

tf.setBackend("cpu");

const run = (spec, sides = [], numClasses = 3) => {
  const { nodes, edges } = buildTemplate({ spec, sides });
  const a = analyze(nodes, edges, numClasses);
  const at = (type, k = 0) => {
    const id = a.order.filter((i) => nodes.find((n) => n.id === i).data.type === type)[k];
    return a.nodeInfo[id];
  };
  return { a, nodes, edges, at };
};

// PyTorch formulas, written out once so the canvas is checked against them rather than itself.
const torchParams = {
  conv: (k, cin, cout) => k * k * cin * cout + cout,
  linear: (i, o) => i * o + o,
  mha: (d) => 3 * d * d + 3 * d + d * d + d, // in_proj (q, k, v) + out_proj
  layerNorm: (d) => 2 * d,
};

test("shape words: token sequences read as tokens × numbers", () => {
  assert.equal(fmtShapeLong([196, 768]), "토큰 196개 × 768칸");
  assert.equal(fmtShapeLong([10]), "10칸");
  assert.equal(fmtShapeLong([7, 7, 3]), "7×7×3");
});

test("Patch Embedding: image → tokens, Conv2d(k=s=patch) parameters, uneven cut is an error", () => {
  const { a, at } = run([["input", { size: 64 }], ["patchembed", { patch: 8, dim: 64 }], ["seqpool"], ["output"]]);
  assert.equal(a.ok, true);
  assert.deepEqual(at("patchembed").outShape, [64, 64]);
  assert.equal(at("patchembed").params, torchParams.conv(8, 3, 64));

  const bad = run([["input", { size: 64 }], ["maxpool", { pool: 3 }], ["patchembed", { patch: 8 }], ["seqpool"], ["output"]]);
  const issue = bad.at("patchembed").issue;
  assert.equal(issue.title, "조각이 딱 나뉘지 않아요"); // 21×21 can't be cut into 8×8
  assert.equal(issue.fixes[0].action.kind, "remove"); // 21 has no divisor among 4, 8, 16

  const tokens = run([["input", { size: 224 }], ["patchembed", { patch: 4 }], ["seqpool"], ["output"]]);
  assert.equal(tokens.at("patchembed").issue.level, "warn"); // 3,136 tokens
  assert.deepEqual(tokens.at("patchembed").issue.fixes[0].action, { kind: "setParam", key: "patch", value: 8 });
});

test("Position Embedding: learned T×D, sine 0, CLS adds one token and D values", () => {
  const base = [
    ["input", { size: 32 }],
    ["patchembed", { patch: 8, dim: 32 }],
  ];
  const learned = run([...base, ["posembed"], ["seqpool"], ["output"]]);
  assert.deepEqual(learned.at("posembed").outShape, [16, 32]);
  assert.equal(learned.at("posembed").params, 16 * 32);
  assert.equal(run([...base, ["posembed", { mode: "sine" }], ["seqpool"], ["output"]]).at("posembed").params, 0);

  const cls = run([...base, ["posembed"], ["attention"], ["seqpool", { mode: "cls" }], ["output"]]);
  assert.equal(cls.a.ok, true);
  assert.deepEqual(cls.at("posembed").outShape, [17, 32]);
  assert.equal(cls.at("posembed").params, 17 * 32 + 32);
  assert.deepEqual(cls.at("seqpool").outShape, [32]);

  const noPos = run([...base, ["seqpool", { mode: "cls" }], ["output"]]);
  assert.equal(noPos.at("seqpool").issue.title, "CLS 토큰이 없어요");
  assert.deepEqual(noPos.at("seqpool").issue.fixes[0].action, { kind: "setParam", key: "mode", value: "mean" });
});

test("Self-Attention: nn.MultiheadAttention parameters; heads must divide D", () => {
  const { at } = run([["input", { size: 64 }], ["patchembed", { patch: 8, dim: 64 }], ["attention", { heads: 4 }], ["seqpool"], ["output"]]);
  assert.deepEqual(at("attention").outShape, [64, 64]);
  assert.equal(at("attention").params, torchParams.mha(64));

  const bad = run([["input", { size: 64 }], ["patchembed", { patch: 8, dim: 60 }], ["attention", { heads: 8 }], ["seqpool"], ["output"]]);
  const issue = bad.at("attention").issue;
  assert.equal(issue.title, "헤드 수로 칸이 안 나눠져요");
  assert.deepEqual(issue.fixes[0].action, { kind: "setParam", key: "heads", value: 4 });

  // An image straight into attention: cut it into patch tokens first.
  const image = run([["input", { size: 64 }], ["attention"], ["seqpool"], ["output"]]);
  assert.equal(image.at("attention").issue.fixes[0].action.type, "patchembed");
});

test("Dense, LayerNorm, Dropout, Activation act on the last axis of tokens", () => {
  const { a, at } = run([
    ["input", { size: 32 }],
    ["patchembed", { patch: 8, dim: 32 }],
    ["layernorm"],
    ["dense", { units: 48, activation: "gelu" }],
    ["dropout"],
    ["activation", { fn: "gelu" }],
    ["seqpool"],
    ["output"],
  ]);
  assert.equal(a.ok, true);
  assert.deepEqual(at("dense").outShape, [16, 48]);
  assert.equal(at("dense").params, torchParams.linear(32, 48));
  assert.equal(at("layernorm").params, torchParams.layerNorm(32));
  assert.deepEqual(at("dropout").outShape, [16, 48]);
});

test("token sequences into vector or image blocks get a 시퀀스 요약 fix", () => {
  const out = run([["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["output"]]);
  assert.deepEqual(out.at("output").issue.fixes[0].action, { kind: "insertBefore", type: "seqpool" });
  const gap = run([["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["gap"], ["output"]]);
  assert.deepEqual(gap.at("gap").issue.fixes[0].action, { kind: "replace", type: "seqpool" });
  const bn = run([["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["batchnorm"], ["seqpool"], ["output"]]);
  assert.deepEqual(bn.at("batchnorm").issue.fixes[0].action, { kind: "replace", type: "layernorm" });
});

test("Add on tokens: same shapes add up; a different D gets a Dense on the side branch", () => {
  const ok = run(
    [["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["dense", { units: 32 }], ["add"], ["seqpool"], ["output"]],
    [{ into: 3, from: 1, spec: [] }],
  );
  assert.equal(ok.a.ok, true);
  assert.equal(ok.a.exportOnly, false);
  const bad = run(
    [["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["dense", { units: 48 }], ["add"], ["seqpool"], ["output"]],
    [{ into: 3, from: 1, spec: [] }],
  );
  const fix = bad.at("add").issue.fixes[0].action;
  assert.equal(fix.type, "dense");
  assert.equal(fix.params.units, 48);
});

const VIT = {
  spec: [
    ["input", { size: 32 }],
    ["patchembed", { patch: 8, dim: 32 }],
    ["posembed"],
    ["layernorm"],
    ["attention", { heads: 4 }],
    ["add"],
    ["layernorm"],
    ["dense", { units: 64, activation: "gelu" }],
    ["dense", { units: 32, activation: "linear" }],
    ["add"],
    ["layernorm"],
    ["seqpool", { mode: "cls" }],
    ["output"],
  ],
  sides: [
    { into: 5, from: 2, spec: [] },
    { into: 9, from: 5, spec: [] },
  ],
};

const trainableCount = (model) => model.trainableWeights.reduce((s, w) => s + w.shape.reduce((x, y) => x * y, 1), 0);

test("the TF.js model for training has exactly the canvas's parameters and shapes", () => {
  const { a, nodes } = run(VIT.spec, VIT.sides, 3);
  assert.equal(a.ok, true);
  const graph = { ...a, byId: new Map(nodes.map((n) => [n.id, n])) };
  const { model, probe, layerNodeIds } = buildModel(graph, a.order[0], [32, 32, 3], 3);
  assert.equal(trainableCount(model), a.trainableParams);
  const outs = probe.predict(tf.zeros([2, 32, 32, 3]));
  outs.forEach((t, i) => assert.deepEqual(t.shape.slice(1), a.nodeInfo[layerNodeIds[i]].outShape, nodes.find((n) => n.id === layerNodeIds[i]).data.type));
  tf.dispose([model, probe, outs].flat());
});

test("a small ViT trains in TF.js: the loss goes down", async () => {
  const { a, nodes } = run(VIT.spec, VIT.sides, 2);
  const { model } = buildModel({ ...a, byId: new Map(nodes.map((n) => [n.id, n])) }, a.order[0], [32, 32, 3], 2);
  // Vertical vs horizontal stripes: every patch shows which one it is, so the CLS token only
  // has to gather it through attention.
  const n = 16;
  const xs = tf.tidy(() => {
    const stripes = tf.tile(tf.tensor1d(Array.from({ length: 32 }, (_, i) => (i >> 1) % 2)).reshape([1, 32, 1, 1]), [n / 2, 1, 32, 3]);
    return tf.concat([stripes, stripes.transpose([0, 2, 1, 3])]).add(tf.randomUniform([n, 32, 32, 3], 0, 0.3));
  });
  const ys = tf.oneHot(tf.tensor1d([...Array(n / 2).fill(0), ...Array(n / 2).fill(1)], "int32"), 2).toFloat();
  model.compile({ optimizer: tf.train.adam(0.003), loss: "categoricalCrossentropy" });
  const h = await model.fit(xs, ys, { epochs: 8, batchSize: 8, shuffle: true, verbose: 0 });
  const loss = h.history.loss;
  assert.ok(loss.every(Number.isFinite));
  assert.ok(loss.at(-1) < Math.min(loss[0], Math.log(2)), loss.join(", "));
  tf.dispose([model, xs, ys]);
});

test("exported PyTorch and Keras code carries the helper classes the model uses", () => {
  const { a, nodes } = run(VIT.spec, VIT.sides, 2);
  const classes = [
    { name: "A", samples: [] },
    { name: "B", samples: [] },
  ];
  const settings = { epochs: 1, batchSize: 4, learningRate: 0.001 };
  const py = pytorchCode(a, nodes, classes, settings);
  for (const s of [
    "import math",
    "class PatchEmbed",
    "class PositionEmbedding",
    "class SelfAttention",
    "class TokenPool",
    "PatchEmbed(3, 32, patch_size=8)",
    "PositionEmbedding(16, 32, learned=True, with_cls=True)",
    "SelfAttention(32, num_heads=4)",
    'TokenPool("cls")',
    "nn.GELU()",
    "x5 = x4 + x2",
  ]) {
    assert.ok(py.includes(s), s);
  }
  assert.ok(!py.includes("class MoEFFN"));
  const keras = kerasCode(a, nodes, classes, settings);
  for (const s of [
    "import tensorflow as tf",
    "class PositionEmbedding",
    "class SelfAttention",
    "class TokenPool",
    "layers.Reshape((16, 32))(layers.Conv2D(32, 8, strides=8)(x))",
    "layers.Add()",
  ]) {
    assert.ok(keras.includes(s), s);
  }
});

test("every new block has its Korean text and is placed in the palette", () => {
  const palette = PALETTE.flatMap((g) => g.items);
  for (const type of ["patchembed", "posembed", "attention", "seqpool", "rmsnorm", "moe", "upsample", "convtranspose"]) {
    const def = LAYERS[type];
    assert.ok(def.ko && def.about && def.summary(def.defaults, { numClasses: 2 }), type);
    assert.equal(!def.exportOnly, !["upsample", "convtranspose"].includes(type), type);
    assert.ok(palette.includes(type), type);
    for (const f of def.fields) assert.ok(f.label, type);
  }
  assert.ok(PALETTE.some((g) => g.title === "트랜스포머 층"));
  assert.ok(!LAYERS.add.exportOnly);
});

test("templates: the small ViT and ConvNeXt block are ready to train in the browser", () => {
  for (const id of ["vit", "convnext", "resnet"]) {
    const t = TEMPLATES.find((x) => x.id === id);
    const { nodes, edges } = buildTemplate(t);
    const a = analyze(nodes, edges, 3);
    assert.equal(a.ok, true, id);
    assert.equal(a.exportOnly, false, id);
    const size = nodes.find((n) => n.data.type === "input").data.params.size;
    const { model, probe } = buildModel({ ...a, byId: new Map(nodes.map((n) => [n.id, n])) }, a.order[0], [size, size, 3], 3);
    assert.equal(trainableCount(model), a.trainableParams, id);
    tf.dispose([model, probe]);
  }
  const vit = buildTemplate(TEMPLATES.find((x) => x.id === "vit"));
  const a = analyze(vit.nodes, vit.edges, 3);
  assert.equal(a.order.filter((id) => vit.nodes.find((n) => n.id === id).data.type === "add").length, 4);
  assert.deepEqual(a.nodeInfo[a.chain.at(-2)].outShape, [64]);
});

test("template: the heads mistake shows one red note, and its fix makes the model whole", () => {
  useApp.getState().loadTemplate("heads-broken");
  let s = useApp.getState();
  let a = analyze(s.nodes, s.edges, s.classes.length);
  const errors = a.issues.filter((i) => i.level === "error");
  assert.equal(errors.length, 1);
  assert.equal(errors[0].title, "헤드 수로 칸이 안 나눠져요");
  s.applyFix(errors[0].nodeId, errors[0].fixes[0].action);
  s = useApp.getState();
  a = analyze(s.nodes, s.edges, s.classes.length);
  assert.equal(a.ok, true);
  assert.equal(s.nodes.find((n) => n.data.type === "attention").data.params.heads, 4);
});

test("RMSNorm: D values on vectors and tokens; images get a LayerNorm fix", () => {
  const tokens = run([["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["rmsnorm"], ["seqpool"], ["rmsnorm"], ["output"]]);
  assert.equal(tokens.a.ok, true);
  assert.equal(tokens.at("rmsnorm").params, 32); // nn.RMSNorm(32).weight
  assert.equal(tokens.at("rmsnorm", 1).params, 32);
  const image = run([["input", { size: 32 }], ["rmsnorm"], ["gap"], ["output"]]);
  assert.deepEqual(image.at("rmsnorm").issue.fixes[0].action, { kind: "replace", type: "layernorm" });
});

test("MoE FFN: gate + experts of Linear-GELU-Linear, shape kept", () => {
  const { a, at } = run([
    ["input", { size: 32 }],
    ["patchembed", { patch: 8, dim: 32 }],
    ["moe", { experts: 4, topk: 2, hidden: 64 }],
    ["seqpool"],
    ["output"],
  ]);
  assert.equal(a.ok, true);
  assert.deepEqual(at("moe").outShape, [16, 32]);
  assert.equal(at("moe").params, torchParams.linear(32, 4) + 4 * (torchParams.linear(32, 64) + torchParams.linear(64, 32)));
  assert.match(LAYERS.moe.about, /모든 전문가를 다 계산/); // the simplification is said out loud
});

test("Upsample2D and ConvTranspose2D grow images and stay export-only", () => {
  const { a, at } = run([
    ["input", { size: 32 }],
    ["upsample", { factor: 2 }],
    ["convtranspose", { filters: 8, kernel: 3, stride: 2, padding: "same" }],
    ["convtranspose", { filters: 4, kernel: 3, stride: 2, padding: "valid" }],
    ["gap"],
    ["output"],
  ]);
  assert.equal(a.ok, true);
  assert.equal(a.exportOnly, true);
  assert.deepEqual(at("upsample").outShape, [64, 64, 3]);
  assert.deepEqual(at("convtranspose").outShape, [128, 128, 8]);
  assert.deepEqual(at("convtranspose", 1).outShape, [257, 257, 4]); // (128 - 1)·2 + 3
  assert.equal(at("convtranspose").params, torchParams.conv(3, 3, 8));
});

test("RMSNorm and MoE build in TF.js with the canvas's parameters and train without errors", async () => {
  const { a, nodes } = run(
    [
      ["input", { size: 32 }],
      ["patchembed", { patch: 8, dim: 32 }],
      ["posembed"],
      ["rmsnorm"],
      ["attention", { heads: 2, mask: "causal" }],
      ["rmsnorm"],
      ["moe", { experts: 3, topk: 1, hidden: 16 }],
      ["seqpool"],
      ["output"],
    ],
    [],
    2,
  );
  assert.equal(a.ok, true);
  const { model } = buildModel({ ...a, byId: new Map(nodes.map((n) => [n.id, n])) }, a.order[0], [32, 32, 3], 2);
  assert.equal(trainableCount(model), a.trainableParams);
  model.compile({ optimizer: tf.train.adam(0.003), loss: "categoricalCrossentropy" });
  const xs = tf.randomUniform([8, 32, 32, 3]);
  const ys = tf.oneHot(tf.tensor1d([0, 1, 0, 1, 0, 1, 0, 1], "int32"), 2).toFloat();
  const h = await model.fit(xs, ys, { epochs: 2, batchSize: 4, verbose: 0 });
  assert.ok(h.history.loss.every(Number.isFinite));
  tf.dispose([model, xs, ys]);
});

test("exported code for RMSNorm, MoE, Upsample and ConvTranspose", () => {
  const classes = [
    { name: "A", samples: [] },
    { name: "B", samples: [] },
  ];
  const settings = { epochs: 1, batchSize: 4, learningRate: 0.001 };
  const tok = run(
    [["input", { size: 32 }], ["patchembed", { patch: 8, dim: 32 }], ["rmsnorm"], ["moe", { experts: 2, topk: 1, hidden: 16 }], ["seqpool"], ["output"]],
    [],
    2,
  );
  const py = pytorchCode(tok.a, tok.nodes, classes, settings);
  for (const s of ['if not hasattr(nn, "RMSNorm")', "nn.RMSNorm(32)", "class MoEFFN", "MoEFFN(32, hidden=16, num_experts=2, top_k=1)"])
    assert.ok(py.includes(s), s);
  const keras = kerasCode(tok.a, tok.nodes, classes, settings);
  for (const s of ["class RMSNorm", "class MoEFFN", "MoEFFN(32, 16, 2, 1)"]) assert.ok(keras.includes(s), s);
  const img = run([["input", { size: 32 }], ["upsample"], ["convtranspose", { filters: 8 }], ["gap"], ["output"]], [], 2);
  const py2 = pytorchCode(img.a, img.nodes, classes, settings);
  assert.ok(py2.includes("nn.Upsample(scale_factor=2)"));
  assert.ok(py2.includes("nn.ConvTranspose2d(3, 8, kernel_size=3, stride=2, padding=1, output_padding=1)"));
  assert.ok(kerasCode(img.a, img.nodes, classes, settings).includes('layers.Conv2DTranspose(8, 3, strides=2, padding="same", activation="relu")'));
});

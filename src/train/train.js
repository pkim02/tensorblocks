// In-browser training with TF.js, the same way Teachable Machine does it:
// a frozen MobileNetV2 turns each photo into features once, and only the blocks after it learn.
// Without a pretrained block, the whole model is trained from the pixels up.
import * as tf from "@tensorflow/tfjs";

// The exact weights Teachable Machine loads (MobileNetV2, alpha 0.35, 224px, no top).
const MOBILENET_URL = "https://storage.googleapis.com/teachable-machine-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top/model.json";

let mobilenet = null;
export async function loadMobileNet() {
  if (!mobilenet) {
    mobilenet = tf.loadLayersModel(MOBILENET_URL).then((m) => tf.model({ inputs: m.inputs, outputs: m.getLayer("out_relu").output }));
    mobilenet.catch(() => (mobilenet = null));
  }
  return mobilenet;
}

const featureCache = new Map(); // sample id → [7, 7, 1280] tensor

function toInput(canvas, cfg) {
  return tf.tidy(() => {
    let x = tf.browser.fromPixels(canvas).toFloat();
    if (cfg.backbone) return x.div(127.5).sub(1).expandDims(0);
    if (cfg.size !== x.shape[0]) x = tf.image.resizeBilinear(x, [cfg.size, cfg.size]);
    return x.div(255).expandDims(0);
  });
}

function makeLayer(node, extra, numClasses) {
  const p = node.data.params;
  switch (node.data.type) {
    case "conv2d":
      return tf.layers.conv2d({ filters: +p.filters, kernelSize: +p.kernel, strides: +p.stride, padding: p.padding, activation: p.activation, ...extra });
    case "maxpool":
      return tf.layers.maxPooling2d({ poolSize: +p.pool, strides: +p.pool, ...extra });
    case "gap":
      return tf.layers.globalAveragePooling2d(extra);
    case "flatten":
      return tf.layers.flatten(extra);
    case "dense":
      return tf.layers.dense({ units: +p.units, activation: p.activation, ...extra });
    case "dropout":
      return tf.layers.dropout({ rate: +p.rate, ...extra });
    case "batchnorm":
      return tf.layers.batchNormalization(extra);
    case "output":
      return tf.layers.dense({ units: numClasses, activation: "softmax", ...extra });
    default:
      throw new Error(`${node.data.type} 블록은 브라우저에서 학습할 수 없어요`);
  }
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export async function trainModel({ chainNodes, classes, settings, onStatus, onEpoch, shouldStop }) {
  const input = chainNodes[0];
  const usesBackbone = chainNodes[1]?.data.type === "mobilenet";
  const headNodes = chainNodes.slice(usesBackbone ? 2 : 1);
  const numClasses = classes.length;
  const size = +input.data.params.size;
  const samples = shuffle(classes.flatMap((c, label) => c.samples.map((s) => ({ s, label }))));

  let backbone = null;
  let xs;
  if (usesBackbone) {
    onStatus("MobileNet 불러오는 중…");
    backbone = await loadMobileNet();
    const live = new Set(samples.map((x) => x.s.id));
    for (const [id, t] of featureCache) {
      if (!live.has(id)) {
        t.dispose();
        featureCache.delete(id);
      }
    }
    const feats = [];
    for (let i = 0; i < samples.length; i++) {
      const { s } = samples[i];
      let f = featureCache.get(s.id);
      if (!f) {
        f = tf.tidy(() => backbone.predict(toInput(s.canvas, { backbone: true })).squeeze([0]));
        featureCache.set(s.id, f);
      }
      feats.push(f);
      if (i % 10 === 0) {
        onStatus(`사진에서 특징 뽑는 중… ${i + 1}/${samples.length}`);
        await tf.nextFrame();
      }
    }
    xs = tf.stack(feats);
  } else {
    onStatus("사진을 숫자로 바꾸는 중…");
    await tf.nextFrame();
    xs = tf.tidy(() => tf.concat(samples.map(({ s }) => toInput(s.canvas, { size }))));
  }
  const ys = tf.tidy(() =>
    tf
      .oneHot(
        tf.tensor1d(
          samples.map((x) => x.label),
          "int32",
        ),
        numClasses,
      )
      .toFloat(),
  );

  const nVal = samples.length >= 20 ? Math.max(numClasses, Math.round(samples.length * 0.15)) : 0;
  const nTrain = samples.length - nVal;
  const [xTrain, xVal] = nVal ? tf.split(xs, [nTrain, nVal]) : [xs, null];
  const [yTrain, yVal] = nVal ? tf.split(ys, [nTrain, nVal]) : [ys, null];

  const head = tf.sequential();
  const inShape = usesBackbone ? backbone.outputs[0].shape.slice(1) : [size, size, 3];
  headNodes.forEach((n, i) => head.add(makeLayer(n, i === 0 ? { inputShape: inShape } : {}, numClasses)));
  head.compile({ optimizer: tf.train.adam(+settings.learningRate), loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  onStatus("학습 중…");
  try {
    await head.fit(xTrain, yTrain, {
      epochs: +settings.epochs,
      batchSize: +settings.batchSize,
      shuffle: true,
      validationData: nVal ? [xVal, yVal] : undefined,
      callbacks: {
        onBatchEnd: () => {
          if (shouldStop()) head.stopTraining = true;
        },
        onEpochEnd: async (epoch, logs) => {
          onEpoch(epoch, {
            loss: logs.loss,
            acc: logs.acc ?? logs.accuracy,
            valLoss: logs.val_loss,
            valAcc: logs.val_acc ?? logs.val_accuracy,
          });
          await tf.nextFrame();
        },
      },
    });
  } finally {
    tf.dispose([xs, ys, xTrain, yTrain, xVal, yVal].filter((t, i, a) => t && a.indexOf(t) === i));
  }

  return {
    head,
    probe: tf.model({ inputs: head.inputs, outputs: head.layers.map((l) => l.output) }),
    backbone,
    size,
    backboneNodeId: usesBackbone ? chainNodes[1].id : null,
    layerNodeIds: headNodes.map((n) => n.id),
    numClasses,
    nTrain,
    nVal,
  };
}

export function disposeTrained(t) {
  if (t?.head) t.head.dispose();
}

// Runs one picture through the model and returns what every block produced:
// a few feature maps for image-shaped outputs, the raw numbers for vectors.
export function inspect(trained, canvas) {
  return tf.tidy(() => {
    let x = toInput(canvas, trained.backbone ? { backbone: true } : { size: trained.size });
    const outs = [];
    if (trained.backbone) {
      x = trained.backbone.predict(x);
      outs.push([trained.backboneNodeId, x]);
    }
    let ys = trained.probe.predict(x);
    if (!Array.isArray(ys)) ys = [ys];
    ys.forEach((t, i) => outs.push([trained.layerNodeIds[i], t]));
    return outs.map(([nodeId, t]) => ({ nodeId, ...summarize(t) }));
  });
}

const MAPS = 6;
function summarize(t) {
  if (t.shape.length === 4) {
    const [, h, w, c] = t.shape;
    const m = t.squeeze([0]);
    const k = Math.min(MAPS, c);
    // Show the channels that reacted most to this picture.
    const { indices } = tf.topk(m.mean([0, 1]), k);
    const picked = tf.gather(m, indices, 2).transpose([2, 0, 1]);
    const min = picked.min([1, 2], true),
      max = picked.max([1, 2], true);
    const norm = picked.sub(min).div(max.sub(min).add(1e-6));
    const data = norm.dataSync();
    const maps = Array.from({ length: k }, (_, i) => data.slice(i * h * w, (i + 1) * h * w));
    return { kind: "maps", h, w, c, maps };
  }
  const values = Array.from(t.squeeze([0]).dataSync());
  return { kind: "vec", n: values.length, values };
}

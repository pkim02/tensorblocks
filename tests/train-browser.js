// Trains one template on the ready-made shapes dataset with the real in-browser pipeline
// (canvas photos → TF.js on WebGL), then looks inside one photo. The result is left on
// window.__result for scripts/check-browser-training.mjs to read.
import * as tf from "@tensorflow/tfjs";
import { useApp } from "../src/store.js";
import { makeShapesDataset } from "../src/data/demo.js";
import { startTraining } from "../src/train/run.js";
import { inspect } from "../src/train/train.js";
import { buildTemplate } from "../src/model/graph.js";

const q = new URLSearchParams(location.search);
const log = document.getElementById("log");
const show = (text) => (log.textContent = text);

try {
  await tf.ready();
  useApp.getState().loadTemplate(q.get("t") || "vit");
  // ?spec=<JSON of a template> tries a model that is not in the template list.
  if (q.get("spec")) useApp.setState(buildTemplate(JSON.parse(q.get("spec"))));
  useApp.setState({
    classes: makeShapesDataset(+(q.get("per") || 40)),
    settings: { epochs: +(q.get("epochs") || 20), batchSize: 16, learningRate: +(q.get("lr") || 0.001) },
  });
  const unsub = useApp.subscribe((s) => show(`${s.train.message} ${s.train.epoch}/${s.train.total}`));
  const t0 = performance.now();
  await startTraining();
  unsub();
  const s = useApp.getState();
  const inside = s.trained ? inspect(s.trained, s.classes[0].samples[0].canvas) : [];
  window.__result = {
    backend: tf.getBackend(),
    seconds: Math.round((performance.now() - t0) / 1000),
    status: s.train.status,
    message: s.train.message,
    logs: s.train.logs,
    inside: inside.map((r) => r.kind),
  };
} catch (e) {
  window.__result = { status: "crash", message: String(e?.stack || e) };
}
show(JSON.stringify(window.__result, null, 1));

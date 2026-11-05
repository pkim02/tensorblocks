import assert from "node:assert/strict";
import { test } from "node:test";
import { useApp } from "../src/store.js";
import { TEMPLATES, analyze, buildTemplate, checkConnect, checkInsert, makeNode, probeShape } from "../src/model/graph.js";

const fresh = (id = "cnn") => {
  const t = TEMPLATES.find((x) => x.id === id);
  useApp.setState({ templateId: id, selectedId: null, preview: null, panel: "data", ...buildTemplate(t) });
  return useApp.getState();
};
const byType = (s, type, k = 0) => s.nodes.filter((n) => n.data.type === type)[k];

test("palette pick previews the block in the block tab without touching the model", () => {
  const before = fresh();
  useApp.getState().previewBlock("dense");
  const s = useApp.getState();
  assert.equal(s.preview, "dense");
  assert.equal(s.panel, "block");
  assert.equal(s.nodes.length, before.nodes.length);
  assert.equal(s.edges, before.edges);
});

test("adding a block places it unconnected, selects it and opens the block tab", () => {
  const before = fresh();
  useApp.getState().previewBlock("dropout");
  const id = useApp.getState().addBlock("dropout", { x: 300, y: 400 }, false);
  const s = useApp.getState();
  assert.equal(s.nodes.length, before.nodes.length + 1);
  assert.equal(s.edges.length, before.edges.length);
  assert.ok(!s.edges.some((e) => e.source === id || e.target === id));
  assert.equal(s.selectedId, id);
  assert.equal(s.panel, "block");
  assert.equal(s.preview, null);
  assert.equal(s.templateId, null);
});

test("inserting a fix switches to the block tab too", () => {
  fresh("broken");
  const dense = byType(useApp.getState(), "dense");
  useApp.getState().applyFix(dense.id, { kind: "insertBefore", type: "flatten" });
  assert.equal(useApp.getState().panel, "block");
});

test("removing one line keeps both blocks", () => {
  const s = fresh();
  const e = s.edges[0];
  s.removeEdge(e.id);
  const after = useApp.getState();
  assert.equal(after.nodes.length, s.nodes.length);
  assert.ok(!after.edges.some((x) => x.id === e.id));
});

test("connection fit check: green where shapes fit, red where they don't", () => {
  const s = fresh();
  const n = s.classes.length;
  const pool = byType(s, "maxpool", 1); // 16×16×32 image
  const flatten = byType(s, "flatten");
  const dense = byType(s, "dense");
  assert.equal(checkConnect(pool.id, flatten.id, s.nodes, s.edges, n), "ok");
  assert.equal(checkConnect(pool.id, dense.id, s.nodes, s.edges, n), "bad"); // Dense needs a vector
  assert.equal(checkConnect(dense.id, dense.id, s.nodes, s.edges, n), "bad");
});

test("fit check follows lines through a block that is only half wired up", () => {
  const s = fresh();
  const n = s.classes.length;
  const flatten = byType(s, "flatten");
  const loose = makeNode("dense", { x: 0, y: 400 }, { units: 8 });
  const nodes = [...s.nodes, loose];
  const edges = [...s.edges, { id: "x1", source: flatten.id, target: loose.id, type: "shape" }];
  assert.deepEqual(probeShape(loose.id, nodes, edges, n), [8]);
  assert.equal(checkConnect(loose.id, byType(s, "output").id, nodes, edges, n), "ok");
  assert.equal(checkConnect(loose.id, byType(s, "conv2d").id, nodes, edges, n), "bad");
  // Nothing flows into a block that has no lines yet, so the fit is unknown.
  const lone = makeNode("dense", { x: 0, y: 600 });
  assert.equal(checkConnect(lone.id, byType(s, "output").id, [...nodes, lone], edges, n), "unknown");
});

test("drop-on-line check looks at both sides of the new block", () => {
  const s = fresh();
  const n = s.classes.length;
  const flatten = byType(s, "flatten");
  const intoFlatten = s.edges.find((e) => e.target === flatten.id); // image → Flatten
  const outOfFlatten = s.edges.find((e) => e.source === flatten.id); // vector → Dense
  assert.equal(checkInsert(intoFlatten.id, "conv2d", s.nodes, s.edges, n), "ok");
  assert.equal(checkInsert(outOfFlatten.id, "conv2d", s.nodes, s.edges, n), "bad"); // Conv2D needs an image
  assert.equal(checkInsert(outOfFlatten.id, "dropout", s.nodes, s.edges, n), "ok");
  assert.equal(checkInsert(intoFlatten.id, "output", s.nodes, s.edges, n), "bad"); // nothing can follow an output
});

test("Korean particles follow how block names are read aloud", async () => {
  const { josa } = await import("../src/model/terms.js");
  const cases = {
    Dense: "Dense는", Flatten: "Flatten은", Conv2D: "Conv2D는", MobileNetV2: "MobileNetV2는", GlobalAvgPool: "GlobalAvgPool은",
    Embedding: "Embedding은", SigLIP: "SigLIP은", "ResNet-18": "ResNet-18은", Concat: "Concat은", Add: "Add는",
    "분류 출력": "분류 출력은", "글 입력": "글 입력은", "이미지 모양(64×64×3)": "이미지 모양(64×64×3)은", "한 줄로 펴진 벡터(64)": "한 줄로 펴진 벡터(64)는",
  };
  for (const [word, want] of Object.entries(cases)) assert.equal(josa(word, "은는"), want);
});

test("shape errors read naturally and keep the PyTorch message apart", () => {
  const s = fresh("broken");
  const dense = byType(s, "dense");
  const issue = analyze(s.nodes, s.edges, s.classes.length).nodeInfo[dense.id].issue;
  assert.equal(issue.title, "여기 모양이 안 맞아요");
  assert.match(issue.text, /^Dense는 한 줄짜리 벡터만 받아요/);
  assert.doesNotMatch(issue.text, /PyTorch|\(는\)/);
  assert.match(issue.code, /mat1 and mat2/);
});


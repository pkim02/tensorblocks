import assert from "node:assert/strict";
import { test } from "node:test";
import { buildTemplate, analyze, TEMPLATES } from "../src/model/graph.js";
import { pytorchCode, kerasCode, notebookCells } from "../src/export/code.js";
import { serializeProject, restoreProject } from "../src/projects/serialize.js";

test("restored tm, CNN, VLM and skip models preserve the existing code and notebook output", () => {
  const classes = [
    { id: "a", name: "A", samples: [] },
    { id: "b", name: "B", samples: [] },
  ];
  const settings = { epochs: 1, batchSize: 16, learningRate: 0.001 };
  for (const id of ["tm", "cnn", "vlm", "resnet"]) {
    const original = { ...buildTemplate(TEMPLATES.find((t) => t.id === id)), templateId: id, classes, settings };
    const state = restoreProject(serializeProject(original));
    const a = analyze(state.nodes, state.edges, classes.length);
    assert.equal(a.ok, true, id);
    const before = analyze(original.nodes, original.edges, classes.length);
    for (const generate of [pytorchCode, kerasCode]) {
      assert.equal(generate(a, state.nodes, classes, settings), generate(before, original.nodes, classes, settings), id);
    }
    const content = (cells) => cells.map(({ cell_type, source }) => ({ cell_type, source }));
    assert.deepEqual(content(notebookCells(a, state.nodes, classes, settings)), content(notebookCells(before, original.nodes, classes, settings)), id);
  }
});

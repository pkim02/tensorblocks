import assert from "node:assert/strict";
import { test } from "node:test";
import { useApp } from "../src/store.js";
import { useHistory, undo, redo, dragStart, dragEnd } from "../src/history.js";

const tick = () => new Promise((r) => setTimeout(r, 0));
const types = () => useApp.getState().nodes.map((n) => n.data.type).join(",");

test("undo and redo step through deletes, adds and line cuts", async () => {
  useApp.getState().loadTemplate("cnn");
  await tick();
  const start = types();
  const dropout = useApp.getState().nodes.find((n) => n.data.type === "dropout");
  useApp.getState().removeNode(dropout.id); // remove + tidy in one tick → one step
  await tick();
  assert.equal(useHistory.getState().canUndo, true);
  assert.doesNotMatch(types(), /dropout/);

  const edgeCount = useApp.getState().edges.length;
  useApp.getState().removeEdge(useApp.getState().edges[0].id);
  await tick();

  undo();
  await tick();
  assert.equal(useApp.getState().edges.length, edgeCount);
  undo();
  await tick();
  assert.equal(types(), start);
  assert.equal(useHistory.getState().canRedo, true);

  redo();
  await tick();
  assert.doesNotMatch(types(), /dropout/);
  // A new change after undo drops the redo path.
  useApp.getState().addBlock("dense", { x: 0, y: 800 }, false);
  await tick();
  assert.equal(useHistory.getState().canRedo, false);
});

test("selecting a block is not a step; a whole drag is one step", async () => {
  useApp.getState().loadTemplate("tm");
  await tick();
  const id = useApp.getState().nodes[1].id;
  const before = useApp.getState().nodes[1].position;
  useApp.getState().select(id);
  useApp.getState().onNodesChange([{ type: "select", id, selected: true }]);
  await tick();
  undo(); // undoes the template load, not the selection
  await tick();
  redo();
  await tick();

  dragStart();
  for (let k = 1; k <= 5; k++) {
    useApp.getState().onNodesChange([{ type: "position", id, position: { x: before.x + k * 10, y: before.y } }]);
    await tick();
  }
  dragEnd();
  await tick();
  undo();
  await tick();
  assert.deepEqual(useApp.getState().nodes.find((n) => n.id === id).position, before);
});

import assert from "node:assert/strict";
import { test } from "node:test";
import { useApp } from "../src/store.js";

const folded = () => useApp.setState({ leftOpen: false, rightOpen: false, selectedId: null, preview: null });

test("clicking a block or picking one from the list opens the folded right panel", () => {
  folded();
  useApp.getState().select(useApp.getState().nodes[0].id);
  assert.equal(useApp.getState().rightOpen, true);
  folded();
  useApp.getState().previewBlock("dense");
  assert.equal(useApp.getState().rightOpen, true);
  assert.equal(useApp.getState().leftOpen, false);
});

test("clicking empty canvas keeps panels as they are", () => {
  folded();
  useApp.getState().select(null);
  assert.equal(useApp.getState().rightOpen, false);
});

test("a step tab opens the right panel on that step", () => {
  folded();
  useApp.getState().setPanel("train");
  const s = useApp.getState();
  assert.equal(s.rightOpen, true);
  assert.equal(s.panel, "train");
});

test("dropping a block on the canvas doesn't reopen a folded panel", () => {
  folded();
  useApp.getState().addBlock("dense", { x: 900, y: 900 }, false);
  assert.equal(useApp.getState().rightOpen, false);
});

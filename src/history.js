// Undo / redo for the model on the canvas (blocks, lines, settings, positions).
// It watches the app store instead of every action reporting in: whatever changed the
// model within one tick (an insert followed by its tidy) becomes one step. A block drag
// is one step too, taken when the drag ends. Opening another project starts afresh.
import { create } from "zustand";
import { useApp } from "./store.js";

const LIMIT = 100;

export const useHistory = create(() => ({ canUndo: false, canRedo: false }));

let past = [];
let future = [];
let stable = null; // the model as of the last step
let dragging = false;
let restoring = false;
let pending = false;

const snapshot = (s) => ({
  nodes: s.nodes.map(({ selected, dragging: _d, ...n }) => n),
  edges: s.edges.map(({ selected, ...e }) => e),
  templateId: s.templateId,
});

// What counts as a change: not selection, sizes React Flow measures, or the drop hint.
const keyOf = (snap) =>
  JSON.stringify([
    snap.nodes.map((n) => [n.id, n.data.type, n.data.params, Math.round(n.position.x), Math.round(n.position.y)]),
    snap.edges.map((e) => [e.id, e.source, e.target]),
  ]);

const publish = () => useHistory.setState({ canUndo: past.length > 0, canRedo: future.length > 0 });

function settle() {
  pending = false;
  if (dragging || restoring) return;
  const now = snapshot(useApp.getState());
  if (keyOf(now) === keyOf(stable)) return;
  past.push(stable);
  if (past.length > LIMIT) past.shift();
  future = [];
  stable = now;
  publish();
}

function reset() {
  past = [];
  future = [];
  stable = snapshot(useApp.getState());
  publish();
}

function restore(snap) {
  restoring = true;
  const s = useApp.getState();
  const ids = new Set(snap.nodes.map((n) => n.id));
  useApp.setState({
    nodes: snap.nodes,
    edges: snap.edges,
    templateId: snap.templateId,
    selectedId: ids.has(s.selectedId) ? s.selectedId : null,
    dropHint: null,
  });
  stable = snap;
  restoring = false;
  publish();
}

export function undo() {
  if (!past.length || useApp.getState().train.status === "training") return;
  future.push(stable);
  restore(past.pop());
}

export function redo() {
  if (!future.length || useApp.getState().train.status === "training") return;
  past.push(stable);
  restore(future.pop());
}

export function dragStart() {
  dragging = true;
}

export function dragEnd() {
  dragging = false;
  settle();
}

stable = snapshot(useApp.getState());
useApp.subscribe((s, prev) => {
  if (s.projectRevision !== prev.projectRevision) return reset();
  if (s.nodes === prev.nodes && s.edges === prev.edges && s.templateId === prev.templateId) return;
  if (!pending) {
    pending = true;
    queueMicrotask(settle);
  }
});

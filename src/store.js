import { create } from "zustand";
import { applyNodeChanges, applyEdgeChanges } from "@xyflow/react";
import { LAYERS } from "./model/layers.js";
import { analyze, buildChain, makeEdge, makeNode, newId, TEMPLATES, GAP_X, NODE_W } from "./model/graph.js";

const newClass = (name) => ({ id: newId("c"), name, samples: [] });

const startTemplate = TEMPLATES[0];

export const useApp = create((set, get) => ({
  // Data ------------------------------------------------------------------
  classes: [newClass("Class 1"), newClass("Class 2")],
  camClassId: null,

  addClass: () => set((s) => ({ classes: [...s.classes, newClass(`Class ${s.classes.length + 1}`)] })),
  renameClass: (id, name) => set((s) => ({ classes: s.classes.map((c) => (c.id === id ? { ...c, name } : c)) })),
  removeClass: (id) => set((s) => ({ classes: s.classes.filter((c) => c.id !== id), camClassId: s.camClassId === id ? null : s.camClassId })),
  addSamples: (id, samples) => set((s) => ({ classes: s.classes.map((c) => (c.id === id ? { ...c, samples: [...c.samples, ...samples] } : c)) })),
  removeSample: (id, sampleId) =>
    set((s) => ({ classes: s.classes.map((c) => (c.id === id ? { ...c, samples: c.samples.filter((x) => x.id !== sampleId) } : c)) })),
  clearSamples: (id) => set((s) => ({ classes: s.classes.map((c) => (c.id === id ? { ...c, samples: [] } : c)) })),
  setClasses: (classes) => set({ classes, camClassId: null }),
  setCamClass: (camClassId) => set({ camClassId }),

  // Model graph -------------------------------------------------------------
  templateId: startTemplate.id,
  ...buildChain(startTemplate.spec),
  selectedId: null,

  onNodesChange: (changes) => set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) => set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: ({ source, target }) => {
    if (source === target) return;
    // One way in, one way out: a new link replaces the old ones on both ends.
    set((s) => ({ templateId: null, edges: [...s.edges.filter((e) => e.source !== source && e.target !== target), makeEdge(source, target)] }));
  },
  select: (selectedId) => set({ selectedId }),

  loadTemplate: (id) => {
    const t = TEMPLATES.find((x) => x.id === id);
    set({ templateId: id, selectedId: null, ...buildChain(t.spec) });
  },

  // Adds a block. Dropped on a spot → placed there; clicked in the palette → slotted
  // in just before the output so the model stays connected.
  addBlock: (type, position) => {
    const s = get();
    if (position) {
      const node = makeNode(type, position);
      const edge = nearestEdge(s, position);
      if (edge) return get().insertOnEdge(edge.id, type, node);
      set({ nodes: [...s.nodes, node], selectedId: node.id, templateId: null });
      return;
    }
    const a = analyze(s.nodes, s.edges, s.classes.length);
    const last = s.nodes.find((n) => n.id === a.chain[a.chain.length - 1]);
    if (type !== "output" && last?.data.type === "output" && a.chain.length > 1) {
      const into = s.edges.find((e) => e.target === last.id);
      if (into) return get().insertOnEdge(into.id, type);
    }
    if (last) return get().appendAfter(last.id, type);
    const node = makeNode(type, { x: 0, y: 0 });
    set({ nodes: [...s.nodes, node], selectedId: node.id });
  },

  insertOnEdge: (edgeId, type, preset) => {
    const s = get();
    const edge = s.edges.find((e) => e.id === edgeId);
    if (!edge) return;
    const node = preset || makeNode(type, { x: 0, y: 0 });
    set({
      nodes: [...s.nodes, node],
      edges: [...s.edges.filter((e) => e.id !== edgeId), makeEdge(edge.source, node.id), makeEdge(node.id, edge.target)],
      selectedId: node.id,
      templateId: null,
    });
    get().tidy();
  },

  appendAfter: (nodeId, type) => {
    const s = get();
    const from = s.nodes.find((n) => n.id === nodeId);
    const node = makeNode(type, { x: from.position.x + GAP_X, y: from.position.y });
    const outs = s.edges.filter((e) => e.source === nodeId);
    set({
      nodes: [...s.nodes, node],
      edges: [...s.edges.filter((e) => e.source !== nodeId), makeEdge(nodeId, node.id), ...outs.map((e) => makeEdge(node.id, e.target))],
      selectedId: node.id,
      templateId: null,
    });
    get().tidy();
  },

  removeNode: (nodeId) => {
    const s = get();
    const node = s.nodes.find((n) => n.id === nodeId);
    if (!node || node.deletable === false) return;
    const into = s.edges.find((e) => e.target === nodeId);
    const outOf = s.edges.find((e) => e.source === nodeId);
    const edges = s.edges.filter((e) => e.source !== nodeId && e.target !== nodeId);
    if (into && outOf) edges.push(makeEdge(into.source, outOf.target));
    set({ nodes: s.nodes.filter((n) => n.id !== nodeId), edges, selectedId: s.selectedId === nodeId ? null : s.selectedId, templateId: null });
    get().tidy();
  },

  removeEdge: (edgeId) => set((s) => ({ templateId: null, edges: s.edges.filter((e) => e.id !== edgeId) })),

  setParam: (nodeId, key, value) =>
    set((s) => ({
      templateId: null,
      nodes: s.nodes.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } } : n)),
    })),

  replaceNode: (nodeId, type) =>
    set((s) => ({ templateId: null, nodes: s.nodes.map((n) => (n.id === nodeId ? { ...n, data: { type, params: { ...LAYERS[type].defaults } } } : n)) })),

  applyFix: (nodeId, action) => {
    const s = get();
    switch (action.kind) {
      case "insertBefore": {
        const into = s.edges.find((e) => e.target === nodeId);
        if (into) get().insertOnEdge(into.id, action.type);
        break;
      }
      case "remove":
        get().removeNode(nodeId);
        break;
      case "setParam":
        get().setParam(nodeId, action.key, action.value);
        break;
      case "setInputSize": {
        const input = s.nodes.find((n) => n.data.type === "input");
        if (input) get().setParam(input.id, "size", action.value);
        break;
      }
      case "replace":
        get().replaceNode(nodeId, action.type);
        break;
      case "append":
        get().appendAfter(nodeId, action.type);
        break;
      case "addInput": {
        const left = Math.min(0, ...s.nodes.map((n) => n.position.x)) - GAP_X;
        const input = makeNode("input", { x: left, y: 0 });
        const first = s.nodes.find((n) => !s.edges.some((e) => e.target === n.id));
        set({ nodes: [...s.nodes, input], edges: first ? [...s.edges, makeEdge(input.id, first.id)] : s.edges });
        get().tidy();
        break;
      }
    }
  },

  // Lines the connected blocks up in one row to the right of the input.
  tidy: () => {
    const s = get();
    const a = analyze(s.nodes, s.edges, s.classes.length);
    const head = s.nodes.find((n) => n.id === a.chain[0]);
    if (!head) return;
    const order = new Map(a.chain.map((id, i) => [id, i]));
    set({
      nodes: s.nodes.map((n) => (order.has(n.id) ? { ...n, position: { x: head.position.x + order.get(n.id) * GAP_X, y: head.position.y } } : n)),
    });
  },

  // Training ------------------------------------------------------------------
  settings: { epochs: 20, batchSize: 16, learningRate: 0.001 },
  setSetting: (key, value) => set((s) => ({ settings: { ...s.settings, [key]: value } })),

  train: { status: "idle", logs: [], message: "", epoch: 0 },
  setTrain: (patch) => set((s) => ({ train: { ...s.train, ...patch } })),
  trained: null,
  setTrained: (trained) => set({ trained }),

  // View ----------------------------------------------------------------------
  demoData: false,
  view: "main",
  setView: (view) => set({ view }),
}));

export const useAnalysis = () => {
  const nodes = useApp((s) => s.nodes);
  const edges = useApp((s) => s.edges);
  const n = useApp((s) => s.classes.length);
  return analyze(nodes, edges, n);
};

// An edge runs from the right middle of one block to the left middle of the next.
function nearestEdge(s, pos) {
  const byId = new Map(s.nodes.map((n) => [n.id, n]));
  const p = { x: pos.x + NODE_W / 2, y: pos.y + 45 };
  let best = null;
  let bestD = 70;
  for (const e of s.edges) {
    const a = byId.get(e.source),
      b = byId.get(e.target);
    if (!a || !b) continue;
    const ax = a.position.x + NODE_W,
      ay = a.position.y + (a.measured?.height ?? 90) / 2;
    const bx = b.position.x,
      by = b.position.y + (b.measured?.height ?? 90) / 2;
    const d = distToSegment(p.x, p.y, ax, ay, bx, by);
    if (d < bestD) {
      bestD = d;
      best = e;
    }
  }
  return best;
}

function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax,
    dy = by - ay;
  const len = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

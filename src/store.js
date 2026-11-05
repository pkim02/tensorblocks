import { create } from "zustand";
import { applyNodeChanges, applyEdgeChanges } from "@xyflow/react";
import { LAYERS } from "./model/layers.js";
import { defaultDataset } from "./data/dataset.js";
import { analyze, buildTemplate, makeEdge, makeNode, newId, sidePosition, TEMPLATES, GAP_X, NODE_W } from "./model/graph.js";

const newClass = (name) => ({ id: newId("c"), name, samples: [] });

// The app opens on the model that trains in the browser; the VLM example sits first in the list.
const startTemplate = TEMPLATES.find((t) => t.id === "tm");

export const useApp = create((set, get) => ({
  projectRevision: 0,
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
  ...buildTemplate(startTemplate),
  selectedId: null,
  // A palette block being read about before it goes on the canvas.
  preview: null,
  // While a palette block is dragged over the canvas: the line it would drop onto and whether it fits there.
  dropHint: null,
  setDropHint: (dropHint) => {
    const cur = get().dropHint;
    if (cur?.edgeId !== dropHint?.edgeId || cur?.state !== dropHint?.state) set({ dropHint });
  },

  onNodesChange: (changes) => set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) => set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: ({ source, target }) => {
    if (source === target) return;
    // One way in: a new link replaces the old one. A merge block (Concat, Add) keeps up to
    // two ways in. A block's output can fork, so its other links stay.
    set((s) => {
      const def = LAYERS[s.nodes.find((n) => n.id === target)?.data.type];
      const merge = def?.merge;
      let edges = s.edges.filter((e) => e.target !== target || (merge && e.source !== source));
      // Already as many ways in as it takes: keep the main line and swap out the side one.
      if (merge && edges.filter((e) => e.target === target).length >= (def.maxInputs ?? 2)) {
        const keep = mainInto({ ...s, edges }, target);
        edges = edges.filter((e) => e.target !== target || e === keep);
      }
      return { templateId: null, edges: [...edges, makeEdge(source, target)] };
    });
  },
  select: (selectedId) => set((s) => ({ selectedId, preview: null, panel: selectedId ? "block" : s.panel })),
  // Picking a block in the palette only explains it; "블록 추가하기" puts it on the canvas.
  previewBlock: (type) => set({ preview: type, selectedId: null, panel: "block" }),

  loadTemplate: (id) => {
    const t = TEMPLATES.find((x) => x.id === id);
    set({ templateId: id, selectedId: null, preview: null, dismissed: [], ...buildTemplate(t) });
  },

  // Adds a block that isn't wired to anything: the learner draws its lines. Dropped on a
  // line (snap), it slots in between instead. Returns the new block's id.
  addBlock: (type, position, snap = true) => {
    const s = get();
    const edge = snap && position && nearestEdge(s, position);
    if (edge) {
      const node = makeNode(type, position);
      get().insertOnEdge(edge.id, type, node);
      return node.id;
    }
    const node = makeNode(type, freeSpot(s.nodes, position ?? { x: 0, y: 0 }));
    set({ nodes: [...s.nodes, node], selectedId: node.id, preview: null, panel: "block", templateId: null });
    return node.id;
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
      preview: null,
      panel: "block",
      templateId: null,
    });
    get().tidy();
  },

  appendAfter: (nodeId, type) => {
    const s = get();
    const from = s.nodes.find((n) => n.id === nodeId);
    const node = makeNode(type, { x: from.position.x + GAP_X, y: from.position.y });
    const outs = s.edges.filter((e) => e.source === nodeId);
    // A fork keeps its branches; only a single link moves behind the new block.
    const moved = outs.length === 1 ? outs : [];
    set({
      nodes: [...s.nodes, node],
      edges: [...s.edges.filter((e) => !moved.includes(e)), makeEdge(nodeId, node.id), ...moved.map((e) => makeEdge(node.id, e.target))],
      selectedId: node.id,
      preview: null,
      panel: "block",
      templateId: null,
    });
    get().tidy();
  },

  removeNode: (nodeId) => {
    const s = get();
    const node = s.nodes.find((n) => n.id === nodeId);
    if (!node || node.deletable === false) return;
    set({ nodes: s.nodes.filter((n) => n.id !== nodeId), edges: edgesWithout(s, nodeId), selectedId: s.selectedId === nodeId ? null : s.selectedId, templateId: null });
    get().tidy();
  },

  // A block already on the canvas, dragged onto a line: it leaves its old spot (the
  // blocks around it join back up) and slots in between the line's two ends.
  moveOntoEdge: (nodeId, edgeId) => {
    const s = get();
    const edges = edgesWithout(s, nodeId);
    const edge = edges.find((e) => e.id === edgeId);
    if (!edge || edge.source === nodeId || edge.target === nodeId) return;
    set({
      edges: [...edges.filter((e) => e !== edge), makeEdge(edge.source, nodeId), makeEdge(nodeId, edge.target)],
      selectedId: nodeId,
      preview: null,
      panel: "block",
      templateId: null,
    });
    get().tidy();
  },

  // Correction notes the learner closed, by what they say; a changed problem shows again.
  dismissed: [],
  dismissIssue: (key) => set((s) => ({ dismissed: [...s.dismissed, key] })),
  showIssues: () => set({ dismissed: [] }),

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
        const main = mainInto(s, nodeId);
        const into = action.side ? s.edges.filter((e) => e.target === nodeId && e !== main)[action.sideIndex ?? 0] : main;
        if (into) get().insertOnEdge(into.id, action.type, action.params && makeNode(action.type, { x: 0, y: 0 }, action.params));
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
      case "addSide": {
        const text = makeNode("text", { x: 0, y: 0 });
        const emb = makeNode("embedding", { x: 0, y: 0 });
        set({ nodes: [...s.nodes, text, emb], edges: [...s.edges, makeEdge(text.id, emb.id), makeEdge(emb.id, nodeId)], templateId: null });
        get().tidy();
        break;
      }
    }
  },

  // Lines the connected blocks up in one row to the right of the input; a side branch
  // goes in the row below, ending just before the block it joins.
  tidy: () => {
    const s = get();
    const a = analyze(s.nodes, s.edges, s.classes.length);
    const head = s.nodes.find((n) => n.id === a.chain[0]);
    if (!head) return;
    const pos = new Map(a.chain.map((id, i) => [id, { x: head.position.x + i * GAP_X, y: head.position.y }]));
    for (const [mergeId, ids] of Object.entries(a.side)) {
      const at = a.chain.indexOf(mergeId);
      ids.forEach((id, k) => pos.set(id, sidePosition(head.position, at, ids.length, k)));
    }
    set({ nodes: s.nodes.map((n) => (pos.has(n.id) ? { ...n, position: pos.get(n.id) } : n)) });
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
  view: "canvas",
  setView: (view) => set({ view }),
  panel: "data",
  setPanel: (panel) => set({ panel }),

  // VLM data: `fields` says which columns hold the photo, question and answer once read.
  dataset: defaultDataset(),
  setDataset: (patch) => set((s) => ({ dataset: { ...s.dataset, ...patch } })),
}));

export const useAnalysis = () => {
  const nodes = useApp((s) => s.nodes);
  const edges = useApp((s) => s.edges);
  const n = useApp((s) => s.classes.length);
  return analyze(nodes, edges, n);
};

// An edge runs from the right middle of one block to the left middle of the next.
// The link into a block along the main line (a merge block has a second one from its side branch).
function mainInto(s, nodeId) {
  const into = s.edges.filter((e) => e.target === nodeId);
  if (into.length < 2) return into[0];
  const a = analyze(s.nodes, s.edges, s.classes.length);
  const i = a.chain.indexOf(nodeId);
  return into.find((e) => i > 0 && e.source === a.chain[i - 1]) || into[0];
}

export const issueKey = (issue) => `${issue.nodeId ?? ""}|${issue.title}|${issue.text}`;

// The lines left once a block is taken out: every outgoing link joins back up to what fed
// it, unless that link already exists (the other side of a skip).
export function edgesWithout(s, nodeId) {
  const into = mainInto(s, nodeId);
  const edges = s.edges.filter((e) => e.source !== nodeId && e.target !== nodeId);
  for (const out of s.edges.filter((e) => e.source === nodeId)) {
    if (into && !edges.some((e) => e.source === into.source && e.target === out.target)) edges.push(makeEdge(into.source, out.target));
  }
  return edges;
}

// Nudges a new block down until it doesn't sit on top of another one.
function freeSpot(nodes, pos) {
  const p = { ...pos };
  const hit = (q) => nodes.some((n) => Math.abs(n.position.x - q.x) < NODE_W + 16 && Math.abs(n.position.y - q.y) < (n.measured?.height ?? 90) + 24);
  for (let i = 0; i < 20 && hit(p); i++) p.y += 120;
  return p;
}

// `skip` leaves out the lines touching that block (the one being dragged).
export function nearestEdge(s, pos, skip) {
  const byId = new Map(s.nodes.map((n) => [n.id, n]));
  const p = { x: pos.x + NODE_W / 2, y: pos.y + 45 };
  let best = null;
  let bestD = 70;
  for (const e of s.edges) {
    if (skip && (e.source === skip || e.target === skip)) continue;
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

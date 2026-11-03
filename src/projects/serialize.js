import { LAYERS } from "../model/layers.js";
import { defaultDataset } from "../data/dataset.js";

export const SCHEMA_VERSION = 2;
export const MAX_BYTES = 700_000; // Leave room for Firestore metadata and field names.
const invalid = () => {
  throw new Error("프로젝트 데이터 형식이 올바르지 않아요.");
};
const record = (v) => v && typeof v === "object" && !Array.isArray(v);
const text = (v, max = 200) => typeof v === "string" && v.length > 0 && v.length <= max;
const id = (v) => text(v, 150) && !v.includes("/");
const finite = (v) => typeof v === "number" && Number.isFinite(v);

function serializeDataset(dataset = defaultDataset()) {
  if (!record(dataset) || !text(dataset.repo, 200) || !/^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)?$/.test(dataset.repo)) invalid();
  if (dataset.repo.split("/").some((s) => s === "." || s === "..")) invalid();
  for (const key of ["config", "split"]) if (dataset[key] !== null && !text(dataset[key], 200)) invalid();
  let fields = null;
  if (dataset.fields !== null) {
    if (!record(dataset.fields)) invalid();
    for (const key of ["image", "question", "answer"]) if (!text(dataset.fields[key], 200)) invalid();
    if (dataset.fields.list !== null && !text(dataset.fields.list, 200)) invalid();
    fields = Object.fromEntries(["image", "question", "answer", "list"].map((key) => [key, dataset.fields[key]]));
  }
  return { repo: dataset.repo, config: dataset.config, split: dataset.split, fields };
}

export function projectName(value) {
  if (typeof value !== "string" || !text(value.trim(), 100)) throw new Error("프로젝트 이름을 1~100자로 입력해 주세요.");
  return value.trim();
}

// Whitelist only durable values. Never walk canvases, TF models or React Flow internals.
export function serializeProject(state) {
  if (!record(state) || !Array.isArray(state.nodes) || !Array.isArray(state.edges) || !Array.isArray(state.classes)) invalid();
  if (state.nodes.length > 500 || state.edges.length > 1000 || state.classes.length > 100) invalid();
  const nodes = state.nodes.map((n) => {
    if (!record(n) || !id(n.id) || !record(n.position) || !finite(n.position.x) || !finite(n.position.y)) invalid();
    const type = n.data?.type;
    if (!Object.hasOwn(LAYERS, type) || !record(n.data.params)) invalid();
    const params = {};
    for (const [key, fallback] of Object.entries(LAYERS[type].defaults)) {
      const value = n.data.params[key];
      if (typeof value !== typeof fallback || (typeof value === "number" ? !finite(value) : typeof value === "string" && value.length > 4000)) invalid();
      params[key] = value;
    }
    for (const field of LAYERS[type].fields) {
      const value = params[field.key];
      if (field.options && !field.options.some((o) => o.value === value)) invalid();
      if (field.type === "number" && (!Number.isInteger(value) || value < field.min || value > field.max)) invalid();
    }
    return { id: n.id, position: { x: n.position.x, y: n.position.y }, data: { type, params } };
  });
  const ids = new Set(nodes.map((n) => n.id));
  if (ids.size !== nodes.length) invalid();
  const edges = state.edges.map((e) => {
    if (!record(e) || !id(e.id) || !ids.has(e.source) || !ids.has(e.target) || e.source === e.target) invalid();
    return { id: e.id, source: e.source, target: e.target };
  });
  if (new Set(edges.map((e) => e.id)).size !== edges.length) invalid();
  const classes = state.classes.map((c) => {
    if (!record(c) || !id(c.id) || !text(c.name)) invalid();
    return { id: c.id, name: c.name };
  });
  if (new Set(classes.map((c) => c.id)).size !== classes.length) invalid();
  const { epochs, batchSize, learningRate } = state.settings ?? {};
  if (
    !Number.isInteger(epochs) ||
    epochs < 1 ||
    epochs > 10000 ||
    !Number.isInteger(batchSize) ||
    batchSize < 1 ||
    batchSize > 10000 ||
    !finite(learningRate) ||
    learningRate <= 0 ||
    learningRate > 1
  )
    invalid();
  if (state.templateId !== null && !text(state.templateId, 100)) invalid();
  const result = {
    schemaVersion: SCHEMA_VERSION,
    templateId: state.templateId,
    nodes,
    edges,
    classes,
    settings: { epochs, batchSize, learningRate },
    dataset: serializeDataset(state.dataset),
  };
  if (new TextEncoder().encode(JSON.stringify(result)).length > MAX_BYTES)
    throw new Error("프로젝트가 너무 커서 저장할 수 없어요. 블록이나 긴 설정을 줄여 주세요.");
  return result;
}

export function restoreProject(data) {
  if (![1, SCHEMA_VERSION].includes(data?.schemaVersion)) throw new Error("지원하지 않는 프로젝트 버전이에요.");
  if (data.schemaVersion === SCHEMA_VERSION && !record(data.dataset)) invalid();
  const value = serializeProject(data);
  return {
    ...value,
    nodes: value.nodes.map((n) => ({ ...n, type: "block", deletable: n.data.type !== "input" })),
    edges: value.edges.map((e) => ({ ...e, type: "shape" })),
    classes: value.classes.map((c) => ({ ...c, samples: [] })),
  };
}

export const fingerprint = (state) => JSON.stringify(serializeProject(state));

// Include local-only data in overwrite detection, even though it is never uploaded.
export function localFingerprint(state) {
  return JSON.stringify([fingerprint(state), state.classes.map((c) => c.samples.map((s) => s.id)), !!state.trained, state.demoData]);
}

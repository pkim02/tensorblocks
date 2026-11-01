// Walks the canvas graph from the image input, works out the tensor shape on every
// connection, and collects the problems (with fixes) a beginner would otherwise hit as
// runtime errors.
import { LAYERS } from "./layers.js";

export const GAP_Y = 128;

let uid = 0;
export const newId = (prefix = "n") => `${prefix}${Date.now().toString(36)}${(uid++).toString(36)}`;

export function makeNode(type, position, params) {
  return {
    id: newId(),
    type: "block",
    position,
    deletable: type !== "input",
    data: { type, params: { ...LAYERS[type].defaults, ...params } },
  };
}

export const makeEdge = (source, target) => ({ id: newId("e"), source, target, type: "shape" });

// A template is a list of [type, params]. Lays it out top to bottom and wires it up.
export function buildChain(spec, origin = { x: 0, y: 0 }) {
  const nodes = spec.map(([type, params], i) => makeNode(type, { x: origin.x, y: origin.y + i * GAP_Y }, params));
  const edges = nodes.slice(1).map((n, i) => makeEdge(nodes[i].id, n.id));
  return { nodes, edges };
}

export const TEMPLATES = [
  {
    id: "tm",
    name: "Teachable Machine 방식",
    text: "사전학습된 MobileNet 위에 작은 층만 학습해요. 데이터가 적어도 잘 돼요.",
    spec: [["input", { size: 224 }], ["mobilenet"], ["gap"], ["dense", { units: 100 }], ["output"]],
  },
  {
    id: "cnn",
    name: "나만의 작은 CNN",
    text: "처음부터 직접 쌓은 합성곱 신경망이에요. 층을 바꿔 가며 실험해 보세요.",
    spec: [
      ["input", { size: 64 }],
      ["conv2d", { filters: 16 }],
      ["maxpool"],
      ["conv2d", { filters: 32 }],
      ["maxpool"],
      ["flatten"],
      ["dense", { units: 64 }],
      ["dropout"],
      ["output"],
    ],
  },
  {
    id: "broken",
    name: "실수 체험: Flatten 빠뜨리기",
    text: "입문자가 가장 많이 겪는 차원 에러를 일부러 만들어 둔 모델이에요. 빨간 선을 고쳐 보세요.",
    spec: [["input", { size: 64 }], ["conv2d", { filters: 16 }], ["maxpool"], ["dense", { units: 64 }], ["output"]],
  },
  {
    id: "blank",
    name: "빈 캔버스",
    text: "입력과 출력만 있어요. 사이를 직접 채워 보세요.",
    spec: [["input", { size: 64 }], ["output"]],
  },
];

const signatureMemo = { key: null, value: null };

export function analyze(nodes, edges, numClasses) {
  if (signatureMemo.key && signatureMemo.key.nodes === nodes && signatureMemo.key.edges === edges && signatureMemo.key.numClasses === numClasses) {
    return signatureMemo.value;
  }
  const value = analyzeUncached(nodes, edges, Math.max(numClasses, 2));
  signatureMemo.key = { nodes, edges, numClasses };
  signatureMemo.value = value;
  return value;
}

function analyzeUncached(nodes, edges, numClasses) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const outs = new Map();
  const ins = new Map();
  for (const e of edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    (outs.get(e.source) || outs.set(e.source, []).get(e.source)).push(e);
    (ins.get(e.target) || ins.set(e.target, []).get(e.target)).push(e);
  }

  const r = {
    chain: [],
    nodeInfo: {},
    edgeInfo: {},
    issues: [],
    totalParams: 0,
    trainableParams: 0,
    ok: false,
    exportOnly: false,
    numClasses,
    signature: "",
  };

  const inputs = nodes.filter((n) => n.data.type === "input");
  if (inputs.length === 0) {
    r.issues.push({ level: "error", title: "입력이 없어요", text: "'이미지 입력' 블록부터 놓아 주세요.", fixes: [{ label: "이미지 입력 추가", action: { kind: "addInput" } }] });
    markDetached(r, nodes);
    return r;
  }

  let cur = inputs[0];
  let shape = null;
  let blocked = false;
  let prev = null;
  const seen = new Set();

  while (cur) {
    if (seen.has(cur.id)) {
      r.issues.push({ level: "error", title: "고리가 생겼어요", text: "블록 연결이 한 바퀴 돌아 제자리로 와요. 위에서 아래로 한 줄로 이어 주세요.", fixes: [] });
      blocked = true;
      break;
    }
    seen.add(cur.id);
    r.chain.push(cur.id);

    const def = LAYERS[cur.data.type];
    const p = cur.data.params;
    const ctx = { numClasses, afterInput: prev?.data.type === "input" };
    const info = { inShape: shape, outShape: null, issue: null, params: 0, blocked };

    if (!blocked) {
      const res = def.infer(shape, p, ctx);
      if (res.error) {
        info.issue = { ...res.error, nodeId: cur.id };
        r.issues.push(info.issue);
        blocked = true;
      } else {
        info.outShape = res.out;
        if (res.warn) {
          info.issue = { ...res.warn, nodeId: cur.id };
          r.issues.push(info.issue);
        }
        info.params = shape || cur.data.type === "input" ? def.params(shape, p, ctx) : 0;
        r.totalParams += info.params;
        if (!def.frozen) r.trainableParams += info.params;
        if (def.exportOnly) r.exportOnly = true;
      }
    }
    r.nodeInfo[cur.id] = info;

    const incoming = prev && (ins.get(cur.id) || []).find((e) => e.source === prev.id);
    if (incoming) r.edgeInfo[incoming.id] = { shape: r.nodeInfo[prev.id].outShape, issue: info.issue, blocked: info.blocked };

    const nextEdges = outs.get(cur.id) || [];
    if (nextEdges.length > 1) {
      r.issues.push({ level: "warn", title: "갈래가 있어요", text: "아직은 한 줄 모델만 학습할 수 있어요. 첫 번째 연결만 따라가요.", nodeId: cur.id, fixes: [] });
    }
    if (cur.data.type === "output" && nextEdges.length) {
      r.issues.push({ level: "error", title: "출력 뒤에 블록이 있어요", text: "'분류 출력'은 맨 끝에 와야 해요.", nodeId: cur.id, fixes: [] });
      blocked = true;
      break;
    }
    shape = info.outShape;
    prev = cur;
    cur = nextEdges.length ? byId.get(nextEdges[0].target) : null;
  }

  const last = byId.get(r.chain[r.chain.length - 1]);
  if (last.data.type !== "output") {
    r.issues.push({
      level: "error",
      title: "끝이 없어요",
      text: "모델 맨 끝에 '분류 출력' 블록을 이어 줘야 클래스별 확률이 나와요.",
      nodeId: last.id,
      fixes: [{ label: "분류 출력 붙이기", action: { kind: "append", type: "output" } }],
    });
  }
  for (const n of r.chain.slice(1)) {
    if (byId.get(n).data.type === "input") {
      r.issues.push({ level: "error", title: "입력이 두 번 나와요", text: "'이미지 입력'은 맨 처음에 한 번만 와요.", nodeId: n, fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }] });
    }
  }

  markDetached(r, nodes);
  r.ok = !r.issues.some((i) => i.level === "error");
  r.signature = JSON.stringify([numClasses, r.chain.map((id) => [byId.get(id).data.type, byId.get(id).data.params])]);
  return r;
}

function markDetached(r, nodes) {
  for (const n of nodes) {
    if (!r.nodeInfo[n.id]) r.nodeInfo[n.id] = { detached: true };
  }
}

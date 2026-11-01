// Walks the canvas graph from the image input, works out the tensor shape on every
// connection, and collects the problems (with fixes) a beginner would otherwise hit as
// runtime errors. The main line runs image input → output; a merge block (Concat) also
// pulls in a side branch that starts at its own source (the text input of a VLM).
import { LAYERS } from "./layers.js";

// Blocks sit in one row, left to right. NODE_W must match .layer-block width in styles.css.
export const NODE_W = 196;
export const GAP_X = NODE_W + 100;

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

// A side branch sits one row below the main line and ends in the column just before
// the block it merges into.
export const SIDE_Y = 200;
export const sidePosition = (head, mergeIndex, count, k) => ({ x: head.x + (mergeIndex - count + k) * GAP_X, y: head.y + SIDE_Y });

// A template is a list of [type, params]. Lays it out left to right and wires it up.
export function buildChain(spec, origin = { x: 0, y: 0 }) {
  const nodes = spec.map(([type, params], i) => makeNode(type, { x: origin.x + i * GAP_X, y: origin.y }, params));
  const edges = nodes.slice(1).map((n, i) => makeEdge(nodes[i].id, n.id));
  return { nodes, edges };
}

// Template with an optional side branch: { spec, side: { into, spec } }.
export function buildTemplate(t) {
  const main = buildChain(t.spec);
  if (!t.side) return main;
  const n = t.side.spec.length;
  const side = t.side.spec.map(([type, params], k) => makeNode(type, sidePosition({ x: 0, y: 0 }, t.side.into, n, k), params));
  const edges = [...main.edges, ...side.slice(1).map((nd, k) => makeEdge(side[k].id, nd.id)), makeEdge(side[n - 1].id, main.nodes[t.side.into].id)];
  return { nodes: [...main.nodes, ...side], edges };
}

export const TEMPLATES = [
  {
    id: "vlm",
    name: "VLM: 사진 보고 답하기",
    text: "구글 PaliGemma 2처럼 사진(SigLIP)과 질문(Gemma)을 합쳐서 글로 답하는 비전-언어 모델이에요. 너무 커서 학습과 실행은 Colab에서 해요.",
    spec: [["input", { size: 224 }], ["siglip"], ["projector"], ["concat"], ["gemma"], ["textout"]],
    side: { into: 3, spec: [["text"], ["embedding"]] },
  },
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
    id: "vlm-broken",
    name: "실수 체험: VLM에서 Projector 빠뜨리기",
    text: "이미지 토큰(1,152칸)과 글 토큰(2,304칸)의 길이가 달라서 이어 붙일 수 없어요. 빨간 선을 고쳐 보세요.",
    spec: [["input", { size: 224 }], ["siglip"], ["concat"], ["gemma"], ["textout"]],
    side: { into: 2, spec: [["text"], ["embedding"]] },
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
    side: {}, // merge block id → its side branch, source first
    order: [], // every connected block, side branches just before the block they join
    nodeInfo: {},
    edgeInfo: {},
    issues: [],
    totalParams: 0,
    trainableParams: 0,
    ok: false,
    exportOnly: false,
    kind: "classifier",
    numClasses,
    signature: "",
  };

  const inputs = nodes.filter((n) => n.data.type === "input");
  if (inputs.length === 0) {
    r.issues.push({
      level: "error",
      title: "입력이 없어요",
      text: "'이미지 입력' 블록부터 놓아 주세요.",
      fixes: [{ label: "이미지 입력 추가", action: { kind: "addInput" } }],
    });
    markDetached(r, nodes);
    return r;
  }

  const seen = new Set();

  // Works out one block given what flows into it; records its issue and parameter count.
  const visit = (node, shape, blocked, ctx) => {
    const def = LAYERS[node.data.type];
    const p = node.data.params;
    const info = { inShape: shape, outShape: null, issue: null, params: 0, blocked };
    if (!blocked) {
      const res = def.infer(shape, p, ctx);
      if (res.error) {
        info.issue = { ...res.error, nodeId: node.id };
        r.issues.push(info.issue);
      } else {
        info.outShape = res.out;
        if (res.warn) {
          info.issue = { ...res.warn, nodeId: node.id };
          r.issues.push(info.issue);
        }
        info.params = shape || def.source ? def.params(shape, p, ctx) : 0;
        r.totalParams += info.params;
        if (!def.frozen) r.trainableParams += info.params;
        if (def.exportOnly) r.exportOnly = true;
      }
    }
    r.nodeInfo[node.id] = info;
    return info;
  };
  // `side` marks the second line into a merge block: it turns red too, but the note sits on the main line.
  const linkInfo = (edge, from, info, side = false) => {
    if (edge) r.edgeInfo[edge.id] = { shape: r.nodeInfo[from].outShape, issue: info.issue, blocked: info.blocked, side };
  };

  // Walks back up a merge block's second input to where that branch starts, then forward.
  const walkSide = (edge) => {
    if (seen.has(edge.source)) return null;
    const ids = [];
    let n = byId.get(edge.source);
    while (n && !seen.has(n.id)) {
      seen.add(n.id);
      ids.unshift(n.id);
      const up = ins.get(n.id) || [];
      n = up.length ? byId.get(up[0].source) : null;
    }
    let shape = null;
    let blocked = false;
    const first = byId.get(ids[0]);
    if (!LAYERS[first.data.type].source) {
      r.issues.push({
        level: "error",
        title: "글 쪽에 시작이 없어요",
        text: "합쳐지는 쪽 갈래도 맨 앞에 입력 블록(글 입력)이 있어야 해요.",
        nodeId: first.id,
        fixes: [],
      });
      blocked = true;
    }
    ids.forEach((id, k) => {
      const info = visit(byId.get(id), shape, blocked, { numClasses, afterInput: false });
      if (k > 0) linkInfo((ins.get(id) || [])[0], ids[k - 1], info);
      if (info.issue?.level === "error") blocked = true;
      shape = info.outShape;
    });
    return { ids, shape: blocked ? null : shape, blocked, edge, last: ids[ids.length - 1] };
  };

  let cur = inputs[0];
  let shape = null;
  let blocked = false;
  let prev = null;

  while (cur) {
    if (seen.has(cur.id)) {
      r.issues.push({
        level: "error",
        title: "고리가 생겼어요",
        text: "블록 연결이 한 바퀴 돌아 제자리로 와요. 왼쪽에서 오른쪽으로 한 줄로 이어 주세요.",
        fixes: [],
      });
      blocked = true;
      break;
    }
    seen.add(cur.id);
    r.chain.push(cur.id);

    const def = LAYERS[cur.data.type];
    const ctx = { numClasses, afterInput: prev?.data.type === "input" };
    const incoming = ins.get(cur.id) || [];
    let side = null;
    if (def.merge) {
      const sideEdge = incoming.find((e) => e.source !== prev?.id);
      side = sideEdge ? walkSide(sideEdge) : null;
      if (side) {
        r.side[cur.id] = side.ids;
        ctx.side = side;
      }
    }
    const info = visit(cur, shape, blocked || !!side?.blocked, ctx);
    if (info.issue?.level === "error" || side?.blocked) blocked = true;
    if (side) linkInfo(side.edge, side.last, info, true);
    if (prev)
      linkInfo(
        incoming.find((e) => e.source === prev.id),
        prev.id,
        info,
      );

    const nextEdges = outs.get(cur.id) || [];
    if (nextEdges.length > 1) {
      r.issues.push({
        level: "warn",
        title: "갈래가 있어요",
        text: "한 블록에서 두 갈래로 나뉘는 모델은 아직 못 그려요. 첫 번째 연결만 따라가요.",
        nodeId: cur.id,
        fixes: [],
      });
    }
    if (def.terminal && nextEdges.length) {
      r.issues.push({ level: "error", title: "출력 뒤에 블록이 있어요", text: `'${def.ko}'은(는) 맨 끝에 와야 해요.`, nodeId: cur.id, fixes: [] });
      blocked = true;
      break;
    }
    shape = info.outShape;
    prev = cur;
    cur = nextEdges.length ? byId.get(nextEdges[0].target) : null;
  }

  const last = byId.get(r.chain[r.chain.length - 1]);
  const vlm = r.chain.some((id) => LAYERS[byId.get(id).data.type].merge || byId.get(id).data.type === "textout");
  if (vlm) r.kind = "vlm";
  if (!LAYERS[last.data.type].terminal) {
    r.issues.push({
      level: "error",
      title: "끝이 없어요",
      text: vlm ? "VLM 맨 끝에 '글 출력' 블록을 이어 줘야 답이 나와요." : "모델 맨 끝에 '분류 출력' 블록을 이어 줘야 클래스별 확률이 나와요.",
      nodeId: last.id,
      fixes: [{ label: vlm ? "글 출력 붙이기" : "분류 출력 붙이기", action: { kind: "append", type: vlm ? "textout" : "output" } }],
    });
  }
  for (const n of r.chain.slice(1)) {
    if (byId.get(n).data.type === "input") {
      r.issues.push({
        level: "error",
        title: "입력이 두 번 나와요",
        text: "'이미지 입력'은 맨 처음에 한 번만 와요.",
        nodeId: n,
        fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
      });
    }
  }

  for (const id of r.chain) r.order.push(...(r.side[id] || []), id);
  markDetached(r, nodes);
  r.ok = !r.issues.some((i) => i.level === "error");
  r.signature = JSON.stringify([numClasses, r.order.map((id) => [byId.get(id).data.type, byId.get(id).data.params])]);
  return r;
}

function markDetached(r, nodes) {
  for (const n of nodes) {
    if (!r.nodeInfo[n.id]) r.nodeInfo[n.id] = { detached: true };
  }
}

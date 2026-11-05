// Walks the canvas graph from the image input, works out the tensor shape on every
// connection, and collects the problems (with fixes) a beginner would otherwise hit as
// runtime errors. The main line runs image input → output; a merge block (Concat, Add) also
// pulls in a side branch that starts at its own source (the text input of a VLM) or forks
// off an earlier block (a ResNet skip connection).
import { LAYERS } from "./layers.js";
import { josa } from "./terms.js";

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

// Template with optional side branches: { spec, side: { into, spec, from } } or sides: [...].
// `from` forks the branch off that main-line block; an empty spec is a bare skip connection.
export function buildTemplate(t) {
  const main = buildChain(t.spec);
  const nodes = [...main.nodes];
  const edges = [...main.edges];
  for (const b of t.sides ?? (t.side ? [t.side] : [])) {
    const n = b.spec.length;
    const side = b.spec.map(([type, params], k) => makeNode(type, sidePosition({ x: 0, y: 0 }, b.into, n, k), params));
    const path = [...(b.from === undefined ? [] : [main.nodes[b.from]]), ...side, main.nodes[b.into]];
    nodes.push(...side);
    edges.push(...path.slice(1).map((nd, k) => makeEdge(path[k].id, nd.id)));
  }
  return { nodes, edges };
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
    id: "resnet",
    name: "ResNet식 skip connection",
    text: "앞 블록의 출력을 건너뛰어 뒤 블록의 출력에 더해요. 깊은 모델도 학습이 잘 되는 비결이에요.",
    spec: [["input", { size: 64 }], ["conv2d", { filters: 16 }], ["conv2d", { filters: 16 }], ["add"], ["gap"], ["output"]],
    side: { into: 3, from: 1, spec: [] },
  },
  {
    id: "vit",
    name: "나만의 작은 ViT",
    text: "구글 ViT를 아주 작게 줄인 모델이에요. [LayerNorm → Self-Attention → 더하기]와 [LayerNorm → Dense 두 개 → 더하기]가 한 층이고, 그걸 두 번 쌓았어요. Transformer는 처음에 천천히 배워요. 학습률을 0.0003으로 낮추고 에폭을 60쯤으로 늘려 보세요.",
    spec: [
      ["input", { size: 64 }],
      ["patchembed", { patch: 8, dim: 64 }],
      ["posembed"],
      ["layernorm"],
      ["attention", { heads: 4 }],
      ["add"],
      ["layernorm"],
      ["dense", { units: 128, activation: "gelu" }],
      ["dense", { units: 64, activation: "linear" }],
      ["add"],
      ["layernorm"],
      ["attention", { heads: 4 }],
      ["add"],
      ["layernorm"],
      ["dense", { units: 128, activation: "gelu" }],
      ["dense", { units: 64, activation: "linear" }],
      ["add"],
      ["layernorm"],
      ["seqpool", { mode: "mean" }],
      ["output"],
    ],
    sides: [
      { into: 5, from: 2, spec: [] },
      { into: 9, from: 5, spec: [] },
      { into: 12, from: 9, spec: [] },
      { into: 16, from: 12, spec: [] },
    ],
  },
  {
    id: "convnext",
    name: "ConvNeXt 블록",
    text: "메타(Facebook)가 Transformer의 장점을 CNN에 옮겨 만든 ConvNeXt의 한 블록이에요. 큰 7×7 창으로 채널마다 훑고, LayerNorm과 GELU를 쓰고, 채널을 4배로 넓혔다가 되돌린 뒤 skip connection으로 더해요.",
    spec: [
      ["input", { size: 64 }],
      ["conv2d", { filters: 32, kernel: 3, stride: 2 }],
      ["depthwise", { kernel: 7, activation: "linear" }],
      ["layernorm"],
      ["conv2d", { filters: 128, kernel: 1, activation: "gelu" }],
      ["conv2d", { filters: 32, kernel: 1, activation: "linear" }],
      ["add"],
      ["gap"],
      ["output"],
    ],
    side: { into: 6, from: 1, spec: [] },
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
    id: "heads-broken",
    name: "실수 체험: 헤드 수가 칸 수로 안 나눠짐",
    text: "토큰 하나는 60칸인데 Self-Attention의 헤드는 8개예요. 60칸을 8묶음으로 똑같이 나눌 수 없어서 PyTorch에서 바로 에러가 나요. 빨간 선을 고쳐 보세요.",
    spec: [
      ["input", { size: 64 }],
      ["patchembed", { patch: 8, dim: 60 }],
      ["posembed"],
      ["layernorm"],
      ["attention", { heads: 8 }],
      ["add"],
      ["layernorm"],
      ["seqpool", { mode: "mean" }],
      ["output"],
    ],
    side: { into: 5, from: 2, spec: [] },
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
    order: [], // every connected block, inputs before the blocks they feed
    inputsOf: {}, // block id → the blocks feeding it, main line first
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

  // Which way a fork continues the main line: towards an output if it can, the longest way otherwise.
  const reach = new Map();
  const reachOf = (id) => {
    if (reach.has(id)) return reach.get(id);
    reach.set(id, 0); // a loop adds nothing
    const next = (outs.get(id) || []).map((e) => reachOf(e.target));
    const value = 1 + (next.length ? Math.max(...next) : LAYERS[byId.get(id).data.type].terminal ? 1000 : 0);
    reach.set(id, value);
    return value;
  };
  const nextLink = (id) => (outs.get(id) || []).reduce((best, e) => (!best || reachOf(e.target) > reachOf(best.target) ? e : best), null);

  for (let cur = inputs[0]; cur && !r.chain.includes(cur.id);) {
    r.chain.push(cur.id);
    const next = LAYERS[cur.data.type].terminal ? null : nextLink(cur.id);
    cur = next ? byId.get(next.target) : null;
  }

  // What feeds a block, main line first.
  const sourcesOf = (id) => {
    const main = r.chain[r.chain.indexOf(id) - 1];
    const sources = [...new Set((ins.get(id) || []).map((e) => e.source))];
    return [...sources.filter((s) => s === main), ...sources.filter((s) => s !== main)];
  };

  // Everything the last block depends on, inputs first: the main line in order, with side branches just before the block they join.
  const walking = new Set();
  const walked = new Set();
  let loopAt = null;
  const walk = (id) => {
    walking.add(id);
    for (const s of sourcesOf(id)) {
      if (walking.has(s)) loopAt ??= id;
      else if (!walked.has(s)) walk(s);
    }
    walking.delete(id);
    walked.add(id);
    r.order.push(id);
  };
  walk(r.chain[r.chain.length - 1]);
  if (loopAt) {
    r.issues.push({
      level: "error",
      title: "고리가 생겼어요",
      text: "블록 연결이 한 바퀴 돌아 제자리로 와요. 왼쪽에서 오른쪽으로 흐르게 이어 주세요.",
      nodeId: loopAt,
      fixes: [],
    });
  }

  // Works out one block given what flows into it; records its issue and parameter count.
  const visit = (node, shape, blocked, ctx, tooMany) => {
    const def = LAYERS[node.data.type];
    const p = node.data.params;
    const info = { inShape: shape, outShape: null, issue: null, params: 0, blocked };
    if (!blocked) {
      const res = tooMany ? { error: tooMany } : def.infer(shape, p, ctx);
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

  const firstPos = r.chain.find((id) => byId.get(id).data.type === "posembed");
  const clsWanted = r.order.some((id) => byId.get(id).data.type === "seqpool" && byId.get(id).data.params.mode === "cls");

  let branch = [];
  for (const id of r.order) {
    const node = byId.get(id);
    const def = LAYERS[node.data.type];
    const at = r.chain.indexOf(id);
    const sources = sourcesOf(id).filter((s) => r.nodeInfo[s]); // a loop's way back is left out
    const shapes = sources.map((s) => r.nodeInfo[s].outShape);
    let blocked = shapes.some((s) => !s);
    if (!def.source && !sources.length) {
      if (!loopAt) {
        r.issues.push({
          level: "error",
          title: "갈래에 시작이 없어요",
          text: "합쳐지는 쪽 갈래도 맨 앞에 입력 블록이 있거나 앞 블록에서 이어져야 해요.",
          nodeId: id,
          fixes: [],
        });
      }
      blocked = true;
    }
    const most = def.maxInputs ?? (def.merge ? 2 : 1);
    const tooMany =
      sources.length > most
        ? {
            level: "error",
            title: "입력이 너무 많아요",
            text: def.merge
              ? `${def.name}의 입력은 ${most}개까지예요.`
              : `${josa(def.name, "은는")} 입력을 하나만 받아요. 여러 갈래를 합치려면 사이에 Add 블록을 넣어 주세요.`,
            fixes: [],
          }
        : null;
    const ctx = { numClasses, afterInput: at > 0 && byId.get(r.chain[at - 1]).data.type === "input" };
    // CLS token: a 시퀀스 요약 set to CLS makes the first Position Embedding put one in front.
    if (node.data.type === "posembed") ctx.cls = clsWanted && id === firstPos;
    if (node.data.type === "seqpool") ctx.clsReady = firstPos !== undefined && (at < 0 || r.chain.indexOf(firstPos) < at);
    if (def.merge && sources.length > 1) ctx.side = { shape: shapes[1], shapes: shapes.slice(1) };
    const info = visit(node, shapes[0] ?? null, blocked, ctx, tooMany);
    info.inShapes = shapes;
    r.inputsOf[id] = sources;

    // `side` marks the other lines into a merge block: they turn red too, but the note sits on the main line.
    for (const e of ins.get(id) || []) {
      if (!sources.includes(e.source)) continue;
      const side = e.source !== sources[0];
      r.edgeInfo[e.id] = {
        shape: r.nodeInfo[e.source].outShape,
        issue: info.issue,
        blocked: info.blocked,
        side,
        skip: side && at >= 0 && r.chain.includes(e.source),
      };
    }

    if (at < 0) branch.push(id);
    else {
      if (branch.length) r.side[id] = branch;
      branch = [];
      if (def.terminal && (outs.get(id) || []).length) {
        r.issues.push({ level: "error", title: "출력 뒤에 블록이 있어요", text: `'${def.ko}'${josa(def.ko, "은는").slice(def.ko.length)} 맨 끝에 와야 해요.`, nodeId: id, fixes: [] });
      }
    }
  }

  // A fork has to come back together; a branch that never joins is left out of the model.
  for (const id of r.order) {
    if (!LAYERS[byId.get(id).data.type].terminal && (outs.get(id) || []).some((e) => !r.order.includes(e.target))) {
      r.issues.push({
        level: "warn",
        title: "갈래가 합쳐지지 않아요",
        text: "나뉜 갈래는 Add 블록으로 다시 합쳐야 해요. 합쳐지지 않은 갈래는 모델에서 빠져요.",
        nodeId: id,
        fixes: [],
      });
    }
  }

  const last = byId.get(r.chain[r.chain.length - 1]);
  const vlm = r.order.some((id) => ["concat", "textout"].includes(byId.get(id).data.type));
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
  for (const n of r.order.slice(1)) {
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

  markDetached(r, nodes);
  r.ok = !r.issues.some((i) => i.level === "error");
  r.signature = JSON.stringify([
    numClasses,
    r.order.map((id) => [byId.get(id).data.type, byId.get(id).data.params, r.inputsOf[id].map((s) => r.order.indexOf(s))]),
  ]);
  return r;
}

function markDetached(r, nodes) {
  for (const n of nodes) {
    if (!r.nodeInfo[n.id]) r.nodeInfo[n.id] = { detached: true };
  }
}

// Fit checks for a line the learner is about to draw ---------------------------------
// They only ask "do the shapes fit?" so the canvas can paint a spot green or red before
// the line exists. A wrong line is still allowed: the red note that follows is the lesson.

// What comes out of a block, following lines back even through blocks the main walk
// skipped (a block just added and only half wired up). null when it can't be known yet.
export function probeShape(id, nodes, edges, numClasses, a = analyze(nodes, edges, numClasses)) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const memo = new Map();
  const go = (nid) => {
    if (memo.has(nid)) return memo.get(nid);
    memo.set(nid, null); // a loop knows nothing
    const node = byId.get(nid);
    if (!node) return null;
    const known = a.nodeInfo[nid];
    if (known && !known.detached) {
      memo.set(nid, known.outShape ?? null);
      return known.outShape ?? null;
    }
    const def = LAYERS[node.data.type];
    const ins = edges.filter((e) => e.target === nid).map((e) => e.source);
    const shapes = ins.map(go);
    if ((!def.source && !ins.length) || shapes.some((s) => !s)) return null;
    const ctx = { numClasses: Math.max(numClasses, 2), afterInput: byId.get(ins[0])?.data.type === "input" };
    if (def.merge && shapes.length > 1) ctx.side = { shape: shapes[1], shapes: shapes.slice(1) };
    const res = def.infer(shapes[0] ?? null, node.data.params, ctx);
    const out = res.error ? null : res.out;
    memo.set(nid, out);
    return out;
  };
  return go(id);
}

// Runs one block on the shapes that would reach it: "ok", "bad", or "unknown".
function fits(def, params, shapes, ctx) {
  if (shapes.some((s) => !s)) return { state: "unknown" };
  // A merge block with one way in is only waiting for its second line.
  if (def.merge && shapes.length < 2) return { state: "ok", out: null };
  if (def.merge) ctx.side = { shape: shapes[1], shapes: shapes.slice(1) };
  const res = def.infer(shapes[0], params, ctx);
  return res.error ? { state: "bad", issue: res.error } : { state: "ok", out: res.out };
}

// The link into a block along the main line (a merge block has a second one from its side branch).
export function mainLineInto(nodes, edges, numClasses, nodeId) {
  const into = edges.filter((e) => e.target === nodeId);
  if (into.length < 2) return into[0];
  const a = analyze(nodes, edges, numClasses);
  const i = a.chain.indexOf(nodeId);
  return into.find((e) => i > 0 && e.source === a.chain[i - 1]) || into[0];
}

// The lines after drawing source → target. One way in: a new link replaces the old one.
// A merge block (Concat, Add) keeps up to maxInputs ways in; when it is full, the main line
// stays and a side one makes room. A block's output can fork, so its other links stay.
export function connectEdges(nodes, edges, numClasses, source, target) {
  const def = LAYERS[nodes.find((n) => n.id === target)?.data.type];
  const merge = def?.merge;
  let next = edges.filter((e) => e.target !== target || (merge && e.source !== source));
  if (merge && next.filter((e) => e.target === target).length >= (def.maxInputs ?? 2)) {
    const keep = mainLineInto(nodes, next, numClasses, target);
    next = next.filter((e) => e.target !== target || e === keep);
  }
  return [...next, makeEdge(source, target)];
}

// Would a line source → target fit? It draws the line the way onConnect will (connectEdges)
// and reads the target's inputs, main line first, the way analyze will see them.
export function checkConnect(sourceId, targetId, nodes, edges, numClasses) {
  if (sourceId === targetId) return "bad";
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const src = byId.get(sourceId);
  const tgt = byId.get(targetId);
  if (!src || !tgt) return "unknown";
  const sdef = LAYERS[src.data.type];
  const def = LAYERS[tgt.data.type];
  if (sdef.terminal || def.source) return "bad";
  const after = connectEdges(nodes, edges, numClasses, sourceId, targetId);
  // A line back into a block that already feeds the source makes a loop ("고리가 생겼어요").
  const upstream = new Set();
  for (const stack = [sourceId]; stack.length;) {
    const id = stack.pop();
    if (id === targetId) return "bad";
    if (upstream.has(id)) continue;
    upstream.add(id);
    for (const e of after) if (e.target === id) stack.push(e.source);
  }
  const a = analyze(nodes, after, numClasses);
  // A block not on the walk yet (half wired) has no order from analyze: take the lines as they are.
  const info = a.nodeInfo[targetId];
  const sources = info && !info.detached && a.inputsOf[targetId] ? a.inputsOf[targetId] : after.filter((e) => e.target === targetId).map((e) => e.source);
  const shapes = sources.map((id) => probeShape(id, nodes, after, numClasses, a));
  const afterInput = byId.get(sources[0])?.data.type === "input";
  return fits(def, tgt.data.params, shapes, { numClasses: Math.max(numClasses, 2), afterInput }).state;
}

// Would dropping a new `type` block onto this line fit on both sides?
export function checkInsert(edgeId, type, nodes, edges, numClasses, params) {
  const edge = edges.find((e) => e.id === edgeId);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const src = edge && byId.get(edge.source);
  const tgt = edge && byId.get(edge.target);
  if (!src || !tgt) return "unknown";
  const def = LAYERS[type];
  if (def.source || def.terminal) return "bad";
  const n = Math.max(numClasses, 2);
  const a = analyze(nodes, edges, numClasses);
  const shape = probeShape(src.id, nodes, edges, numClasses, a);
  const first = fits(def, { ...def.defaults, ...params }, [shape], { numClasses: n, afterInput: src.data.type === "input" });
  if (first.state !== "ok" || !first.out) return first.state === "ok" ? "unknown" : first.state;
  const tdef = LAYERS[tgt.data.type];
  const into = edges.filter((e) => e.target === tgt.id);
  const shapes = into.map((e) => (e.id === edgeId ? first.out : probeShape(e.source, nodes, edges, numClasses, a)));
  return fits(tdef, tgt.data.params, shapes, { numClasses: n, afterInput: false }).state;
}

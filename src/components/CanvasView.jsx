import { useCallback, useEffect, useRef } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Handle,
  Position,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useConnection,
  useNodesInitialized,
  useReactFlow,
} from "@xyflow/react";
import { useApp, useAnalysis, nearestEdge, edgesWithout } from "../store.js";
import { LAYERS, PALETTE, fmtShape, fmtShapeLong, fmtCount, torchLines, torchShape } from "../model/layers.js";
import { TEMPLATES, NODE_W, checkConnect, checkInsert } from "../model/graph.js";

import AuthButton from "./AuthButton.jsx";
import ProjectControls from "./ProjectControls.jsx";
import IssueNote from "./IssueNote.jsx";
import ViewTabs from "./ViewTabs.jsx";
import WorkPanel from "./WorkPanel.jsx";
import { fieldPath } from "../data/hf.js";
import { Glossed } from "./Term.jsx";
import { useHistory, undo, redo, dragStart, dragEnd } from "../history.js";

// What the numbers on a line stand for, shown on hover.
const shapeTip = (s) => (!s ? undefined : s.length === 3 ? "세로×가로×채널" : s.length === 2 ? `토큰 개수×토큰 하나의 길이: ${fmtShapeLong(s)}` : "한 줄짜리 벡터");

const DRAG_TYPE = "application/x-tensorblock";
// dataTransfer can't be read during dragover, so the palette notes what is being dragged.
let draggingType = null;

// Would a line between these two blocks fit? The dragged end may be either side.
function connectState(fromId, fromType, otherId) {
  const s = useApp.getState();
  const [source, target] = fromType === "target" ? [otherId, fromId] : [fromId, otherId];
  return checkConnect(source, target, s.nodes, s.edges, s.classes.length);
}

export default function CanvasView() {
  const nodes = useApp((s) => s.nodes);
  const edges = useApp((s) => s.edges);
  const onNodesChange = useApp((s) => s.onNodesChange);
  const onEdgesChange = useApp((s) => s.onEdgesChange);
  const onConnect = useApp((s) => s.onConnect);
  const select = useApp((s) => s.select);
  const addBlock = useApp((s) => s.addBlock);
  const tidy = useApp((s) => s.tidy);
  const templateId = useApp((s) => s.templateId);
  const loadTemplate = useApp((s) => s.loadTemplate);
  const training = useApp((s) => s.train.status === "training");
  const projectRevision = useApp((s) => s.projectRevision);
  const leftOpen = useApp((s) => s.leftOpen);
  const rightOpen = useApp((s) => s.rightOpen);
  const a = useAnalysis();
  const rf = useReactFlow();

  const errors = a.issues.filter((i) => i.level === "error").length;
  const warns = a.issues.length - errors;

  // Delete / Backspace removes the selected block and joins its neighbours back up.
  // Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) step back and forward.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("dialog[open]")) return;
      if (e.target.closest("input, select, textarea, [contenteditable]")) return;
      const key = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (key === "z" || key === "y")) {
        e.preventDefault();
        if (key === "y" || e.shiftKey) redo();
        else undo();
        return;
      }
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const s = useApp.getState();
      if (s.selectedId) s.removeNode(s.selectedId);
      else s.edges.filter((x) => x.selected).forEach((x) => s.removeEdge(x.id));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const flowRef = useRef(null);

  // Frames the row at a readable size: the whole model if it fits, otherwise starting
  // from the input with the rest a sideways pan away. Sits a little high so the
  // correction notes hanging under a line have room.
  const fit = useCallback(
    () =>
      setTimeout(() => {
        const el = flowRef.current;
        const ns = rf.getNodes();
        if (!el || !ns.length) return;
        const b = rf.getNodesBounds(ns);
        const { leftOpen, rightOpen } = useApp.getState();
        // A folded panel leaves a small tab on that edge; keep the blocks clear of it.
        const pad = leftOpen && rightOpen ? 32 : 96;
        const zoom = Math.max(0.45, Math.min(1, (el.clientWidth - 2 * pad) / b.width));
        const fits = b.width * zoom <= el.clientWidth - 2 * pad;
        const x = fits ? (el.clientWidth - b.width * zoom) / 2 - b.x * zoom : pad - b.x * zoom;
        const y = el.clientHeight * (leftOpen || rightOpen ? 0.38 : 0.48) - (b.y + b.height / 2) * zoom;
        rf.setViewport({ x, y, zoom }, { duration: 200 });
      }, 60),
    [rf],
  );

  // Pans to the first problem so it is never off-screen during a demo, and brings back
  // any correction notes the learner closed.
  const showProblem = () => {
    useApp.getState().showIssues();
    const issue = a.issues.find((i) => i.level === "error" && i.nodeId) || a.issues.find((i) => i.nodeId);
    const n = issue && rf.getNode(issue.nodeId);
    if (n) rf.setCenter(n.position.x, n.position.y + 120, { zoom: Math.max(rf.getZoom(), 0.9), duration: 300 });
  };

  // Re-frame when blocks are added or removed, except for a loose block the learner just
  // placed in view: jumping away from it would lose it.
  const keepView = useRef(false);
  const count = nodes.length;
  useEffect(() => {
    if (keepView.current) keepView.current = false;
    else fit();
  }, [count, fit, projectRevision, leftOpen, rightOpen]);
  // The first frame runs before the blocks have sizes, which left the last block cut off.
  const measured = useNodesInitialized();
  useEffect(() => {
    if (measured) fit();
  }, [measured, fit]);

  // A block already on the canvas, dragged over a line: same green/red hint as a palette
  // drag, and letting go slots it in between. Input and output blocks only ever sit at the ends.
  const onNodeDrag = (_, node) => {
    const s = useApp.getState();
    const def = LAYERS[node.data.type];
    if (def.source || def.terminal) return;
    // Only lines drawn on screen now; the fit is judged as if the block had already left its spot.
    const edge = nearestEdge(s, node.position, node.id);
    const rest = edgesWithout(s, node.id);
    s.setDropHint(edge ? { edgeId: edge.id, nodeId: node.id, state: checkInsert(edge.id, node.data.type, s.nodes, rest, s.classes.length, node.data.params) } : null);
  };

  const onNodeDragStop = (_, node) => {
    const s = useApp.getState();
    const hint = s.dropHint;
    s.setDropHint(null);
    if (hint?.nodeId === node.id) s.moveOntoEdge(node.id, hint.edgeId);
    dragEnd();
  };

  const dropPoint = (e) => {
    const p = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY });
    return { x: p.x - NODE_W / 2, y: p.y - 45 };
  };

  // While a palette block hovers over a line, paint that line green (fits) or red (doesn't).
  const onDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (!draggingType) return;
    const s = useApp.getState();
    const edge = nearestEdge(s, dropPoint(e));
    s.setDropHint(edge ? { edgeId: edge.id, state: checkInsert(edge.id, draggingType, s.nodes, s.edges, s.classes.length) } : null);
  };

  const onDrop = (e) => {
    const type = e.dataTransfer.getData(DRAG_TYPE);
    useApp.getState().setDropHint(null);
    if (!type) return;
    e.preventDefault();
    const at = dropPoint(e);
    keepView.current = !nearestEdge(useApp.getState(), at);
    addBlock(type, at);
  };

  // "블록 추가하기": a loose block in the empty space under the row, in view.
  const addFromPreview = (type) => {
    const el = flowRef.current;
    if (!el) return addBlock(type);
    const r = el.getBoundingClientRect();
    const p = rf.screenToFlowPosition({ x: r.left + r.width / 2, y: r.top + r.height * 0.68 });
    keepView.current = true;
    addBlock(type, { x: p.x - NODE_W / 2, y: p.y - 45 }, false);
  };

  return (
    <div className={`canvas-view${leftOpen || rightOpen ? "" : " simple"}`}>
      <header className="work-bar">
        <ViewTabs />
        <h1 className="visually-hidden">블록 캔버스</h1>
        <select
          className="bar-extra"
          aria-label="모델 틀 불러오기"
          value={templateId ?? ""}
          disabled={training}
          onChange={(e) => {
            loadTemplate(e.target.value);
            fit();
          }}
        >
          {templateId == null && <option value="">직접 고친 모델</option>}
          {TEMPLATES.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
        <span className="grow" />
        {errors || warns ? (
          <button className={`badge badge-btn ${errors ? "danger" : "warning"}`} onClick={showProblem} role="status">
            {errors ? `고칠 곳 ${errors}개` : `주의 ${warns}개`} 보기
          </button>
        ) : (
          <span className="badge success bar-extra" role="status">
            모양이 모두 맞아요
          </span>
        )}
        <span className="bar-rest bar-extra">
          <span className="small muted bar-params">
            학습할 값 <span className="num-inline">{fmtCount(a.trainableParams)}</span>개
          </span>
          <button
            className="btn outline small"
            onClick={() => {
              tidy();
              fit();
            }}
          >
            한 줄로 정렬
          </button>
          <ProjectControls />
          <AuthButton />
        </span>
      </header>

      <div className="canvas-body">
        <Palette />
        <div
          className="flow"
          ref={flowRef}
          onDragOver={onDragOver}
          onDragLeave={(e) => !e.currentTarget.contains(e.relatedTarget) && useApp.getState().setDropHint(null)}
          onDrop={onDrop}
        >
          <UndoRedo disabled={training} />
          <FoldedTabs leftOpen={leftOpen} rightOpen={rightOpen} />
          <ReactFlow
            nodes={nodes}
            edges={edges.map((e) => (training ? { ...e, className: "flowing" } : e))}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDrag={onNodeDrag}
            onNodeDragStart={dragStart}
            onNodeDragStop={onNodeDragStop}
            connectionLineComponent={FitConnectionLine}
            onNodeClick={(_, n) => select(n.id)}
            onPaneClick={() => select(null)}
            onEdgeClick={() => select(null)}
            deleteKeyCode={null}
            panOnScroll
            minZoom={0.2}
            maxZoom={1.6}
            colorMode="light"
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Lines} gap={24} className="graph-paper" />
            <Controls showInteractive={false} fitViewOptions={{ padding: 0.1 }} />
            <MiniMap pannable zoomable ariaLabel="모델 전체 지도" nodeStrokeWidth={2} />
          </ReactFlow>
        </div>
        <WorkPanel inspector={<Inspector onAdd={addFromPreview} />} />
      </div>
    </div>
  );
}

// What's left of a folded panel: a small tab on that edge of the canvas that opens it again.
function FoldedTabs({ leftOpen, rightOpen }) {
  const setLeftOpen = useApp((s) => s.setLeftOpen);
  const setPanel = useApp((s) => s.setPanel);
  return (
    <>
      {!leftOpen && (
        <div className="edge-tabs edge-left">
          <button onClick={() => setLeftOpen(true)}>블록 목록</button>
        </div>
      )}
      {!rightOpen && (
        <div className="edge-tabs edge-right" role="group" aria-label="작업 단계 열기">
          <button onClick={() => setPanel("data")}>1 데이터</button>
          <button onClick={() => setPanel("train")}>2 학습</button>
          <button onClick={() => setPanel("test")}>3 시험</button>
        </div>
      )}
    </>
  );
}

function Palette() {
  const preview = useApp((s) => s.preview);
  const previewBlock = useApp((s) => s.previewBlock);
  const open = useApp((s) => s.leftOpen);
  const setLeftOpen = useApp((s) => s.setLeftOpen);
  return (
    <aside className={`palette${open ? "" : " panel-closed"}`} aria-label="블록 상자">
      <div className="palette-top">
        <h2 className="palette-title">블록 목록</h2>
        <button className="panel-fold" aria-label="블록 목록 접기" title="블록 목록 접기" onClick={() => setLeftOpen(false)}>
          ‹
        </button>
      </div>
      {PALETTE.map((g) => (
        <div key={g.title} className="palette-group">
          <h2 className="palette-h">{g.title}</h2>
          <ul role="list">
            {g.items.map((t) => (
              <li key={t}>
                <button
                  className="palette-item"
                  aria-pressed={preview === t}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(DRAG_TYPE, t);
                    e.dataTransfer.effectAllowed = "copy";
                    draggingType = t;
                  }}
                  onDragEnd={() => {
                    draggingType = null;
                    useApp.getState().setDropHint(null);
                  }}
                  onClick={() => previewBlock(t)}
                >
                  <span className="pi-name">{LAYERS[t].name}</span>
                  <span className="pi-ko">{LAYERS[t].ko}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </aside>
  );
}

function UndoRedo({ disabled }) {
  const canUndo = useHistory((h) => h.canUndo);
  const canRedo = useHistory((h) => h.canRedo);
  return (
    <div className="undo-redo" role="group" aria-label="되돌리기와 다시 하기">
      <button onClick={undo} disabled={disabled || !canUndo} aria-label="되돌리기" title="되돌리기 (Ctrl+Z)">
        ↶
      </button>
      <button onClick={redo} disabled={disabled || !canRedo} aria-label="다시 하기" title="다시 하기 (Ctrl+Shift+Z)">
        ↷
      </button>
    </div>
  );
}

// The line being drawn turns green over a block it fits, red over one it doesn't.
function FitConnectionLine({ fromX, fromY, toX, toY, fromPosition, toPosition, fromNode, fromHandle, toNode }) {
  const state = toNode && fromNode && toNode.id !== fromNode.id ? connectState(fromNode.id, fromHandle?.type, toNode.id) : null;
  const [path] = getBezierPath({ sourceX: fromX, sourceY: fromY, sourcePosition: fromPosition, targetX: toX, targetY: toY, targetPosition: toPosition });
  return <path d={path} fill="none" className={`conn-line conn-line-${state ?? "idle"}`} />;
}

const FIT_LABEL = { ok: "모양이 맞아요", bad: "모양이 안 맞아요", unknown: "아직 몰라요" };

const DATA_PINS = {
  input: (f) => `데이터: ${f.image}`,
  text: (f) => `데이터: ${fieldPath(f, "question")}`,
  textout: (f) => `정답: ${fieldPath(f, "answer")}`,
};

function BlockNode({ id, data }) {
  const a = useAnalysis();
  const selected = useApp((s) => s.selectedId === id);
  const def = LAYERS[data.type];
  // React Flow can paint a block one frame before the analysis knows about it.
  const info = a.nodeInfo[id] || { detached: true };
  const level = info.issue?.level;
  const fields = useApp((s) => s.dataset.fields);
  // Where the VLM's dataset columns enter the model.
  const pin = a.kind === "vlm" && fields && DATA_PINS[data.type]?.(fields);
  // Problems that belong to this block but have no incoming line to sit on.
  const extra = a.issues.filter((i) => i.nodeId === id && i !== info.issue);
  // While a line is being drawn, every block it could end on says whether the shapes fit.
  const drawing = useConnection((c) => (c.inProgress ? { from: c.fromNode?.id, end: c.fromHandle?.type } : null));
  const canEnd = drawing && drawing.from !== id && (drawing.end === "target" ? !def.terminal : !def.source);
  // A block being dragged over a line shows the same colour as the line, which it covers.
  const dropFit = useApp((s) => (s.dropHint?.nodeId === id ? s.dropHint.state : null));
  const fit = canEnd ? connectState(drawing.from, drawing.end, id) : dropFit;
  const cls = [
    "layer-block",
    `g-${def.group}`,
    selected && "selected",
    level === "error" && "has-error",
    info.detached && "detached",
    info.blocked && "blocked",
    fit && `fit-${fit}`,
  ];

  return (
    <div className={cls.filter(Boolean).join(" ")}>
      {pin && <span className="data-pin">{pin}</span>}
      {canEnd && fit && <span className={`fit-tag fit-tag-${fit}`}>{FIT_LABEL[fit]}</span>}
      {!def.source && <Handle type="target" position={Position.Left} />}
      <div className="block-head">
        <span className="block-name">{def.name}</span>
        <span className="block-ko">{def.ko}</span>
      </div>
      <div className="block-sum">{def.summary(data.params, { numClasses: a.numClasses })}</div>
      <div className="block-foot">
        {info.detached
          ? "연결 안 됨"
          : info.blocked
            ? "앞에서 막혔어요"
            : def.frozen
              ? `고정된 값 ${fmtCount(info.params)}`
              : info.params
                ? `학습할 값 ${fmtCount(info.params)}`
                : "학습할 값 없음"}
        {def.terminal && info.outShape && <span className="mono"> → {fmtShape(info.outShape)}</span>}
      </div>
      {!def.terminal && <Handle type="source" position={Position.Right} />}
      {extra.map((issue, i) => (
        <IssueNote key={i} issue={issue} closable className="note-side nodrag" />
      ))}
    </div>
  );
}

function ShapeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected }) {
  const a = useAnalysis();
  const info = a.edgeInfo[id];
  // A skip connection arcs over the blocks it jumps past, which sit in the same row.
  const [path, lx, ly] = info?.skip
    ? [
        `M ${sourceX} ${sourceY} C ${sourceX + 80} ${sourceY - 170}, ${targetX - 80} ${targetY - 170}, ${targetX} ${targetY}`,
        (sourceX + targetX) / 2,
        (sourceY + targetY) / 2 - 128,
      ]
    : getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const level = info?.issue && !info.blocked ? info.issue.level : null;
  const state = !info ? "idle" : (level ?? (info.blocked ? "blocked" : "ok"));
  const drop = useApp((s) => (s.dropHint?.edgeId === id ? s.dropHint.state : null));
  const removeEdge = useApp((s) => s.removeEdge);
  const training = useApp((s) => s.train.status === "training");

  return (
    <>
      <BaseEdge id={id} path={path} className={`edge-${state} ${selected ? "edge-selected" : ""} ${drop ? `edge-drop-${drop}` : ""}`} />
      <EdgeLabelRenderer>
        {drop && (
          <div className={`fit-tag fit-tag-${drop} edge-fit nopan`} style={{ transform: `translate(-50%, -100%) translate(${lx}px, ${ly - 16}px)` }}>
            {drop === "ok" ? "여기에 끼울 수 있어요" : drop === "bad" ? "모양이 안 맞아요" : "아직 몰라요"}
          </div>
        )}
        {selected && !drop && !training && (
          <button
            className="edge-cut nodrag nopan"
            style={{ transform: `translate(-50%, -100%) translate(${lx}px, ${ly - 16}px)` }}
            onClick={(e) => {
              e.stopPropagation();
              removeEdge(id);
            }}
          >
            ✂ 연결 끊기
          </button>
        )}
        <div className={`shape-tag tag-${state} nodrag nopan`} title={info && !info.blocked ? shapeTip(info.shape) : undefined} style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}>
          {!info ? "연결 안 됨" : info.blocked ? "?" : fmtShape(info.shape)}
        </div>
        {level && !info.side && (
          <IssueNote issue={info.issue} closable className="note-edge nodrag nopan" style={{ transform: `translate(-50%, 0) translate(${lx}px, ${ly + 66}px)` }} />
        )}
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes = { block: BlockNode };
const edgeTypes = { shape: ShapeEdge };

function Inspector({ onAdd }) {
  const selectedId = useApp((s) => s.selectedId);
  const node = useApp((s) => s.nodes.find((n) => n.id === s.selectedId));
  const preview = useApp((s) => s.preview);
  const setParam = useApp((s) => s.setParam);
  const removeNode = useApp((s) => s.removeNode);
  const training = useApp((s) => s.train.status === "training");
  const a = useAnalysis();

  if (!selectedId && preview && LAYERS[preview]) {
    const def = LAYERS[preview];
    return (
      <>
        <p className="small muted insp-kicker">블록 상자에서 고른 블록</p>
        <h2 className="insp-h">
          {def.name} <span className="muted insp-ko">{def.ko}</span>
        </h2>
        <p className="small">
        <Glossed text={def.about} />
      </p>
        {def.fields.length > 0 && (
          <>
            <h3 className="insp-sub">바꿀 수 있는 값</h3>
            <ul className="small tips">
              {def.fields.map((f) => (
                <li key={f.key}>
                  {f.label}
                  {f.hint && (
                    <span className="muted">
                      {" "}
                      — <Glossed text={f.hint} />
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
        <button className="btn primary block" disabled={training} onClick={() => onAdd(preview)}>
          블록 추가하기
        </button>
        <p className="small muted">
          아무 데도 연결되지 않은 채로 캔버스에 놓여요. 블록 양쪽의 점을 끌어 다른 블록과 직접 이어 보세요. 이을 수 있는 곳은 초록색, 모양이 안 맞는 곳은
          빨간색으로 보여요.
        </p>
      </>
    );
  }

  if (!selectedId || !node) {
    const loose = a.issues.filter((i) => !i.nodeId);
    return (
      <>
        <h2 className="insp-h">모델 한눈에</h2>
        <table className="table small">
          <tbody>
            <tr>
              <td>블록</td>
              <td className="num">{a.order.length}개</td>
            </tr>
            <tr>
              <td>학습할 값</td>
              <td className="num">{a.trainableParams.toLocaleString("en-US")}</td>
            </tr>
            <tr>
              <td>고정된 값</td>
              <td className="num">{(a.totalParams - a.trainableParams).toLocaleString("en-US")}</td>
            </tr>
          </tbody>
        </table>
        {loose.map((issue, i) => (
          <IssueNote key={i} issue={issue} compact />
        ))}
        <h3 className="insp-sub">이렇게 써요</h3>
        <ul className="small tips">
          <li>
            <Glossed text="선 위의 글자는 그 선을 지나가는 데이터의 모양이에요. 64×64×16은 세로 64, 가로 64, 채널 16이라는 뜻이에요." />
          </li>
          <li>왼쪽 블록 상자에서 블록을 누르면 설명이 나오고, '블록 추가하기'를 누르면 캔버스에 놓여요.</li>
          <li>블록 오른쪽 점에서 다음 블록 왼쪽 점으로 끌면 연결돼요. 끄는 동안 이을 수 있는 블록은 초록색, 모양이 안 맞는 블록은 빨간색이에요.</li>
          <li>빨간 곳에 이어도 괜찮아요. 왜 안 맞는지 설명과 고치는 버튼이 나와요.</li>
          <li>선을 누르면 '연결 끊기' 버튼이 나와요. Delete 키로도 끊을 수 있어요.</li>
          <li>블록을 고르고 Delete 키를 누르면 지워지고, 앞뒤 블록이 다시 이어져요.</li>
        </ul>
      </>
    );
  }

  const def = LAYERS[node.data.type];
  const info = a.nodeInfo[node.id] || {};
  const p = node.data.params;
  const torch = info.inShape && info.outShape ? torchLines(node.data.type, p, info.inShape, info.outShape) : [];

  return (
    <>
      <h2 className="insp-h">
        {def.name} <span className="muted insp-ko">{def.ko}</span>
      </h2>
      <p className="small">
        <Glossed text={def.about} />
      </p>
      <dl className="shape-io">
        <dt>들어오는 모양</dt>
        <dd className="mono">{info.inShape ? fmtShape(info.inShape) : "–"}</dd>
        <dt>나가는 모양</dt>
        <dd className="mono">{info.outShape ? fmtShape(info.outShape) : "–"}</dd>
      </dl>
      {def.fields.map((f) => (
        <div className="field" key={f.key}>
          <label htmlFor={`f-${f.key}`}>{f.label}</label>
          {f.type === "select" ? (
            <select
              id={`f-${f.key}`}
              value={String(p[f.key])}
              onChange={(e) => setParam(node.id, f.key, f.options.find((o) => String(o.value) === e.target.value).value)}
            >
              {f.options.map((o) => (
                <option key={String(o.value)} value={String(o.value)}>
                  {o.label}
                </option>
              ))}
            </select>
          ) : f.type === "text" ? (
            <input id={`f-${f.key}`} type="text" value={p[f.key]} onChange={(e) => setParam(node.id, f.key, e.target.value)} />
          ) : (
            <input
              id={`f-${f.key}`}
              type="number"
              min={f.min}
              max={f.max}
              value={p[f.key]}
              onChange={(e) => setParam(node.id, f.key, Math.max(f.min, Math.min(f.max, Math.round(+e.target.value) || f.min)))}
            />
          )}
          {f.hint && (
            <p className="hint">
              <Glossed text={f.hint} />
            </p>
          )}
        </div>
      ))}
      {info.issue && <IssueNote issue={info.issue} compact />}
      {torch.length > 0 && (
        <>
          <h3 className="insp-sub">PyTorch로는</h3>
          <pre className="code-snippet">
            <code>{`# ${torchShape(info.inShape)} → ${torchShape(info.outShape)}\n${torch.join("\n")}`}</code>
          </pre>
        </>
      )}
      {node.deletable !== false && (
        <button className="btn outline small danger-outline" onClick={() => removeNode(node.id)}>
          이 블록 지우기
        </button>
      )}
    </>
  );
}

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
  useReactFlow,
} from "@xyflow/react";
import { useApp, useAnalysis } from "../store.js";
import { LAYERS, PALETTE, fmtShape, fmtCount, torchLines, torchShape } from "../model/layers.js";
import { TEMPLATES, NODE_W } from "../model/graph.js";
import IssueNote from "./IssueNote.jsx";
import ViewTabs from "./ViewTabs.jsx";

const DRAG_TYPE = "application/x-tensorblock";

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
  const a = useAnalysis();
  const rf = useReactFlow();

  const errors = a.issues.filter((i) => i.level === "error").length;
  const warns = a.issues.length - errors;

  // Delete / Backspace removes the selected block and joins its neighbours back up.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (e.target.closest("input, select, textarea, [contenteditable]")) return;
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
        const pad = 48;
        const zoom = Math.max(0.8, Math.min(1, (el.clientWidth - 2 * pad) / b.width));
        const fits = b.width * zoom <= el.clientWidth - 2 * pad;
        const x = fits ? (el.clientWidth - b.width * zoom) / 2 - b.x * zoom : pad - b.x * zoom;
        const y = el.clientHeight * 0.38 - (b.y + b.height / 2) * zoom;
        rf.setViewport({ x, y, zoom }, { duration: 200 });
      }, 60),
    [rf],
  );

  // Pans to the first problem so it is never off-screen during a demo.
  const showProblem = () => {
    const issue = a.issues.find((i) => i.level === "error" && i.nodeId) || a.issues.find((i) => i.nodeId);
    const n = issue && rf.getNode(issue.nodeId);
    if (n) rf.setCenter(n.position.x, n.position.y + 120, { zoom: Math.max(rf.getZoom(), 0.9), duration: 300 });
  };

  // Re-frame when blocks are added or removed.
  const count = nodes.length;
  useEffect(() => {
    fit();
  }, [count, fit]);

  const onDrop = (e) => {
    const type = e.dataTransfer.getData(DRAG_TYPE);
    if (!type) return;
    e.preventDefault();
    const p = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY });
    addBlock(type, { x: p.x - NODE_W / 2, y: p.y - 45 });
  };

  return (
    <div className="canvas-view">
      <header className="work-bar">
        <ViewTabs />
        <h1 className="visually-hidden">블록 캔버스</h1>
        {errors || warns ? (
          <button className={`badge badge-btn ${errors ? "danger" : "warning"}`} onClick={showProblem} role="status">
            {errors ? `고칠 곳 ${errors}개` : `주의 ${warns}개`} 보기
          </button>
        ) : (
          <span className="badge success" role="status">
            모양이 모두 맞아요
          </span>
        )}
        <span className="small muted bar-params">
          학습할 값 <span className="num-inline">{fmtCount(a.trainableParams)}</span>개
        </span>
        <span className="grow" />
        <select
          aria-label="모델 틀 불러오기"
          value={templateId ?? ""}
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
        <button
          className="btn outline small"
          onClick={() => {
            tidy();
            fit();
          }}
        >
          한 줄로 정렬
        </button>
      </header>

      <div className="canvas-body">
        <Palette onAdd={(t) => addBlock(t)} />
        <div className="flow" ref={flowRef} onDragOver={(e) => (e.preventDefault(), (e.dataTransfer.dropEffect = "copy"))} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={edges.map((e) => (training ? { ...e, className: "flowing" } : e))}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
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
        <Inspector />
      </div>
    </div>
  );
}

function Palette({ onAdd }) {
  return (
    <aside className="palette" aria-label="블록 상자">
      <p className="small muted">끌어다 놓거나 눌러서 넣어요. 선 위에 놓으면 그 사이에 끼워져요.</p>
      {PALETTE.map((g) => (
        <div key={g.title} className="palette-group">
          <h2 className="palette-h">{g.title}</h2>
          <ul role="list">
            {g.items.map((t) => (
              <li key={t}>
                <button
                  className="palette-item"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(DRAG_TYPE, t);
                    e.dataTransfer.effectAllowed = "copy";
                  }}
                  onClick={() => onAdd(t)}
                >
                  <span className="pi-name">{LAYERS[t].name}</span>
                  <span className="pi-ko">{LAYERS[t].exportOnly ? "코드 전용" : LAYERS[t].ko}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </aside>
  );
}

function BlockNode({ id, data }) {
  const a = useAnalysis();
  const selected = useApp((s) => s.selectedId === id);
  const def = LAYERS[data.type];
  // React Flow can paint a block one frame before the analysis knows about it.
  const info = a.nodeInfo[id] || { detached: true };
  const level = info.issue?.level;
  // Problems that belong to this block but have no incoming line to sit on.
  const extra = a.issues.filter((i) => i.nodeId === id && i !== info.issue);
  const cls = [
    "layer-block",
    `g-${def.group}`,
    selected && "selected",
    level === "error" && "has-error",
    info.detached && "detached",
    info.blocked && "blocked",
  ];

  return (
    <div className={cls.filter(Boolean).join(" ")}>
      {!def.source && <Handle type="target" position={Position.Left} />}
      <div className="block-head">
        <span className="block-name">{def.name}</span>
        <span className="block-ko">{def.exportOnly ? "코드 전용" : def.ko}</span>
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
        <IssueNote key={i} issue={issue} className="note-side nodrag" />
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

  return (
    <>
      <BaseEdge id={id} path={path} className={`edge-${state} ${selected ? "edge-selected" : ""}`} />
      <EdgeLabelRenderer>
        <div className={`shape-tag tag-${state} nodrag nopan`} style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}>
          {!info ? "연결 안 됨" : info.blocked ? "?" : fmtShape(info.shape)}
        </div>
        {level && !info.side && (
          <IssueNote issue={info.issue} className="note-edge nodrag nopan" style={{ transform: `translate(-50%, 0) translate(${lx}px, ${ly + 66}px)` }} />
        )}
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes = { block: BlockNode };
const edgeTypes = { shape: ShapeEdge };

function Inspector() {
  const selectedId = useApp((s) => s.selectedId);
  const node = useApp((s) => s.nodes.find((n) => n.id === s.selectedId));
  const setParam = useApp((s) => s.setParam);
  const removeNode = useApp((s) => s.removeNode);
  const a = useAnalysis();

  if (!selectedId || !node) {
    const loose = a.issues.filter((i) => !i.nodeId);
    return (
      <aside className="inspector" aria-label="블록 설명">
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
          <li>선 위의 글자는 그 선을 지나가는 데이터의 모양이에요. 64×64×16은 세로 64, 가로 64, 채널 16이라는 뜻이에요.</li>
          <li>모양이 안 맞으면 선이 빨개지고, 고치는 버튼이 나와요.</li>
          <li>블록 오른쪽 점에서 다음 블록 왼쪽 점으로 끌면 연결돼요.</li>
          <li>블록을 고르고 Delete 키를 누르면 지워지고, 앞뒤 블록이 다시 이어져요.</li>
        </ul>
      </aside>
    );
  }

  const def = LAYERS[node.data.type];
  const info = a.nodeInfo[node.id] || {};
  const p = node.data.params;
  const torch = info.inShape && info.outShape ? torchLines(node.data.type, p, info.inShape, info.outShape) : [];

  return (
    <aside className="inspector" aria-label="블록 설정">
      <h2 className="insp-h">
        {def.name} <span className="muted insp-ko">{def.ko}</span>
      </h2>
      <p className="small">{def.about}</p>
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
          {f.hint && <p className="hint">{f.hint}</p>}
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
    </aside>
  );
}

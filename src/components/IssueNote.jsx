import { useApp } from "../store.js";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyze } from "../model/graph.js";

// The red-pen correction note: what's wrong in one sentence, and the fix as a button.
export default function IssueNote({ issue, compact = false, className = "", style }) {
  const applyFix = useApp((s) => s.applyFix);
  return (
    <div
      className={`note ${issue.level} ${compact ? "compact" : ""} ${className}`}
      style={style}
      role={issue.level === "error" ? "alert" : undefined}
      onClick={(e) => e.stopPropagation()}
    >
      <p className="note-title">{issue.title}</p>
      <p className="note-text">{issue.text}</p>
      {issue.fixes?.length > 0 && (
        <div className="note-fixes">
          {issue.fixes.map((f, i) => (
            <button key={i} className={`btn small ${i === 0 ? "primary" : "outline"}`} onClick={() => applyFix(issue.nodeId, f.action)}>
              {f.label}
              {f.note && <span className="fix-note">{f.note}</span>}
            </button>
          ))}
        </div>
      )}
      {issue.level === "error" && <AiTutor issue={issue} />}
    </div>
  );
}

// Send model structure and rule results only; samples and images stay in the browser.
function AiTutor({ issue }) {
  const nodes = useApp((s) => s.nodes);
  const edges = useApp((s) => s.edges);
  const numClasses = useApp((s) => s.classes.length);
  const payload = useMemo(() => {
    const analysis = analyze(nodes, edges, numClasses);
    const info = analysis.nodeInfo[issue.nodeId];
    return JSON.stringify({
      graph: {
        nodes: nodes.map(({ id, data }) => ({ id, type: data.type, params: data.params })),
        edges: edges.map(({ source, target }) => ({ source, target })),
        numClasses,
      },
      issue: {
        nodeId: issue.nodeId ?? null, title: issue.title, text: issue.text, level: issue.level,
        fixes: (issue.fixes ?? []).map(({ label, note }) => ({ label, note })),
      },
      shapes: {
        input: info?.inShape ?? null,
        output: info?.outShape ?? null,
        incoming: edges.filter((e) => e.target === issue.nodeId).map((e) => ({
          source: e.source, shape: analysis.nodeInfo[e.source]?.outShape ?? null,
        })),
      },
    });
  }, [nodes, edges, numClasses, issue]);
  // A changed model resets the answer and unmounts/cancels the previous request.
  return <TutorRequest key={payload} payload={payload} />;
}

function TutorRequest({ payload }) {
  const [state, setState] = useState({ status: "idle", text: "" });
  const active = useRef(null);
  useEffect(() => () => active.current?.abort(), []);

  async function ask() {
    if (active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setState({ status: "loading", text: "" });
    const timeout = setTimeout(() => controller.abort("timeout"), 25000);
    try {
      const response = await fetch("/.netlify/functions/gemini-tutor", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: payload, signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "AI 설명을 불러오지 못했어요. 잠시 후 다시 시도해 주세요.");
      if (typeof data.explanation !== "string" || !data.explanation.trim()) throw new Error("AI 설명이 비어 있어요. 다시 시도해 주세요.");
      if (!controller.signal.aborted) setState({ status: "done", text: data.explanation });
    } catch (error) {
      if (!controller.signal.aborted || controller.signal.reason === "timeout") {
        setState({ status: "error", text: controller.signal.aborted ? "응답 시간이 길어졌어요. 다시 시도해 주세요." : error.message });
      }
    } finally {
      clearTimeout(timeout);
      if (active.current === controller) active.current = null;
    }
  }

  return (
    <div className="nodrag nopan" style={{ marginTop: "var(--space-3)", minWidth: 0 }} onPointerDown={(e) => e.stopPropagation()}>
      <button type="button" className="btn small outline" style={{ whiteSpace: "normal", height: "auto", minHeight: "var(--control-h-sm)" }} onClick={ask} disabled={state.status === "loading"}>
        {state.status === "loading" ? "AI가 설명하고 있어요…" : state.status === "error" ? "AI에게 다시 묻기" : "AI에게 묻기"}
      </button>
      <div aria-live="polite" aria-busy={state.status === "loading"}>
        {state.text && <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: "18rem", overflowY: "auto", fontSize: "var(--step--1)", lineHeight: 1.6, userSelect: "text" }}>{state.text}</p>}
      </div>
      {state.status === "done" && <p style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 0 }}>AI가 생성한 참고 설명이에요. 에러 판정은 기존 규칙을 따릅니다.</p>}
    </div>
  );
}

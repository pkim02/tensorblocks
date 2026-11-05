import { useApp } from "../store.js";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyze } from "../model/graph.js";
import { Glossed } from "./Term.jsx";

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
      <p className="note-title" style={{ paddingInline: "var(--space-1)" }}>{issue.title}</p>
      <p className="note-text" style={{ paddingInline: "var(--space-1)" }}>
        <Glossed text={issue.text} />
      </p>
      {issue.code && (
        <p className="note-code" style={{ paddingInline: "var(--space-1)" }}>
          코드에서는: <code>{issue.code}</code>
        </p>
      )}
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
        nodeId: issue.nodeId ?? null, title: issue.title, text: issue.code ? `${issue.text} (코드에서는: ${issue.code})` : issue.text, level: issue.level,
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

const SECTIONS = [
  { key: "why", label: "왜 발생했나요", color: "var(--danger-11)" },
  { key: "shape", label: "shape 비교", color: "var(--text-muted)" },
  { key: "fix", label: "어떻게 고치나요", color: "var(--accent-11)" },
];

// Render only **bold** and `code` marks as React elements — never raw HTML.
function Inline({ text }) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i} style={{ fontWeight: 700, background: "var(--warning-3, transparent)", borderRadius: 2, paddingInline: 2 }}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i} style={{ fontFamily: "var(--font-mono)", fontSize: "0.92em", background: "var(--gray-3)", borderRadius: 3, padding: "0 3px", whiteSpace: "nowrap" }}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function Explanation({ data }) {
  return (
    <div style={{ marginTop: "var(--space-3)", paddingInline: "var(--space-1)", overflowWrap: "anywhere", maxHeight: "22rem", overflowY: "auto", userSelect: "text" }}>
      {data.summary && (
        <p style={{ margin: "0 0 var(--space-3)", fontWeight: 600, lineHeight: 1.45, paddingInlineStart: "var(--space-2)", borderInlineStart: "3px solid var(--danger-9)" }}>
          <Inline text={data.summary} />
        </p>
      )}
      {SECTIONS.filter((s) => data[s.key]?.length).map((s) => (
        <section key={s.key} style={{ marginTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-1)", fontSize: 11, fontWeight: 700, letterSpacing: "0.02em", color: s.color }}>{s.label}</h4>
          <ul style={{ margin: 0, paddingInlineStart: "1.1em", display: "flex", flexDirection: "column", gap: 2 }}>
            {data[s.key].map((line, i) => <li key={i} style={{ lineHeight: 1.5 }}><Inline text={line} /></li>)}
          </ul>
        </section>
      ))}
    </div>
  );
}

const isExplanation = (x) => x && typeof x === "object" && typeof x.summary === "string" &&
  SECTIONS.every((s) => Array.isArray(x[s.key]) && x[s.key].every((l) => typeof l === "string"));

function TutorRequest({ payload }) {
  const [state, setState] = useState({ status: "idle", text: "", data: null });
  const active = useRef(null);
  useEffect(() => () => active.current?.abort(), []);

  async function ask() {
    if (active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setState({ status: "loading", text: "", data: null });
    const timeout = setTimeout(() => controller.abort("timeout"), 25000);
    try {
      const response = await fetch("/.netlify/functions/gemini-tutor", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: payload, signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "AI 설명을 불러오지 못했어요. 잠시 후 다시 시도해 주세요.");
      if (!isExplanation(data.explanation)) throw new Error("AI 설명이 비어 있어요. 다시 시도해 주세요.");
      if (!controller.signal.aborted) setState({ status: "done", text: "", data: data.explanation });
    } catch (error) {
      if (!controller.signal.aborted || controller.signal.reason === "timeout") {
        setState({ status: "error", data: null, text: controller.signal.aborted ? "응답 시간이 길어졌어요. 다시 시도해 주세요." : error.message });
      }
    } finally {
      clearTimeout(timeout);
      if (active.current === controller) active.current = null;
    }
  }

  return (
    <div className="nodrag nopan" style={{ marginTop: "var(--space-3)", minWidth: 0, textAlign: "start" }} onPointerDown={(e) => e.stopPropagation()}>
      <button type="button" className="btn small outline" style={{ whiteSpace: "normal", height: "auto", minHeight: "var(--control-h-sm)", padding: "var(--space-1) var(--space-2)", justifyContent: "flex-start", textAlign: "start" }} onClick={ask} disabled={state.status === "loading"}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" style={{ flexShrink: 0 }}>
          <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" />
          <path d="M20 2v4m-2-2h4" />
        </svg>
        {state.status === "loading" ? "AI가 설명하고 있어요…" : state.status === "error" ? "AI에게 다시 묻기" : "AI에게 묻기"}
      </button>
      <div aria-live="polite" aria-busy={state.status === "loading"} style={{ minWidth: 0 }}>
        {state.data && <Explanation data={state.data} />}
        {state.status === "error" && state.text && <p role="status" style={{ margin: "var(--space-2) 0 0", paddingInline: "var(--space-1)", color: "var(--danger-11)", fontSize: 12 }}>{state.text}</p>}
      </div>
      {state.status === "done" && <p style={{ fontSize: 11, color: "var(--text-muted)", margin: "var(--space-3) 0 0", paddingTop: "var(--space-2)", paddingInline: "var(--space-1)", borderTop: "1px dashed var(--border)" }}>AI가 생성한 참고 설명이에요. 에러 판정은 기존 규칙을 따릅니다.</p>}
    </div>
  );
}

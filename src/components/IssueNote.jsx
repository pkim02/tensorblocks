import { useApp } from "../store.js";

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
    </div>
  );
}

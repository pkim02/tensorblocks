import { useState } from "react";
import { createPortal } from "react-dom";
import { GLOSSARY } from "../model/terms.js";

const KEYS = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
const PATTERN = new RegExp(`(${KEYS.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");

// Plain text with the glossary words marked; each word is marked once per text.
export function Glossed({ text }) {
  if (typeof text !== "string") return text ?? null;
  const seen = new Set();
  return text.split(PATTERN).map((part, i) => {
    if (!GLOSSARY[part] || seen.has(part)) return part;
    seen.add(part);
    return <Term key={i} word={part} />;
  });
}

// A dotted word that shows its meaning on hover or keyboard focus. The bubble sits on
// <body> so scrolling panels and canvas notes don't clip it.
export function Term({ word, children }) {
  const [at, setAt] = useState(null);
  const show = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - 268, r.left + r.width / 2 - 130));
    setAt(r.bottom + 90 > window.innerHeight ? { left, bottom: window.innerHeight - r.top + 6 } : { left, top: r.bottom + 6 });
  };
  return (
    <>
      <span className="term" tabIndex={0} onMouseEnter={show} onMouseLeave={() => setAt(null)} onFocus={show} onBlur={() => setAt(null)}>
        {children ?? word}
      </span>
      {at &&
        createPortal(
          <span className="term-tip" role="tooltip" style={at}>
            <b>{word}</b> {GLOSSARY[word]}
          </span>,
          document.body,
        )}
    </>
  );
}

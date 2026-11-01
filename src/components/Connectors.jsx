import { useLayoutEffect, useRef, useState } from "react";
import { useApp } from "../store.js";

// The curved lines between steps, like Teachable Machine: every class card feeds the
// training card, and training feeds the preview. Measured from the DOM, redrawn on resize.
export default function Connectors() {
  const svgRef = useRef(null);
  const classCount = useApp((s) => s.classes.length);
  const training = useApp((s) => s.train.status === "training");
  const [paths, setPaths] = useState({ data: [], out: null });

  useLayoutEffect(() => {
    // Measure against our own parent: a parent's ref isn't attached yet when this runs.
    const wrap = svgRef.current?.parentElement;
    if (!wrap) return;
    const measure = () => {
      const train = wrap.querySelector('[data-port="train"]');
      const preview = wrap.querySelector('[data-port="preview"]');
      // Stacked layout (phones): no lines.
      if (!train || !preview || getComputedStyle(wrap).display !== "grid" || getComputedStyle(wrap).gridTemplateColumns.split(" ").length < 3) {
        setPaths({ data: [], out: null });
        return;
      }
      const o = wrap.getBoundingClientRect();
      const box = (el) => {
        const r = el.getBoundingClientRect();
        return { l: r.left - o.left, r: r.right - o.left, m: r.top - o.top + Math.min(r.height / 2, 120) };
      };
      const curve = (x1, y1, x2, y2) => {
        const dx = (x2 - x1) / 2;
        return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
      };
      const t = box(train),
        p = box(preview);
      setPaths({
        data: [...wrap.querySelectorAll('[data-port="class"]')].map((el) => {
          const c = box(el);
          return curve(c.r, c.m, t.l, t.m);
        }),
        out: curve(t.r, t.m, p.l, p.m),
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(wrap);
    wrap.querySelectorAll("[data-port]").forEach((el) => ro.observe(el));
    return () => ro.disconnect();
  }, [classCount]);

  return (
    <svg ref={svgRef} className={`connectors ${training ? "flowing" : ""}`} aria-hidden="true">
      {paths.data.map((d, i) => (
        <path key={i} d={d} className="link data-link" />
      ))}
      {paths.out && <path d={paths.out} className="link" />}
    </svg>
  );
}

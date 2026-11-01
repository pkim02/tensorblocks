import { useState } from "react";

// Accuracy per epoch: training vs validation, one 0–100% axis.
const W = 320,
  H = 150;
const M = { l: 34, r: 44, t: 10, b: 22 };

export default function AccuracyChart({ logs, total }) {
  const [hover, setHover] = useState(null);
  if (!logs.length) return null;
  const hasVal = logs.some((l) => l.valAcc != null);
  const n = Math.max(total, logs.length, 2);
  const x = (e) => M.l + ((e - 1) / (n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - v) * (H - M.t - M.b);
  const line = (key) =>
    logs
      .filter((l) => l[key] != null)
      .map((l, i) => `${i ? "L" : "M"}${x(l.epoch).toFixed(1)},${y(l[key]).toFixed(1)}`)
      .join("");
  const last = logs[logs.length - 1];
  const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const ep = Math.round(((px - M.l) / (W - M.l - M.r)) * (n - 1) + 1);
    setHover(logs.find((l) => l.epoch === Math.max(1, Math.min(logs.length, ep))) || null);
  };

  // End labels: keep them apart when the two lines finish close together.
  let ya = y(last.acc),
    yb = hasVal ? y(last.valAcc) : null;
  if (yb != null && Math.abs(ya - yb) < 12) {
    const mid = (ya + yb) / 2;
    [ya, yb] = ya <= yb ? [mid - 6, mid + 6] : [mid + 6, mid - 6];
  }

  return (
    <figure className="chart">
      <figcaption className="chart-legend">
        <span className="key k1">학습 정확도</span>
        {hasVal && <span className="key k2">검증 정확도 (처음 보는 사진)</span>}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`정확도 그래프, 마지막 에폭 학습 ${pct(last.acc)}${hasVal ? `, 검증 ${pct(last.valAcc)}` : ""}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={M.l - 6} y={y(v) + 4} textAnchor="end">
              {v * 100}%
            </text>
          </g>
        ))}
        <text className="tick" x={M.l} y={H - 4}>
          1
        </text>
        <text className="tick" x={W - M.r} y={H - 4} textAnchor="end">
          에폭 {n}
        </text>
        <path className="series s1" d={line("acc")} />
        {hasVal && <path className="series s2" d={line("valAcc")} />}
        <circle className="dot s1" cx={x(last.epoch)} cy={y(last.acc)} r="4" />
        {hasVal && <circle className="dot s2" cx={x(last.epoch)} cy={y(last.valAcc)} r="4" />}
        <text className="end-label" x={x(last.epoch) + 8} y={ya + 4}>
          {pct(last.acc)}
        </text>
        {hasVal && (
          <text className="end-label" x={x(last.epoch) + 8} y={yb + 4}>
            {pct(last.valAcc)}
          </text>
        )}
        {hover && <line className="crosshair" x1={x(hover.epoch)} x2={x(hover.epoch)} y1={M.t} y2={H - M.b} />}
      </svg>
      {hover && (
        <div className="chart-tip" style={{ left: `${(x(hover.epoch) / W) * 100}%` }}>
          <strong>에폭 {hover.epoch}</strong>
          <span className="key k1">학습 {pct(hover.acc)}</span>
          {hasVal && <span className="key k2">검증 {pct(hover.valAcc)}</span>}
          <span>손실 {hover.loss?.toFixed(3)}</span>
        </div>
      )}
    </figure>
  );
}

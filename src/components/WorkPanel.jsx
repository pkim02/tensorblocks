import { useApp } from "../store.js";
import DataPanel from "./DataPanel.jsx";
import TrainPanel, { TrainAction } from "./TrainPanel.jsx";
import PreviewPanel from "./PreviewPanel.jsx";

const TABS = [
  ["data", "1", "데이터"],
  ["train", "2", "학습"],
  ["test", "3", "시험"],
  ["block", "", "블록"],
];

// Right side of the canvas: data → training → testing, plus the selected block's settings.
export default function WorkPanel({ inspector }) {
  const panel = useApp((s) => s.panel);
  const setPanel = useApp((s) => s.setPanel);
  const projectRevision = useApp((s) => s.projectRevision);
  return (
    <aside className="work-panel" aria-label="작업 패널">
      <nav className="work-tabs" aria-label="작업 단계">
        {TABS.map(([id, n, label]) => (
          <button key={id} aria-current={panel === id ? "step" : undefined} onClick={() => setPanel(id)}>
            {n && <span className="mono">{n}</span>} {label}
          </button>
        ))}
      </nav>
      <div className="work-body">
        {panel === "data" && <DataPanel key={projectRevision} />}
        {panel === "train" && <TrainPanel />}
        {panel === "test" && <PreviewPanel />}
        {panel === "block" && inspector}
      </div>
      <TrainAction />
    </aside>
  );
}

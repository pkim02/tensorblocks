import { useApp } from "../store.js";

const VIEWS = [
  ["canvas", "블록"],
  ["three", "3D"],
  ["code", "Colab 코드"],
];

// The same model seen three ways. Sits at the start of each of those screens' top bar.
export default function ViewTabs() {
  const view = useApp((s) => s.view);
  const setView = useApp((s) => s.setView);
  return (
    <>
      <span className="brand">TensorBlocks</span>
      <nav className="view-tabs" aria-label="모델 보기">
        {VIEWS.map(([id, label]) => (
          <button key={id} aria-current={view === id ? "page" : undefined} onClick={() => setView(id)}>
            {label}
          </button>
        ))}
      </nav>
    </>
  );
}

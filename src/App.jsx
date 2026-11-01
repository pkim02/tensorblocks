import { ReactFlowProvider } from "@xyflow/react";
import { useApp } from "./store.js";
import DataPanel from "./components/DataPanel.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import PreviewPanel from "./components/PreviewPanel.jsx";
import CanvasView from "./components/CanvasView.jsx";
import ExportDialog from "./components/ExportDialog.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";

export default function App() {
  const view = useApp((s) => s.view);
  return (
    <ReactFlowProvider>
      {view === "canvas" ? (
        <ErrorBoundary>
          <CanvasView />
        </ErrorBoundary>
      ) : (
        <div className="app">
          <header className="topbar">
            <h1 className="brand">TensorBlocks</h1>
            <p className="tagline">Teachable Machine처럼 쉽게 만들고, 모델 속은 블록으로 열어 보고, PyTorch 코드로 가져가요.</p>
          </header>
          <main className="columns">
            <DataPanel />
            <ModelPanel />
            <PreviewPanel />
          </main>
        </div>
      )}
      <ExportDialog />
    </ReactFlowProvider>
  );
}

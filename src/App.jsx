import { ReactFlowProvider } from "@xyflow/react";
import { useApp } from "./store.js";
import DataPanel, { DemoButton } from "./components/DataPanel.jsx";
import Connectors from "./components/Connectors.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import PreviewPanel from "./components/PreviewPanel.jsx";
import CanvasView from "./components/CanvasView.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import ThreeView from "./components/ThreeView.jsx";
import CodeView from "./components/CodeView.jsx";

export default function App() {
  const view = useApp((s) => s.view);
  return (
    <ReactFlowProvider>
      {view === "three" ? (
        <ErrorBoundary>
          <ThreeView />
        </ErrorBoundary>
      ) : view === "code" ? (
        <ErrorBoundary>
          <CodeView />
        </ErrorBoundary>
      ) : view === "canvas" ? (
        <ErrorBoundary>
          <CanvasView />
        </ErrorBoundary>
      ) : (
        <div className="app">
          <header className="topbar">
            <div>
              <h1 className="brand">TensorBlocks</h1>
              <p className="tagline">Teachable Machine처럼 쉽게 만들고, 모델 속은 블록으로 열어 보고, PyTorch 코드로 가져가요.</p>
            </div>
            <DemoButton />
          </header>
          <main className="flow-main">
            <Connectors />
            <DataPanel />
            <ModelPanel />
            <PreviewPanel />
          </main>
        </div>
      )}
    </ReactFlowProvider>
  );
}

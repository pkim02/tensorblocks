import { ReactFlowProvider } from "@xyflow/react";
import { useApp } from "./store.js";
import CanvasView from "./components/CanvasView.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import ThreeView from "./components/ThreeView.jsx";
import CodeView from "./components/CodeView.jsx";
import { ProjectSession } from "./components/ProjectControls.jsx";

export default function App() {
  const view = useApp((s) => s.view);
  return (
    <ReactFlowProvider>
      <ProjectSession />
      <ErrorBoundary>{view === "three" ? <ThreeView /> : view === "code" ? <CodeView /> : <CanvasView />}</ErrorBoundary>
    </ReactFlowProvider>
  );
}

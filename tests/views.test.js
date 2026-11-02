import assert from "node:assert/strict";
import { test } from "node:test";
import { createServer } from "vite";
import { renderToString } from "react-dom/server";
import { createElement } from "react";
import { ReactFlowProvider } from "@xyflow/react";

test("main, block, 3D and Colab screens render without Firebase configuration", async () => {
  const server = await createServer({
    envDir: ".local-tools/no-env",
    define: Object.fromEntries(["API_KEY", "AUTH_DOMAIN", "PROJECT_ID", "APP_ID"].map((key) => [`import.meta.env.VITE_FIREBASE_${key}`, JSON.stringify("")])),
    server: { middlewareMode: true },
    appType: "custom",
  });
  try {
    const { default: App } = await server.ssrLoadModule("/src/App.jsx");
    const main = renderToString(createElement(App));
    assert.match(main, /모델 구성과 설정을 저장합니다/);
    assert.match(main, /프로젝트 저장·불러오기/);
    assert.match(main, /disabled/);
    // Zustand's server snapshot uses initial state, so render each actual screen directly.
    for (const [file, label] of [
      ["CanvasView", "블록 캔버스"],
      ["ThreeView", "3D"],
      ["CodeView", "Colab"],
    ]) {
      const { default: Screen } = await server.ssrLoadModule(`/src/components/${file}.jsx`);
      const html = renderToString(createElement(ReactFlowProvider, null, createElement(Screen)));
      assert.ok(html.length > 500, file);
      assert.ok(html.includes(label), file);
    }
  } finally {
    await server.close();
  }
});

import { createServer } from "vite";
import path from "node:path";

// Isolated browser test server. Never reads .env.local and never uses a real project.
const values = {
  VITE_FIREBASE_API_KEY: "demo-key",
  VITE_FIREBASE_AUTH_DOMAIN: "demo-tensorblocks.firebaseapp.com",
  VITE_FIREBASE_PROJECT_ID: "demo-tensorblocks",
  VITE_FIREBASE_APP_ID: "demo-app",
};
const server = await createServer({
  mode: "emulator",
  envDir: path.resolve(".local-tools/no-env"),
  define: Object.fromEntries(Object.entries(values).map(([key, value]) => [`import.meta.env.${key}`, JSON.stringify(value)])),
  server: { host: "127.0.0.1", port: 5183, strictPort: true },
});
await server.listen();
console.log("Local-only browser test: http://127.0.0.1:5183/tests/browser-emulator.html");

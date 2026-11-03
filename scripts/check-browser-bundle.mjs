import assert from "node:assert/strict";
import { build } from "vite";

// Inspect bundled module names, never emitted code (which contains public Firebase configuration).
const modules = new Set();
await build({
  logLevel: "silent",
  build: { write: false },
  plugins: [
    {
      name: "inspect-browser-dependencies",
      generateBundle(_options, bundle) {
        for (const item of Object.values(bundle)) {
          if (item.type === "chunk") for (const id of Object.keys(item.modules)) modules.add(id.replaceAll("\\", "/"));
        }
      },
    },
  ],
});
const grpcIncluded = [...modules].some((id) => id.includes("/@grpc/grpc-js/"));
const firestoreNodeEntryIncluded = [...modules].some((id) => id.includes("firestore") && id.includes("index.node."));
const firestoreBrowserEntryIncluded = [...modules].some((id) => id.includes("/@firebase/firestore/dist/index.esm.js"));
console.log(JSON.stringify({ grpcIncluded, firestoreNodeEntryIncluded, firestoreBrowserEntryIncluded }, null, 2));
assert.equal(grpcIncluded, false);
assert.equal(firestoreNodeEntryIncluded, false);
assert.equal(firestoreBrowserEntryIncluded, true);

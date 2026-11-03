import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { doc, getDoc, getDocs, collection, setDoc, updateDoc, deleteDoc, deleteField, serverTimestamp } from "firebase/firestore";
import { restoreProject } from "../src/projects/serialize.js";
import { buildTemplate, TEMPLATES } from "../src/model/graph.js";
import { serializeProject } from "../src/projects/serialize.js";
import { createProjectRepository } from "../src/projects/repository.js";
import { createStore } from "zustand/vanilla";

// No fallback to a real project. The optional suite can only contact a loopback emulator.
const host = process.env.FIRESTORE_EMULATOR_HOST;
test("Firestore emulator: actual rules and SDK repository", { timeout: 20000, skip: !host && "Firestore Emulator가 실행되지 않았어요." }, async (t) => {
  assert.match(host, /^(127\.0\.0\.1|localhost):\d+$/);
  const { initializeTestEnvironment, assertSucceeds, assertFails } = await import("@firebase/rules-unit-testing");
  const [hostname, port] = host.split(":");
  const env = await initializeTestEnvironment({
    projectId: "demo-tensorblocks",
    firestore: {
      host: hostname,
      port: +port,
      rules: await readFile(new URL("../firestore.rules", import.meta.url), "utf8"),
    },
  });
  try {
    await env.clearFirestore();
    const alice = env.authenticatedContext("alice").firestore();
    const bob = env.authenticatedContext("bob").firestore();
    const anonymous = env.unauthenticatedContext().firestore();
    const data = {
      ...serializeProject({
        ...buildTemplate(TEMPLATES.find((t) => t.side)),
        templateId: null,
        classes: [
          { id: "c1", name: "A" },
          { id: "c2", name: "B" },
        ],
        settings: { epochs: 20, batchSize: 16, learningRate: 0.001 },
      }),
      name: "VLM",
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    };
    await t.test("owner can create, list, read and update own project", async () => {
      const ref = doc(alice, "users/alice/projects/p1");
      await assertSucceeds(setDoc(ref, data));
      await assertSucceeds(getDoc(ref));
      await assertSucceeds(getDocs(collection(alice, "users/alice/projects")));
      await assertSucceeds(updateDoc(ref, { name: "변경", updatedAt: serverTimestamp() }));
    });
    await t.test("anonymous and other users cannot read, list or write Alice's projects", async () => {
      for (const db of [anonymous, bob]) {
        const ref = doc(db, "users/alice/projects/p1");
        await assertFails(getDoc(ref));
        await assertFails(getDocs(collection(db, "users/alice/projects")));
        await assertFails(setDoc(ref, data));
        await assertFails(updateDoc(ref, { name: "침입", updatedAt: serverTimestamp() }));
        await assertFails(deleteDoc(ref));
      }
    });
    await t.test("bad schema, extra image fields, forged timestamps and deletes are denied", async () => {
      const ref = doc(alice, "users/alice/projects/p1");
      await assertFails(updateDoc(ref, { schemaVersion: 99, updatedAt: serverTimestamp() }));
      await assertFails(updateDoc(ref, { photos: "private", updatedAt: serverTimestamp() }));
      await assertFails(updateDoc(ref, { createdAt: new Date(0), updatedAt: serverTimestamp() }));
      await assertFails(updateDoc(ref, { updatedAt: new Date(0) }));
      await assertFails(deleteDoc(ref));
      for (const invalid of [
        { name: "" },
        { nodes: "not a list" },
        { edges: {} },
        { classes: [], settings: {} },
        { settings: { epochs: -1, batchSize: 16, learningRate: 0.001 } },
        { settings: { epochs: 20, batchSize: 16, learningRate: 0 } },
        { nodes: Array(501).fill(null) },
        { edges: Array(1001).fill(null) },
        { classes: Array(101).fill(null) },
      ])
        await assertFails(updateDoc(ref, { ...invalid, updatedAt: serverTimestamp() }));
    });
    await t.test("real repository saves, overwrites, lists and loads VLM data", async () => {
      const auth = createStore(() => ({ ready: true, user: { uid: "alice" } }));
      // Keep the exact database configured by the rules-test context (host AND mock credentials).
      // Reopening from its compat app can otherwise create a separate modular instance.
      const repository = createProjectRepository(
        alice.app,
        auth,
        () => true,
        () => alice,
      );
      const id = await repository.save("alice", null, "저장", data);
      await repository.save("alice", id, "수정", data);
      const loaded = await repository.load("alice", id);
      assert.equal(loaded.name, "수정");
      assert.deepEqual(loaded.edges, data.edges);
      assert.deepEqual(loaded.dataset, data.dataset);
      assert.ok(loaded.createdAt.toMillis());
      assert.ok((await repository.list("alice")).some((p) => p.id === id));
    });
    await t.test("v1 remains readable and upgrades to v2 without changing creation time", async () => {
      const legacy = { ...data, schemaVersion: 1 };
      delete legacy.dataset;
      const ref = doc(alice, "users/alice/projects/legacy");
      await assertSucceeds(setDoc(ref, legacy));
      const before = (await getDoc(ref)).data();
      const restored = restoreProject(before);
      const auth = createStore(() => ({ ready: true, user: { uid: "alice" } }));
      const repository = createProjectRepository(
        alice.app,
        auth,
        () => true,
        () => alice,
      );
      await repository.save("alice", "legacy", "v2", restored);
      const after = (await getDoc(ref)).data();
      assert.equal(after.schemaVersion, 2);
      assert.equal(after.createdAt.toMillis(), before.createdAt.toMillis());
    });
    await t.test("v2 requires bounded metadata and rejects image/preview fields", async () => {
      const ref = doc(alice, "users/alice/projects/p1");
      await assertFails(updateDoc(ref, { dataset: deleteField(), updatedAt: serverTimestamp() }));
      for (const dataset of [
        { ...data.dataset, repo: "x".repeat(201) },
        { ...data.dataset, preview: [] },
        { ...data.dataset, fields: { image: "photo", question: "q", answer: "a", list: null, images: [] } },
        { ...data.dataset, config: 42 },
      ])
        await assertFails(updateDoc(ref, { dataset, updatedAt: serverTimestamp() }));
    });
  } finally {
    await env.cleanup();
  }
});

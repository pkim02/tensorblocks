import assert from "node:assert/strict";
import { test } from "node:test";
import { createStore } from "zustand/vanilla";
import { defaultDataset } from "../src/data/dataset.js";
import { analyze, buildTemplate, makeNode, makeEdge, TEMPLATES } from "../src/model/graph.js";
import { serializeProject, restoreProject, fingerprint, projectName } from "../src/projects/serialize.js";
import { createProjectController } from "../src/projects/controller.js";
import { createProjectRepository, projectError } from "../src/projects/repository.js";

function workspace(template = TEMPLATES.find((t) => t.id === "tm")) {
  return {
    ...buildTemplate(template),
    templateId: template.id,
    classes: [
      { id: "c2", name: "두 번째", samples: [] },
      { id: "c1", name: "첫 번째", samples: [] },
    ],
    settings: { epochs: 20, batchSize: 16, learningRate: 0.001 },
    dataset: defaultDataset(),
    train: { status: "idle", logs: [] },
    trained: null,
    demoData: false,
    projectRevision: 0,
  };
}
const saved = () => ({ ...serializeProject(workspace()), name: "저장한 모델" });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
function setup(repository = {}) {
  const app = createStore(() => workspace());
  const auth = createStore(() => ({ ready: true, user: { uid: "alice" } }));
  const calls = [];
  const controller = createProjectController({
    app,
    auth,
    repository: {
      list: async () => [{ id: "p1", name: "내 모델" }],
      load: async () => saved(),
      save: async (...args) => {
        calls.push(args);
        return args[1] || "p1";
      },
      ...repository,
    },
    dispose: (v) => calls.push(["dispose", v]),
    stopTraining: () => calls.push(["stop"]),
  });
  controller.start();
  return { app, auth, controller, calls, state: () => controller.store.getState() };
}

test("all existing templates round-trip without changing IDs, parameters, class order or analysis", () => {
  for (const t of TEMPLATES) {
    const original = workspace(t);
    const data = serializeProject(original);
    const restored = restoreProject(JSON.parse(JSON.stringify(data)));
    assert.deepEqual(serializeProject(restored), data, t.id);
    assert.deepEqual(
      restored.classes.map((c) => c.id),
      ["c2", "c1"],
    );
    assert.deepEqual(analyze(restored.nodes, restored.edges, 2), analyze(original.nodes, original.edges, 2), t.id);
  }
});
test("VLM preserves both incoming branches at Concat", () => {
  const original = workspace(TEMPLATES.find((t) => t.side));
  const restored = restoreProject(serializeProject(original));
  const concat = restored.nodes.find((n) => n.data.type === "concat");
  assert.ok(concat);
  assert.equal(restored.edges.filter((e) => e.target === concat.id).length, 2);
  assert.deepEqual(
    restored.edges.map((e) => [e.id, e.source, e.target]),
    original.edges.map((e) => [e.id, e.source, e.target]),
  );
});
test("serialization never touches canvas / trained objects or transient flow fields", () => {
  const s = workspace();
  const cyclic = {};
  cyclic.self = cyclic;
  s.classes[0].samples.push({ id: "photo", canvas: cyclic, thumb: "private-photo" });
  s.trained = cyclic;
  s.nodes[0].measured = cyclic;
  const json = JSON.stringify(serializeProject(s));
  assert.ok(!json.includes("private-photo"));
  assert.ok(!json.includes("measured"));
  assert.deepEqual(restoreProject(JSON.parse(json)).classes[0].samples, []);
});
test("invalid versions, graph references, IDs, params, settings and oversized data are rejected", () => {
  assert.throws(() => restoreProject({ ...saved(), schemaVersion: 99 }));
  for (const mutate of [
    (s) => {
      s.edges[0].source = "missing";
    },
    (s) => {
      s.nodes[1].id = s.nodes[0].id;
    },
    (s) => {
      s.nodes[0].data.type = "unknown";
    },
    (s) => {
      s.nodes[0].position.x = Infinity;
    },
    (s) => {
      s.nodes[0].data.params.size = -5;
    },
    (s) => {
      s.settings.learningRate = NaN;
    },
    (s) => {
      s.classes[1].id = s.classes[0].id;
    },
  ]) {
    const s = workspace();
    mutate(s);
    assert.throws(() => serializeProject(s));
  }
  const s = workspace(TEMPLATES.find((t) => t.side));
  const text = s.nodes.find((n) => n.data.type === "text");
  s.nodes = Array.from({ length: 200 }, (_, i) => ({ ...text, id: `text${i}`, data: { type: "text", params: { prompt: "가".repeat(4000) } } }));
  s.edges = [];
  assert.throws(() => serializeProject(s), /너무 커/);
  assert.throws(() => projectName(" "));
});
test("new save, overwrite, save as copy and creation of clean baseline", async () => {
  const { controller: c, calls, state, app } = setup();
  c.setName(" 첫 모델 ");
  await c.save();
  assert.equal(state().id, "p1");
  assert.equal(calls[0][1], null);
  assert.equal(calls[0][2], "첫 모델");
  // Trimmed name is compared consistently.
  assert.equal(c.dirty(), false);
  app.setState({ settings: { ...app.getState().settings, epochs: 30 } });
  assert.equal(c.dirty(), true);
  await c.save();
  assert.equal(calls[1][1], "p1");
  await c.save(true);
  assert.equal(calls[2][1], null);
});
test("load requires explicit confirmation and cancellation preserves local work", async () => {
  const { controller: c, state, app } = setup();
  const before = fingerprint(app.getState());
  await c.load("other");
  assert.ok(state().pending);
  assert.equal(fingerprint(app.getState()), before);
  c.cancelLoad();
  assert.equal(state().pending, null);
  await c.load("other");
  c.confirmLoad();
  assert.equal(state().id, "other");
  assert.equal(c.dirty(), false);
  assert.equal(app.getState().trained, null);
  assert.equal(app.getState().projectRevision, 1);
});
test("photos and trained results still require confirmation after saving model configuration", async () => {
  const { controller: c, app, state } = setup();
  app.getState().classes[0].samples.push({ id: "photo" });
  c.setName("사진 제외 모델");
  await c.save();
  assert.equal(c.dirty(), true);
  await c.load("other");
  assert.ok(state().pending);
});
test("edits made during a save remain dirty", async () => {
  const wait = deferred();
  const { controller: c, app } = setup({ save: () => wait.promise });
  c.setName("모델");
  const saving = c.save();
  app.setState({ settings: { ...app.getState().settings, epochs: 45 } });
  wait.resolve("p1");
  await saving;
  assert.equal(c.dirty(), true);
});
test("anonymous access is blocked before calling repository", async () => {
  let called = false;
  const {
    controller: c,
    auth,
    state,
  } = setup({
    list: () => {
      called = true;
    },
  });
  auth.setState({ user: null });
  await c.list();
  assert.equal(called, false);
  assert.match(state().message, /로그인/);
  const repo = createProjectRepository(null, auth);
  await assert.rejects(repo.list("alice"), /로그인/);
});
test("repository refuses requests for a different user and offline requests", async () => {
  const auth = createStore(() => ({ ready: true, user: { uid: "alice" } }));
  await assert.rejects(createProjectRepository(null, auth).load("bob", "p1"), /로그인/);
  await assert.rejects(createProjectRepository({}, auth, () => false).list("alice"), /인터넷/);
});
test("network and malformed-data errors leave the workspace intact and release busy state", async () => {
  for (const load of [
    async () => {
      throw { code: "unavailable" };
    },
    async () => ({ schemaVersion: 99 }),
  ]) {
    const { controller: c, app, state } = setup({ load });
    const before = fingerprint(app.getState());
    await c.load("bad");
    assert.equal(fingerprint(app.getState()), before);
    assert.equal(state().busy, false);
    assert.equal(state().pending, null);
    assert.ok(state().message.length > 0);
  }
  assert.match(projectError({ code: "permission-denied" }), /권한/);
});
test("training blocks load before request, after request and at confirmation", async () => {
  let calls = 0;
  const wait = deferred();
  const {
    controller: c,
    app,
    state,
  } = setup({
    load: () => {
      calls++;
      return wait.promise;
    },
  });
  app.setState({ train: { status: "training" } });
  await c.load("p1");
  assert.equal(calls, 0);
  assert.match(state().message, /학습/);
  app.setState({ train: { status: "idle" } });
  const loading = c.load("p1");
  app.setState({ train: { status: "training" } });
  wait.resolve(saved());
  await loading;
  assert.throws(() => c.confirmLoad(), /학습/);
  assert.equal(state().id, null);
});
test("account switch clears project state, images and model; late requests cannot restore old user's data", async () => {
  for (const operation of ["list", "load", "save"]) {
    const wait = deferred();
    const { controller: c, auth, app, state } = setup({ [operation]: () => wait.promise });
    c.setName("비공개 모델");
    app.getState().classes[0].samples.push({ id: "secret" });
    const running = operation === "save" ? c.save() : operation === "load" ? c.load("old") : c.list();
    auth.setState({ user: { uid: "bob" } });
    wait.resolve(operation === "list" ? [{ id: "old", name: "old" }] : operation === "load" ? saved() : "old");
    await running;
    assert.equal(state().owner, "bob");
    assert.equal(state().id, null);
    assert.equal(state().name, "");
    assert.equal(state().savedName, null);
    assert.deepEqual(state().items, []);
    assert.equal(state().pending, null);
    assert.ok(app.getState().classes.every((c) => c.samples.length === 0));
    assert.equal(app.getState().trained, null);
    auth.setState({ user: null });
    assert.equal(state().owner, null);
  }
});
test("late failures from an old account do not overwrite new account messages", async () => {
  const wait = deferred();
  const { controller: c, auth, state } = setup({ list: () => wait.promise });
  const running = c.list();
  auth.setState({ user: { uid: "bob" } });
  wait.reject(new Error("old account failure"));
  await running;
  assert.equal(state().message, "");
  assert.equal(state().busy, false);
});

test("failed saves preserve current identity and dirty work", async () => {
  const {
    controller: c,
    state,
    app,
  } = setup({
    save: async () => {
      throw { code: "permission-denied" };
    },
  });
  c.setName("실패 테스트");
  const before = fingerprint(app.getState());
  await c.save();
  assert.equal(state().id, null);
  assert.equal(state().busy, false);
  assert.equal(c.dirty(), true);
  assert.equal(fingerprint(app.getState()), before);
  assert.match(state().message, /권한/);
});

test("logout clears a previously saved project, fetched list and pending confirmation", async () => {
  const { controller: c, auth, state } = setup();
  c.setName("이전 계정의 이름");
  await c.save();
  await c.list();
  c.setName("변경");
  await c.load("other");
  assert.ok(state().pending);
  assert.equal(state().items.length, 1);
  auth.setState({ user: null });
  assert.equal(state().savedName, null);
  assert.equal(state().id, null);
  assert.equal(state().name, "");
  assert.deepEqual(state().items, []);
  assert.equal(state().pending, null);
  c.confirmLoad();
  assert.equal(state().id, null);
});

test("v2 preserves lightweight dataset selection and fields, excluding preview rows and images", () => {
  const s = workspace();
  const cyclic = {};
  cyclic.self = cyclic;
  s.dataset = {
    repo: "team/qa",
    config: "subset",
    split: "validation",
    fields: { image: "photo", question: "q", answer: "a", list: "turns" },
    preview: cyclic,
    images: cyclic,
  };
  const data = serializeProject(s);
  assert.equal(data.schemaVersion, 2);
  assert.deepEqual(data.dataset, {
    repo: "team/qa",
    config: "subset",
    split: "validation",
    fields: { image: "photo", question: "q", answer: "a", list: "turns" },
  });
  assert.deepEqual(restoreProject(data).dataset, data.dataset);
  assert.ok(!JSON.stringify(data).includes("preview"));
});

test("legacy v1 documents migrate to v2 with safe defaults, not the current user's dataset", async () => {
  const legacy = saved();
  delete legacy.dataset;
  legacy.schemaVersion = 1;
  const restored = restoreProject(legacy);
  assert.equal(restored.schemaVersion, 2);
  assert.deepEqual(restored.dataset, defaultDataset());
  const { app, controller: c, state } = setup({ load: async () => legacy });
  app.setState({ view: "three", panel: "test", dataset: { ...defaultDataset(), repo: "private/old-selection" } });
  await c.load("legacy");
  c.confirmLoad();
  assert.equal(state().id, "legacy");
  assert.equal(app.getState().view, "canvas");
  assert.equal(app.getState().panel, "data");
  assert.deepEqual(app.getState().dataset, defaultDataset());
});

test("malformed dataset metadata and v2 documents missing dataset are rejected", () => {
  const missing = saved();
  delete missing.dataset;
  assert.throws(() => restoreProject(missing));
  for (const dataset of [
    null,
    { ...defaultDataset(), repo: "https://example.com/photo.jpg" },
    { ...defaultDataset(), fields: { image: {}, question: "q", answer: "a", list: null } },
    { ...defaultDataset(), split: 123 },
  ]) {
    assert.throws(() => serializeProject({ ...workspace(), dataset }));
  }
});

test("dataset selection changes are dirty and account transitions reset v2 panels and metadata", async () => {
  const { app, auth, controller: c } = setup();
  c.setName("모델");
  await c.save();
  app.setState({ dataset: { ...defaultDataset(), split: "validation" } });
  assert.equal(c.dirty(), true);
  app.setState({ view: "code", panel: "block", selectedId: "old" });
  auth.setState({ user: { uid: "bob" } });
  assert.equal(app.getState().view, "canvas");
  assert.equal(app.getState().panel, "data");
  assert.equal(app.getState().selectedId, null);
  assert.deepEqual(app.getState().dataset, defaultDataset());
});

test("skip connections, nested Add and three incoming edges survive round-trip with original IDs", () => {
  const s = workspace(TEMPLATES.find((t) => t.id === "resnet"));
  const add = s.nodes.find((n) => n.data.type === "add");
  const conv = s.nodes.find((n) => n.data.type === "conv2d");
  const extra = makeNode("conv2d", { x: 100, y: 400 }, { filters: 16 });
  const nested = makeNode("add", { x: 300, y: 400 });
  s.nodes.push(extra, nested);
  s.edges.push(makeEdge(s.nodes[0].id, extra.id), makeEdge(extra.id, nested.id), makeEdge(conv.id, nested.id), makeEdge(nested.id, add.id));
  const before = serializeProject(s);
  const restored = restoreProject(JSON.parse(JSON.stringify(before)));
  assert.deepEqual(serializeProject(restored), before);
  assert.equal(restored.edges.filter((e) => e.target === add.id).length, 3);
  assert.equal(restored.edges.filter((e) => e.target === nested.id).length, 2);
  assert.ok(restored.edges.filter((e) => e.source === conv.id).length > 1);
});

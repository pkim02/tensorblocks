import { createStore } from "zustand/vanilla";
import { buildTemplate, newId, TEMPLATES } from "../model/graph.js";
import { fingerprint, localFingerprint, projectName, restoreProject, serializeProject } from "./serialize.js";
import { projectError } from "./repository.js";
import { defaultDataset } from "../data/dataset.js";

export function createProjectController({ app, auth, repository, dispose, stopTraining }) {
  const initial = () => ({
    owner: null,
    id: null,
    name: "",
    savedName: null,
    baseline: null,
    localBaseline: null,
    items: [],
    busy: false,
    message: "",
    pending: null,
  });
  const store = createStore(initial);
  let generation = 0;
  let unsubscribe;
  let previousUid;
  const currentUid = () => (auth.getState().ready ? (auth.getState().user?.uid ?? null) : null);
  function resetWorkspace() {
    const s = app.getState();
    stopTraining();
    dispose(s.trained);
    const template = TEMPLATES.find((t) => t.id === "tm");
    app.setState({
      ...buildTemplate(template),
      templateId: template.id,
      classes: [1, 2].map((i) => ({ id: newId("c"), name: `Class ${i}`, samples: [] })),
      settings: { epochs: 20, batchSize: 16, learningRate: 0.001 },
      selectedId: null,
      preview: null,
      dropHint: null,
      camClassId: null,
      trained: null,
      demoData: false,
      view: "canvas",
      panel: "data",
      dataset: defaultDataset(),
      projectRevision: s.projectRevision + 1,
      train: { status: s.train.status === "training" ? "training" : "idle", logs: [], message: "", epoch: 0 },
    });
  }
  function syncAccount() {
    const uid = currentUid();
    if (uid === previousUid) return;
    generation++;
    // Preserve an anonymous draft on first login. Never carry a signed-in user's workspace to another account.
    if (previousUid) resetWorkspace();
    previousUid = uid;
    store.setState({ ...initial(), owner: uid });
  }
  async function run(task) {
    syncAccount();
    const uid = currentUid();
    if (!uid) {
      store.setState({ message: "먼저 Google 로그인을 해 주세요." });
      return;
    }
    if (store.getState().busy) return;
    const token = generation;
    store.setState({ busy: true, message: "처리 중이에요…" });
    const active = () => token === generation && currentUid() === uid;
    try {
      await task(uid, active);
    } catch (error) {
      if (active()) store.setState({ message: projectError(error) });
    } finally {
      if (active()) store.setState({ busy: false });
    }
  }
  function dirty() {
    const s = store.getState();
    try {
      if (app.getState().classes.some((c) => c.samples.length) || app.getState().trained) return true;
      return !s.localBaseline || localFingerprint(app.getState()) !== s.localBaseline || projectName(s.name) !== s.savedName;
    } catch {
      return true;
    }
  }
  function applyLoaded(data, projectId, active) {
    if (!active()) return;
    if (app.getState().train.status === "training") throw new Error("학습을 마치거나 중지한 뒤 불러와 주세요.");
    const restored = restoreProject(data);
    const name = projectName(data.name);
    dispose(app.getState().trained);
    const { schemaVersion, ...content } = restored;
    app.setState({
      ...content,
      selectedId: null,
      preview: null,
      dropHint: null,
      camClassId: null,
      trained: null,
      demoData: false,
      view: "canvas",
      panel: "data",
      projectRevision: app.getState().projectRevision + 1,
      train: { status: "idle", logs: [], message: "", epoch: 0 },
    });
    store.setState({
      id: projectId,
      name,
      savedName: name,
      baseline: fingerprint(app.getState()),
      localBaseline: localFingerprint(app.getState()),
      pending: null,
      message: "프로젝트를 불러왔어요. 사진은 다시 추가해 주세요.",
    });
  }
  return {
    store,
    dirty,
    start() {
      syncAccount();
      unsubscribe = auth.subscribe(syncAccount);
      return () => {
        unsubscribe?.();
        generation++;
      };
    },
    setName(name) {
      store.setState({ name });
    },
    cancelLoad() {
      store.setState({ pending: null });
    },
    list: () =>
      run(async (uid, active) => {
        const items = await repository.list(uid);
        if (active()) store.setState({ items, message: items.length ? "프로젝트 목록을 불러왔어요." : "아직 저장한 프로젝트가 없어요." });
      }),
    save: (asNew = false) =>
      run(async (uid, active) => {
        const s = store.getState();
        const name = projectName(s.name);
        const state = serializeProject(app.getState());
        const local = localFingerprint(app.getState());
        const id = await repository.save(uid, asNew ? null : s.id, name, state);
        if (active())
          store.setState({
            id,
            savedName: name,
            baseline: fingerprint(state),
            localBaseline: local,
            items: [{ id, name, updatedAt: null }, ...store.getState().items.filter((item) => item.id !== id)],
            message: "프로젝트를 저장했어요. 사진과 학습 결과는 포함되지 않았어요.",
          });
      }),
    load: (projectId) =>
      run(async (uid, active) => {
        if (app.getState().train.status === "training") throw new Error("학습을 마치거나 중지한 뒤 불러와 주세요.");
        const data = await repository.load(uid, projectId);
        if (!active()) return;
        restoreProject(data); // Validate before asking to discard anything.
        projectName(data.name);
        if (dirty()) store.setState({ pending: { data, projectId, uid }, message: "현재 작업을 바꾸기 전에 확인해 주세요." });
        else applyLoaded(data, projectId, active);
      }),
    confirmLoad() {
      syncAccount();
      const pending = store.getState().pending;
      if (pending && pending.uid === currentUid()) applyLoaded(pending.data, pending.projectId, () => pending.uid === currentUid());
    },
  };
}

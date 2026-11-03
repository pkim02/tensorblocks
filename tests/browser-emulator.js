import { connectAuthEmulator, createUserWithEmailAndPassword, getAuth, signInWithEmailAndPassword } from "firebase/auth";
import { connectFirestoreEmulator, getFirestore } from "firebase/firestore";
import { firebaseApp } from "../src/auth/google.js";
import { useApp } from "../src/store.js";
import { makeShapesDataset } from "../src/data/demo.js";

if (!import.meta.env.DEV || firebaseApp?.options.projectId !== "demo-tensorblocks" || !["localhost", "127.0.0.1"].includes(location.hostname)) {
  throw new Error("This test page requires the isolated emulator browser server.");
}
const auth = getAuth(firebaseApp);
connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true });
connectFirestoreEmulator(getFirestore(firebaseApp), "127.0.0.1", 8080);
// Keep metadata restoration tests deterministic without downloading external images.
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = new URL(typeof input === "string" ? input : input.url);
  if (url.origin === "https://datasets-server.huggingface.co") {
    if (url.pathname === "/splits") return Promise.resolve(Response.json({ splits: ["train", "validation"].map((split) => ({ config: "cocoqa", split })) }));
    if (url.pathname === "/rows")
      return Promise.resolve(
        Response.json({
          features: [
            { name: "image", type: { _type: "Image" } },
            { name: "question", type: { _type: "Value" } },
            { name: "answer", type: { _type: "Value" } },
          ],
          rows: [],
        }),
      );
  }
  return realFetch(input, init);
};
const status = document.querySelector("#test-status");
const modelStatus = document.querySelector("#model-test-status");
useApp.subscribe((s) => {
  modelStatus.textContent = `${s.train.status} · ${s.train.epoch} epoch · revision ${s.projectRevision} · trained ${!!s.trained}`;
});
document.querySelector("#small-model").onclick = () => {
  if (useApp.getState().train.status === "training") return;
  const s = useApp.getState();
  s.loadTemplate("cnn");
  for (const n of useApp.getState().nodes) {
    if (n.data.type === "input") s.setParam(n.id, "size", 32);
    if (n.data.type === "conv2d") s.setParam(n.id, "filters", 2);
    if (n.data.type === "dense") s.setParam(n.id, "units", 4);
  }
  s.setClasses(makeShapesDataset(5));
  s.setSetting("epochs", 1);
  useApp.setState({ demoData: true, panel: "train" });
};
for (const name of ["alice", "bob"]) {
  document.querySelector(`#${name}`).onclick = async () => {
    status.textContent = "테스트 사용자 연결 중";
    const email = `${name}@example.test`;
    const password = "local-emulator-only";
    try {
      try {
        await signInWithEmailAndPassword(auth, email, password);
      } catch (error) {
        if (!["auth/user-not-found", "auth/invalid-credential"].includes(error.code)) throw error;
        await createUserWithEmailAndPassword(auth, email, password);
      }
      status.textContent = `${name} · 로컬 Emulator 연결됨`;
    } catch (error) {
      status.textContent = `로컬 테스트 오류: ${error.code}`;
    }
  };
}
await import("../src/main.jsx");
status.textContent = "로컬 Emulator 테스트 준비 완료";

import { connectAuthEmulator, createUserWithEmailAndPassword, getAuth, signInWithEmailAndPassword } from "firebase/auth";
import { connectFirestoreEmulator, getFirestore } from "firebase/firestore";
import { firebaseApp } from "../src/auth/google.js";

if (!import.meta.env.DEV || firebaseApp?.options.projectId !== "demo-tensorblocks" || !["localhost", "127.0.0.1"].includes(location.hostname)) {
  throw new Error("This test page requires the isolated emulator browser server.");
}
const auth = getAuth(firebaseApp);
connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true });
connectFirestoreEmulator(getFirestore(firebaseApp), "127.0.0.1", 8080);
const status = document.querySelector("#test-status");
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

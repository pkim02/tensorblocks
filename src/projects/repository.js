import { collection, doc, getDocFromServer, getDocsFromServer, getFirestore, serverTimestamp, setDoc, updateDoc, Timestamp } from "firebase/firestore";
import { projectName, serializeProject } from "./serialize.js";

// Auth is checked on every operation; the server rules are the actual access boundary.
export function createProjectRepository(app, auth, online = () => globalThis.navigator?.onLine !== false, database = getFirestore) {
  function projects(uid) {
    if (!uid || !auth.getState().ready || auth.getState().user?.uid !== uid) throw new Error("로그인한 사용자만 프로젝트를 사용할 수 있어요.");
    if (!app) throw new Error("Firebase 설정이 필요해요.");
    if (!online()) throw new Error("인터넷 연결을 확인하고 다시 시도해 주세요.");
    return collection(database(app), "users", uid, "projects");
  }
  return {
    async list(uid) {
      const snap = await getDocsFromServer(projects(uid));
      return snap.docs
        .map((d) => ({
          id: d.id,
          name: typeof d.data().name === "string" ? d.data().name : "이름 없는 프로젝트",
          updatedAt: d.data().updatedAt instanceof Timestamp ? d.data().updatedAt : null,
        }))
        .sort((a, b) => (b.updatedAt?.toMillis?.() ?? 0) - (a.updatedAt?.toMillis?.() ?? 0));
    },
    async load(uid, projectId) {
      const snap = await getDocFromServer(doc(projects(uid), projectId));
      if (!snap.exists()) throw new Error("프로젝트가 없거나 삭제되었어요.");
      return snap.data();
    },
    async save(uid, projectId, name, state) {
      const parent = projects(uid);
      const data = { ...serializeProject(state), name: projectName(name), updatedAt: serverTimestamp() };
      const ref = projectId ? doc(parent, projectId) : doc(parent);
      if (projectId) await updateDoc(ref, data);
      else await setDoc(ref, { ...data, createdAt: serverTimestamp() });
      return ref.id;
    },
  };
}

export function projectError(error) {
  const code = error?.code?.replace("firestore/", "");
  if (code === "permission-denied") return "프로젝트 접근 권한이 없어요. 로그인 계정과 Firestore 규칙을 확인해 주세요.";
  if (["unavailable", "deadline-exceeded"].includes(code)) return "서버에 연결하지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요.";
  if (code === "unauthenticated") return "다시 로그인해 주세요.";
  return error?.message || "프로젝트 처리 중 문제가 생겼어요.";
}

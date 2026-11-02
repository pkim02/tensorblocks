import { useEffect, useRef } from "react";
import { useStore } from "zustand";
import { firebaseApp, useAuth } from "../auth/google.js";
import { useApp } from "../store.js";
import { disposeTrained } from "../train/train.js";
import { stopTraining } from "../train/run.js";
import { createProjectController } from "../projects/controller.js";
import { createProjectRepository, projectError } from "../projects/repository.js";

const controller = createProjectController({
  app: useApp,
  auth: useAuth,
  repository: createProjectRepository(firebaseApp, useAuth),
  dispose: disposeTrained,
  stopTraining,
});
const NOTICE = "모델 구성과 설정을 저장합니다. 사진과 학습 결과는 포함되지 않습니다.";

// Stay mounted across view changes so account changes also clear canvas / 3D workspaces.
export default function ProjectControls({ visible }) {
  const dialog = useRef(null);
  const user = useAuth((s) => s.user);
  const ready = useAuth((s) => s.ready);
  const state = useStore(controller.store);
  const app = useApp();
  useEffect(() => controller.start(), []);
  useEffect(() => {
    dialog.current?.close();
  }, [user?.uid, visible]);
  const training = app.train.status === "training";
  const dirty = controller.dirty();
  const close = () => {
    controller.cancelLoad();
    dialog.current?.close();
  };
  const date = (value) => value?.toDate?.().toLocaleString("ko-KR") ?? "시간 확인 중";
  return (
    <>
      {visible && (
        <div style={{ padding: "0 24px 12px", display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12 }}>
          <button
            className="btn outline small"
            disabled={!ready || !user || !firebaseApp}
            onClick={() => {
              dialog.current.showModal();
              controller.list();
            }}
          >
            프로젝트 저장·불러오기
          </button>
          <span className="hint">
            {user ? (state.id ? `${state.name} · ${dirty ? "저장하지 않은 변경 있음" : "저장됨"}` : "새 프로젝트") : "Google 로그인 후 사용할 수 있어요."}
          </span>
          <span className="hint">{NOTICE}</span>
        </div>
      )}
      <dialog ref={dialog} className="dialog" aria-labelledby="project-title" onCancel={controller.cancelLoad}>
        <h2 id="project-title">내 프로젝트</h2>
        <p className="hint">{NOTICE}</p>
        <label className="field">
          프로젝트 이름
          <input value={state.name} maxLength={100} onChange={(e) => controller.setName(e.target.value)} disabled={state.busy} />
        </label>
        <div className="actions">
          <button className="btn outline" disabled={state.busy || !user} onClick={() => controller.save(true)}>
            새 프로젝트로 저장
          </button>
          <button className="btn" disabled={state.busy || !state.id || !user} onClick={() => controller.save()}>
            변경 사항 저장
          </button>
        </div>
        <p role="status" aria-live="polite">
          {state.message}
        </p>
        {state.busy && <p className="hint">연결 상태에 따라 시간이 걸릴 수 있어요.</p>}
        <hr />
        <div className="actions">
          <button className="btn outline small" disabled={state.busy} onClick={controller.list}>
            목록 새로고침
          </button>
        </div>
        {!state.items.length && <p className="hint">저장된 프로젝트 목록이 여기에 표시돼요.</p>}
        <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 12 }}>
          {state.items.map((item) => (
            <li key={item.id} style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <div style={{ flex: 1, overflowWrap: "anywhere" }}>
                <strong>{item.name}</strong>
                <br />
                <span className="hint">{date(item.updatedAt)}</span>
              </div>
              <button
                className="btn outline small"
                disabled={state.busy || training || !!state.pending}
                onClick={() => controller.load(item.id)}
                aria-label={`${item.name} 불러오기`}
              >
                불러오기
              </button>
            </li>
          ))}
        </ul>
        {training && <p className="hint">학습을 마치거나 중지한 뒤 프로젝트를 불러올 수 있어요.</p>}
        {state.pending && (
          <section aria-label="불러오기 확인" className="callout">
            <p>저장하지 않은 작업이 있을 수 있어요. 불러오면 현재 블록·사진·학습 결과가 바뀝니다. 계속할까요?</p>
            <div className="actions">
              <button className="btn outline" onClick={controller.cancelLoad}>
                취소
              </button>
              <button
                className="btn"
                disabled={training || state.busy}
                onClick={() => {
                  try {
                    controller.confirmLoad();
                  } catch (error) {
                    controller.store.setState({ message: projectError(error) });
                  }
                }}
              >
                현재 작업을 바꾸고 불러오기
              </button>
            </div>
          </section>
        )}
        <div className="actions">
          <button className="btn outline" onClick={close}>
            닫기
          </button>
        </div>
      </dialog>
    </>
  );
}

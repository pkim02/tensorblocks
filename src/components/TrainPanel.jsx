import { useApp, useAnalysis } from "../store.js";
import { blockers, startTraining, stopTraining } from "../train/run.js";
import IssueNote from "./IssueNote.jsx";
import AccuracyChart from "./AccuracyChart.jsx";

export default function TrainPanel() {
  const a = useAnalysis();
  const settings = useApp((s) => s.settings);
  const setSetting = useApp((s) => s.setSetting);
  const train = useApp((s) => s.train);
  const busy = train.status === "training";
  // A model with code-only blocks (a VLM, ResNet…) goes to Colab instead of training here.
  const codeOnly = a.ok && a.exportOnly;

  return (
    <>
      <h2 className="panel-h">학습</h2>
      {a.issues
        .filter((i) => i.level === "error")
        .slice(0, 2)
        .map((issue, i) => (
          <IssueNote key={i} issue={issue} compact />
        ))}

      {a.kind === "vlm" ? (
        <p className="callout small">
          VLM은 브라우저에서 돌리기엔 너무 커요. 지금 Colab 노트북은 PaliGemma 2로 내 사진에 답하는 데까지 해요. 데이터셋으로 학습하는 코드는 다음 단계에서
          들어가요.
        </p>
      ) : codeOnly ? (
        <p className="callout small">코드 전용 블록이 있어서 브라우저에서 학습할 수 없어요. Colab 코드로 가져가서 돌려 주세요.</p>
      ) : (
        <>
          <div className="grid fit settings-grid">
            <div className="field">
              <label htmlFor="ep">에폭</label>
              <input
                id="ep"
                type="number"
                min="1"
                max="200"
                value={settings.epochs}
                onChange={(e) => setSetting("epochs", Math.max(1, +e.target.value || 1))}
              />
            </div>
            <div className="field">
              <label htmlFor="bs">배치 크기</label>
              <select id="bs" value={settings.batchSize} onChange={(e) => setSetting("batchSize", +e.target.value)}>
                {[8, 16, 32, 64].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="lr">학습률</label>
              <select id="lr" value={settings.learningRate} onChange={(e) => setSetting("learningRate", +e.target.value)}>
                {[0.0001, 0.0005, 0.001, 0.005, 0.01].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </div>
          </div>
          <p className="small muted">에폭: 전체 사진을 몇 번 반복해서 볼지. 배치: 한 번에 몇 장씩 볼지. 학습률: 한 번에 얼마나 크게 고칠지.</p>
        </>
      )}

      {train.status !== "idle" && !a.exportOnly && (
        <div className="train-status" aria-live="polite">
          {busy && train.epoch > 0 && <progress max={train.total} value={train.epoch} aria-label="학습 진행" />}
          <p className={train.status === "error" ? "error-text" : "small"}>
            {busy && train.epoch > 0 ? `에폭 ${train.epoch} / ${train.total}` : train.message}
          </p>
          <AccuracyChart logs={train.logs} total={train.total} />
        </div>
      )}
    </>
  );
}

// The one primary action of the screen, kept in view whichever tab is open.
export function TrainAction() {
  const a = useAnalysis();
  const classes = useApp((s) => s.classes);
  const train = useApp((s) => s.train);
  const setView = useApp((s) => s.setView);
  const setPanel = useApp((s) => s.setPanel);
  const blocks = blockers(a, classes);
  const busy = train.status === "training";
  const codeOnly = a.ok && a.exportOnly;

  return (
    <div className="work-action">
      {blocks.length > 0 && !busy && !codeOnly && (
        <ul className="blockers">
          {blocks.map((b, i) => (
            <li key={i}>{b.text}</li>
          ))}
        </ul>
      )}
      {codeOnly ? (
        <button className="btn primary block large" onClick={() => setView("code")}>
          Colab 코드로 가져가기
        </button>
      ) : busy ? (
        <button className="btn outline block large" onClick={stopTraining}>
          학습 멈추기
        </button>
      ) : (
        <button
          className="btn primary block large"
          disabled={blocks.length > 0}
          onClick={() => {
            setPanel("train");
            startTraining();
          }}
        >
          {train.status === "done" ? "다시 학습시키기" : "학습시키기"}
        </button>
      )}
    </div>
  );
}

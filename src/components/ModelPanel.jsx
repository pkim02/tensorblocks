import { Fragment } from "react";
import { useApp, useAnalysis } from "../store.js";
import { LAYERS, fmtShape, fmtCount } from "../model/layers.js";
import { TEMPLATES } from "../model/graph.js";
import { blockers, startTraining, stopTraining } from "../train/run.js";
import IssueNote from "./IssueNote.jsx";
import AccuracyChart from "./AccuracyChart.jsx";

export default function ModelPanel() {
  const a = useAnalysis();
  const nodes = useApp((s) => s.nodes);
  const classes = useApp((s) => s.classes);
  const templateId = useApp((s) => s.templateId);
  const loadTemplate = useApp((s) => s.loadTemplate);
  const setView = useApp((s) => s.setView);
  const select = useApp((s) => s.select);
  const settings = useApp((s) => s.settings);
  const setSetting = useApp((s) => s.setSetting);
  const train = useApp((s) => s.train);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const blocks = blockers(a, classes);
  const busy = train.status === "training";

  const openCanvas = (nodeId = null) => {
    select(nodeId);
    setView("canvas");
  };

  return (
    <section className="lane" aria-labelledby="model-h">
      <div className="card step" data-port="train">
        <h2 id="model-h" className="step-h">
          모델과 학습
        </h2>
        <div className="field">
          <label htmlFor="tpl">모델 틀</label>
          <select id="tpl" value={templateId ?? ""} onChange={(e) => loadTemplate(e.target.value)} disabled={busy}>
            {templateId == null && <option value="">직접 고친 모델</option>}
            {TEMPLATES.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
          <p className="hint">{TEMPLATES.find((t) => t.id === templateId)?.text ?? "캔버스에서 직접 고친 모델이에요."}</p>
        </div>

        <ol className="mini-chain" aria-label="모델 구조">
          {a.chain.map((id, i) => {
            const n = byId.get(id);
            const def = LAYERS[n.data.type];
            const info = a.nodeInfo[id];
            const bad = info.issue?.level === "error";
            return (
              <Fragment key={id}>
                {i > 0 && (
                  <li className={`mini-shape ${bad ? "bad" : info.issue ? "warn" : ""}`} aria-hidden="true">
                    {info.blocked ? "?" : fmtShape(info.inShape)}
                    {bad && " ← 안 맞음"}
                  </li>
                )}
                <li>
                  <button className={`mini-block ${bad ? "bad" : ""}`} onClick={() => openCanvas(id)}>
                    <span className="mini-name">{def.name}</span>
                    <span className="mini-sum">{def.summary(n.data.params, { numClasses: a.numClasses })}</span>
                  </button>
                </li>
              </Fragment>
            );
          })}
        </ol>
        <p className="small muted">
          학습할 값 <span className="num-inline">{fmtCount(a.trainableParams)}</span>개
          {a.totalParams > a.trainableParams && <> (고정된 사전학습 값 {fmtCount(a.totalParams - a.trainableParams)}개 별도)</>}
        </p>

        {a.issues
          .filter((i) => i.level === "error")
          .slice(0, 2)
          .map((issue, i) => (
            <IssueNote key={i} issue={issue} compact />
          ))}

        <div className="split-btns">
          <button className="btn outline" onClick={() => openCanvas()}>
            블록 캔버스 열기
          </button>
          <button className="btn outline" onClick={() => setView("three")}>
            3D로 보기
          </button>
        </div>

        <details className="settings">
          <summary>학습 설정</summary>
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
          <p className="hint small">에폭: 전체 사진을 몇 번 반복해서 볼지. 배치: 한 번에 몇 장씩 볼지. 학습률: 한 번에 얼마나 크게 고칠지.</p>
        </details>

        {blocks.length > 0 && !busy && (
          <ul className="blockers">
            {blocks.map((b, i) => (
              <li key={i}>{b.text}</li>
            ))}
          </ul>
        )}

        {busy ? (
          <button className="btn outline block large" onClick={stopTraining}>
            학습 멈추기
          </button>
        ) : (
          <button className="btn primary block large" onClick={startTraining} disabled={blocks.length > 0}>
            {train.status === "done" ? "다시 학습시키기" : "학습시키기"}
          </button>
        )}

        {train.status !== "idle" && (
          <div className="train-status" aria-live="polite">
            {busy && train.epoch > 0 && <progress max={train.total} value={train.epoch} aria-label="학습 진행" />}
            <p className={train.status === "error" ? "error-text" : "small"}>
              {busy && train.epoch > 0 ? `에폭 ${train.epoch} / ${train.total}` : train.message}
            </p>
            <AccuracyChart logs={train.logs} total={train.total} />
          </div>
        )}
      </div>
    </section>
  );
}

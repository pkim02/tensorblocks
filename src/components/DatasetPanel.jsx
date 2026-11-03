import { useEffect, useState } from "react";
import { useApp } from "../store.js";
import { fetchConfigs, fetchPreview, fieldPath, repoName } from "../data/hf.js";

// Data for a VLM: a dataset of photo + question + answer instead of classes of photos.
export default function DatasetPanel() {
  const dataset = useApp((s) => s.dataset);
  const setDataset = useApp((s) => s.setDataset);
  const [draft, setDraft] = useState(dataset.repo);
  const [configs, setConfigs] = useState([]);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState("");
  const [at, setAt] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    setConfigs([]);
    setPreview(null);
    setError("");
    fetchConfigs(dataset.repo, ctrl.signal)
      .then((list) => {
        setConfigs(list);
        const current = list.find((c) => c.config === dataset.config) || list[0];
        const split = current.splits.includes(dataset.split) ? dataset.split : current.splits.includes("train") ? "train" : current.splits[0];
        setDataset({ config: current.config, split });
      })
      .catch((e) => !ctrl.signal.aborted && failed(e));
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.repo]);

  useEffect(() => {
    if (!dataset.config) return;
    const ctrl = new AbortController();
    setPreview(null);
    setError("");
    setAt(0);
    fetchPreview(dataset.repo, dataset.config, dataset.split, ctrl.signal)
      .then((p) => {
        setPreview(p);
        setDataset({ fields: p.fields });
      })
      .catch((e) => !ctrl.signal.aborted && failed(e));
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.repo, dataset.config, dataset.split]);

  const failed = (e) => {
    setDataset({ fields: null });
    setError(e.message || "데이터셋을 읽지 못했어요.");
  };

  const load = (e) => {
    e.preventDefault();
    const repo = repoName(draft);
    if (repo && repo !== dataset.repo) setDataset({ repo, config: null, split: null, fields: null });
  };

  const fields = preview?.fields;
  const example = preview?.examples[at];
  const current = configs.find((c) => c.config === dataset.config);

  return (
    <>
      <h2 className="panel-h">사진 · 질문 · 답</h2>
      <p className="small muted">VLM은 클래스 대신 사진마다 질문과 답이 붙은 데이터셋을 써요.</p>

      <form className="field" onSubmit={load}>
        <label htmlFor="hf-repo">Hugging Face 데이터셋</label>
        <div className="input-row">
          <input id="hf-repo" type="text" value={draft} onChange={(e) => setDraft(e.target.value)} spellCheck="false" />
          <button className="btn outline small">불러오기</button>
        </div>
      </form>

      {configs.length > 0 && (
        <div className="field-pair">
          <div className="field">
            <label htmlFor="hf-config">묶음</label>
            <select
              id="hf-config"
              value={dataset.config ?? ""}
              onChange={(e) => {
                const next = configs.find((c) => c.config === e.target.value);
                setDataset({ config: next.config, split: next.splits.includes("train") ? "train" : next.splits[0], fields: null });
              }}
            >
              {configs.map((c) => (
                <option key={c.config}>{c.config}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="hf-split">나눔</label>
            <select id="hf-split" value={dataset.split ?? ""} onChange={(e) => setDataset({ split: e.target.value, fields: null })}>
              {current?.splits.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {error && <p className="callout warning small">{error}</p>}
      {!error && !preview && <p className="small muted">데이터 구조를 읽는 중…</p>}
      {preview && !fields && <p className="callout warning small">이 데이터에서 사진, 질문, 답 칸을 찾지 못했어요. 다른 묶음이나 데이터셋을 골라 주세요.</p>}

      {fields && (
        <>
          <table className="table small">
            <thead>
              <tr>
                <th>데이터 칸</th>
                <th>들어가는 블록</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono">{fields.image}</td>
                <td>Image Input</td>
              </tr>
              <tr>
                <td className="mono">{fieldPath(fields, "question")}</td>
                <td>Text Input</td>
              </tr>
              <tr>
                <td className="mono">{fieldPath(fields, "answer")}</td>
                <td>Text Output 정답</td>
              </tr>
            </tbody>
          </table>

          {example && (
            <div className="stack-tight">
              <div className="row-between">
                <span className="small">
                  <strong>미리보기</strong>
                </span>
                <span className="cluster tight">
                  <button className="btn ghost small" disabled={at === 0} onClick={() => setAt(at - 1)}>
                    이전
                  </button>
                  <span className="small muted mono">
                    {at + 1} / {preview.examples.length}
                  </span>
                  <button className="btn ghost small" disabled={at === preview.examples.length - 1} onClick={() => setAt(at + 1)}>
                    다음
                  </button>
                </span>
              </div>
              <article className="card qa-card">
                {example.image && <img src={example.image} alt="데이터셋 예제 사진" width="96" height="96" />}
                <dl>
                  <dt>질문</dt>
                  <dd>{example.question}</dd>
                  <dt>답</dt>
                  <dd className="qa-answer">{example.answer}</dd>
                  {example.source && (
                    <>
                      <dt>출처</dt>
                      <dd>{example.source}</dd>
                    </>
                  )}
                </dl>
              </article>
            </div>
          )}
        </>
      )}
    </>
  );
}

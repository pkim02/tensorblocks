import { useEffect, useRef, useState } from "react";
import { useApp, useAnalysis } from "../store.js";
import { useWebcam } from "../data/webcam.js";
import { fileToCanvas, grabFrame } from "../data/images.js";
import { drawShape } from "../data/demo.js";
import { inspect } from "../train/train.js";
import { fmtShape } from "../model/layers.js";

export default function PreviewPanel() {
  const trained = useApp((s) => s.trained);
  const setExportOpen = useApp((s) => s.setExportOpen);
  const a = useAnalysis();

  return (
    <section className="col" aria-labelledby="prev-h">
      <div className="col-head">
        <h2 id="prev-h">3. 미리보기</h2>
      </div>
      {trained ? (
        <Tester trained={trained} stale={trained.signature !== a.signature} />
      ) : (
        <div className="empty">
          <h3>아직 학습한 모델이 없어요</h3>
          <p>학습이 끝나면 여기서 웹캠이나 사진으로 바로 시험하고, 각 블록이 사진을 어떻게 보는지 들여다볼 수 있어요.</p>
        </div>
      )}
      <hr />
      <h3 className="sub-h">코드로 가져가기</h3>
      <p className="small muted">지금 캔버스의 모델을 그대로 PyTorch, Keras 학습 코드로 바꿔 줘요. 학습 전에도 받을 수 있어요.</p>
      <button className="btn outline block" onClick={() => setExportOpen(true)} disabled={!a.ok}>
        코드 내보내기
      </button>
      {!a.ok && <p className="hint small">모델에 고칠 곳이 있으면 코드를 만들 수 없어요.</p>}
    </section>
  );
}

function Tester({ trained, stale }) {
  const demoData = useApp((s) => s.demoData);
  const [source, setSource] = useState(demoData ? "shape" : "cam");
  const [result, setResult] = useState(null);
  const [still, setStill] = useState(null); // canvas for file / shape sources
  const { videoRef, state, error } = useWebcam(source === "cam");
  const fileRef = useRef(null);

  // Webcam: re-run the model about 8 times a second.
  useEffect(() => {
    if (source !== "cam" || state !== "live") return;
    let alive = true;
    const tick = () => {
      if (!alive) return;
      const c = grabFrame(videoRef.current);
      if (c) setResult({ input: c, layers: inspect(trained, c) });
      setTimeout(() => requestAnimationFrame(tick), 120);
    };
    tick();
    return () => (alive = false);
  }, [source, state, trained, videoRef]);

  useEffect(() => {
    if (source === "shape") setStill(drawShape());
    else setStill(null);
    setResult(null);
  }, [source]);

  useEffect(() => {
    if (still) setResult({ input: still, layers: inspect(trained, still) });
  }, [still, trained]);

  const probs = result?.layers[result.layers.length - 1]?.values ?? [];
  const top = probs.indexOf(Math.max(...probs));

  return (
    <div className="stack">
      {stale && (
        <div className="callout warning small">
          <strong>모델이 바뀌었어요</strong>
          학습한 뒤에 블록이나 클래스가 바뀌었어요. 다시 학습해야 반영돼요.
        </div>
      )}
      <div className="segmented" role="radiogroup" aria-label="시험할 입력">
        <label>
          <input type="radio" name="src" checked={source === "cam"} onChange={() => setSource("cam")} /> 웹캠
        </label>
        <label>
          <input type="radio" name="src" checked={source === "file"} onChange={() => setSource("file")} /> 사진 파일
        </label>
        {demoData && (
          <label>
            <input type="radio" name="src" checked={source === "shape"} onChange={() => setSource("shape")} /> 새 도형
          </label>
        )}
      </div>

      <div className="preview-frame">
        {source === "cam" && (state === "error" ? <p className="callout danger small">{error}</p> : <video ref={videoRef} playsInline muted className="cam-video" />)}
        {source !== "cam" && still && <CanvasImage canvas={still} />}
        {source === "file" && !still && <p className="muted small">시험할 사진을 골라 주세요.</p>}
      </div>
      {source === "file" && (
        <>
          <button className="btn outline small" onClick={() => fileRef.current.click()}>
            사진 고르기
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={async (e) => {
              const f = e.target.files[0];
              e.target.value = "";
              if (f) setStill(await fileToCanvas(f));
            }}
          />
        </>
      )}
      {source === "shape" && (
        <button className="btn outline small" onClick={() => setStill(drawShape())}>
          다른 도형 그리기
        </button>
      )}

      <ul className="probs" role="list" aria-label="클래스별 확률">
        {trained.classNames.map((name, i) => (
          <li key={i} className={i === top ? "top" : ""}>
            <span className="prob-name">{name}</span>
            <span className="prob-bar">
              <span style={{ width: `${(probs[i] ?? 0) * 100}%` }} />
            </span>
            <span className="prob-val">{probs.length ? `${Math.round(probs[i] * 100)}%` : "–"}</span>
          </li>
        ))}
      </ul>

      {result && <Inside trained={trained} result={result} />}
    </div>
  );
}

// "모델 속 보기": what each block made of this picture.
function Inside({ trained, result }) {
  const rows = result.layers.slice(0, -1);
  return (
    <details className="inside" open>
      <summary>모델 속 보기</summary>
      <p className="small muted">위에서부터 사진이 블록을 하나씩 지나요. 밝은 곳일수록 그 블록이 강하게 반응한 곳이에요.</p>
      <ol className="inside-list" role="list">
        <li>
          <span className="inside-name">입력 사진</span>
          <span className="inside-shape">
            {trained.backbone ? "224×224×3" : `${trained.size}×${trained.size}×3`}
          </span>
          <div className="maps">
            <CanvasImage canvas={result.input} small />
          </div>
        </li>
        {rows.map((r) => (
          <li key={r.nodeId}>
            <span className="inside-name">{trained.blockNames[r.nodeId]}</span>
            <span className="inside-shape">{r.kind === "maps" ? fmtShape([r.h, r.w, r.c]) : fmtShape([r.n])}</span>
            <div className="maps">{r.kind === "maps" ? r.maps.map((m, i) => <MapTile key={i} data={m} w={r.w} h={r.h} />) : <VecStrip values={r.values} />}</div>
          </li>
        ))}
      </ol>
    </details>
  );
}

function CanvasImage({ canvas, small }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    el.width = canvas.width;
    el.height = canvas.height;
    el.getContext("2d").drawImage(canvas, 0, 0);
  }, [canvas]);
  return <canvas ref={ref} className={small ? "tile" : "still"} />;
}

function MapTile({ data, w, h }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    el.width = w;
    el.height = h;
    const ctx = el.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < data.length; i++) {
      const v = Math.round(data[i] * 255);
      img.data.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [data, w, h]);
  return <canvas ref={ref} className="tile pixel" />;
}

const STRIP = 48;
function VecStrip({ values }) {
  const ref = useRef(null);
  useEffect(() => {
    const v = values.slice(0, STRIP);
    const max = Math.max(1e-6, ...v.map(Math.abs));
    const el = ref.current;
    el.width = v.length;
    el.height = 1;
    const ctx = el.getContext("2d");
    const img = ctx.createImageData(v.length, 1);
    v.forEach((x, i) => {
      const g = Math.round((Math.abs(x) / max) * 255);
      img.data.set([g, g, g, 255], i * 4);
    });
    ctx.putImageData(img, 0, 0);
  }, [values]);
  return (
    <span className="vec">
      <canvas ref={ref} className="strip pixel" />
      {values.length > STRIP && <span className="small muted">앞 {STRIP}칸</span>}
    </span>
  );
}

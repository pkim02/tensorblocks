import { useRef, useState } from "react";
import { useApp, useAnalysis } from "../store.js";
import { useWebcam } from "../data/webcam.js";
import { filesToSamples, grabFrame, makeSample } from "../data/images.js";
import { makeShapesDataset } from "../data/demo.js";
import DatasetPanel from "./DatasetPanel.jsx";

// No confirm box (embedded browsers auto-answer them); the toast offers an undo instead.
export function DemoButton() {
  const loadDemo = () => {
    const { classes, setClasses } = useApp.getState();
    const before = classes;
    const total = classes.reduce((n, c) => n + c.samples.length, 0);
    setClasses(makeShapesDataset(40));
    useApp.setState({ demoData: true });
    window.kit?.toast("예제 도형 120장을 불러왔어요", total > 0 ? { action: "되돌리기", onAction: () => setClasses(before) } : undefined);
  };
  return (
    <button className="btn outline small" onClick={loadDemo}>
      예제 도형 불러오기
    </button>
  );
}

export default function DataPanel() {
  const classes = useApp((s) => s.classes);
  const addClass = useApp((s) => s.addClass);
  const a = useAnalysis();

  if (a.kind === "vlm") return <DatasetPanel />;
  return (
    <>
      <div className="row-between">
        <h2 className="panel-h">클래스별 사진</h2>
        <DemoButton />
      </div>
      {classes.map((c) => (
        <ClassCard key={c.id} cls={c} canDelete={classes.length > 2} />
      ))}
      <button className="btn outline block add-class" onClick={addClass}>
        클래스 추가
      </button>
    </>
  );
}

function ClassCard({ cls, canDelete }) {
  const renameClass = useApp((s) => s.renameClass);
  const removeClass = useApp((s) => s.removeClass);
  const clearSamples = useApp((s) => s.clearSamples);
  const addSamples = useApp((s) => s.addSamples);
  const removeSample = useApp((s) => s.removeSample);
  const camOpen = useApp((s) => s.camClassId === cls.id);
  const setCamClass = useApp((s) => s.setCamClass);
  const fileRef = useRef(null);
  const menuId = `menu-${cls.id}`;

  const onFiles = async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    const samples = await filesToSamples(files);
    if (samples.length) addSamples(cls.id, samples);
    if (samples.length < files.length) window.kit?.toast(`${files.length - samples.length}개 파일은 사진이 아니라서 건너뛰었어요`);
  };

  return (
    <article className="card class-card" data-port="class">
      <div className="class-head">
        <input className="class-name" value={cls.name} onChange={(e) => renameClass(cls.id, e.target.value)} aria-label="클래스 이름" />
        <button className="btn ghost small" commandfor={menuId} command="toggle-popover">
          더보기
        </button>
        <div className="menu" id={menuId} popover="auto">
          <button onClick={() => clearSamples(cls.id)} disabled={!cls.samples.length}>
            사진 모두 지우기
          </button>
          <button className="danger" onClick={() => removeClass(cls.id)} disabled={!canDelete}>
            클래스 삭제
          </button>
        </div>
      </div>
      <p className="class-count">사진 {cls.samples.length}장</p>

      {camOpen && <CamCapture onShot={(s) => addSamples(cls.id, [s])} />}

      <div className="cluster tight">
        <button className={`btn small ${camOpen ? "primary" : "outline"}`} onClick={() => setCamClass(camOpen ? null : cls.id)} aria-pressed={camOpen}>
          {camOpen ? "웹캠 닫기" : "웹캠"}
        </button>
        <button className="btn outline small" onClick={() => fileRef.current.click()}>
          사진 올리기
        </button>
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={onFiles} />
      </div>

      {cls.samples.length > 0 && (
        <ul className="thumbs" role="list">
          {cls.samples.map((s) => (
            <li key={s.id}>
              <button className="thumb" onClick={() => removeSample(cls.id, s.id)} aria-label="이 사진 지우기" title="눌러서 지우기">
                <img src={s.thumb} alt="" width="44" height="44" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

function CamCapture({ onShot }) {
  const { videoRef, state, error } = useWebcam(true);
  const timer = useRef(null);
  const [holding, setHolding] = useState(false);

  const shoot = () => {
    const c = grabFrame(videoRef.current);
    if (c) onShot(makeSample(c));
  };
  const start = (e) => {
    e.preventDefault();
    if (timer.current) return;
    setHolding(true);
    shoot();
    timer.current = setInterval(shoot, 120);
  };
  const stop = () => {
    clearInterval(timer.current);
    timer.current = null;
    setHolding(false);
  };

  if (state === "error") return <p className="callout danger small">{error}</p>;
  return (
    <div className="cam">
      <video ref={videoRef} playsInline muted className="cam-video" />
      <button
        className="btn primary block"
        disabled={state !== "live"}
        onPointerDown={start}
        onPointerUp={stop}
        onPointerLeave={stop}
        onPointerCancel={stop}
        onKeyDown={(e) => (e.key === " " || e.key === "Enter") && (e.preventDefault(), shoot())}
        aria-busy={state === "starting"}
      >
        {holding ? "찍는 중…" : "누르고 있으면 계속 찍혀요"}
      </button>
    </div>
  );
}

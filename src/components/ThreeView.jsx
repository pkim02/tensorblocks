import { useEffect, useMemo, useRef, useState } from "react";
import { useApp, useAnalysis } from "../store.js";
import { LAYERS, fmtShape, fmtCount, torchLines, torchShape } from "../model/layers.js";
import { inspect } from "../train/train.js";
import { createScene } from "../three/scene.js";
import ViewTabs from "./ViewTabs.jsx";

// "3D로 보기": the model as 3D tensors on the right, a step-by-step tour on the left.
// Step 0 is the overview; step i (1…n) is the i-th block in the chain.
export default function ThreeView() {
  const a = useAnalysis();
  const nodes = useApp((s) => s.nodes);
  const classes = useApp((s) => s.classes);
  const trained = useApp((s) => s.trained);
  const [step, setStep] = useState(0);
  const [sample, setSample] = useState(() => randomSample(classes));
  const mountRef = useRef(null);
  const sceneRef = useRef(null);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  // Tour order: the main line, with a side branch (VLM text) just before the block it joins.
  const chain = a.order.map((id) => byId.get(id));
  const joinsAt = new Map(Object.entries(a.side).flatMap(([merge, ids]) => ids.map((id) => [id, a.order.indexOf(merge)])));
  const live = trained && trained.signature === a.signature && sample;
  const acts = useMemo(() => {
    if (!live) return null;
    return Object.fromEntries(inspect(trained, sample.canvas).map((r) => [r.nodeId, r]));
  }, [live, trained, sample]);

  const spec = useMemo(
    () => ({
      items: chain.map((n) => {
        const info = a.nodeInfo[n.id];
        const def = LAYERS[n.data.type];
        const act = acts?.[n.id];
        const base = {
          name: def.name,
          shapeText: info.outShape ? fmtShape(info.outShape) : info.issue?.level === "error" ? "모양이 안 맞아요" : "?",
          shape: info.outShape,
          error: info.issue?.level === "error",
        };
        if (joinsAt.has(n.id)) Object.assign(base, { lane: 1, joins: joinsAt.get(n.id) });
        if (!info.outShape) return { ...base, kind: "ghost" };
        // Pretrained blocks get a dark "frozen" block in front of what they output (3D maps or a vector, as with ViT).
        if (def.pretrained) base.frozen = def.name;
        if (n.data.type === "input") return { ...base, kind: "image", input: sample?.canvas };
        if (n.data.type === "output") return { ...base, kind: "output", values: act?.values, classNames: classes.map((c) => c.name) };
        if (n.data.type === "text") return { ...base, kind: "vector", caption: `“${n.data.params.prompt}”` };
        if (info.outShape.length === 2) return { ...base, kind: "tokens" };
        if (info.outShape.length === 3) return { ...base, kind: "image", maps: act?.kind === "maps" ? act.maps : null };
        return { ...base, kind: "vector", values: act?.values };
      }),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [a, acts, sample, classes],
  );

  useEffect(() => {
    const s = createScene(mountRef.current, { onPick: (i) => setStep(i + 1) });
    sceneRef.current = s;
    return () => s.dispose();
  }, []);

  useEffect(() => {
    sceneRef.current.setModel(spec);
    sceneRef.current.focus(step ? step - 1 : null, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec]);

  useEffect(() => {
    sceneRef.current.focus(step ? step - 1 : null);
  }, [step]);

  // Arrow keys walk the tour.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, select, textarea")) return;
      if (e.key === "ArrowRight") setStep((s) => Math.min(chain.length, s + 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(0, s - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chain.length]);

  const node = step ? chain[step - 1] : null;
  const total = classes.reduce((n, c) => n + c.samples.length, 0);

  return (
    <div className="three-view">
      <header className="work-bar">
        <ViewTabs />
      </header>
      <div className="three-body">
        <aside className="tour" aria-label="모델 둘러보기">
          <div className="tour-nav">
            <span className="small muted">둘러보기</span>
            <span className="tour-count">{step === 0 ? "전체" : `${step} / ${chain.length}`}</span>
          </div>
          <div className="tour-body">
            {node ? (
              <Step
                node={node}
                info={a.nodeInfo[node.id]}
                joined={a.side[node.id] && a.nodeInfo[a.side[node.id].at(-1)].outShape}
                act={acts?.[node.id]}
                live={live}
                classes={classes}
              />
            ) : (
              <Overview a={a} chain={chain} live={live} trained={trained} />
            )}
          </div>

          {live && (
            <div className="tour-sample">
              <img src={sample.thumb} alt="모델에 넣은 사진" width="40" height="40" />
              <span className="small grow">이 사진이 블록을 지나가는 모습이에요.</span>
              <button className="btn outline small" onClick={() => setSample(randomSample(classes, sample))} disabled={total < 2}>
                다른 사진
              </button>
            </div>
          )}

          <ol className="toc" aria-label="블록 목록">
            <li>
              <button className={step === 0 ? "on" : ""} onClick={() => setStep(0)}>
                전체 보기
              </button>
            </li>
            {chain.map((n, i) => (
              <li key={n.id}>
                <button className={`${step === i + 1 ? "on" : ""} ${a.nodeInfo[n.id].issue?.level === "error" ? "bad" : ""}`} onClick={() => setStep(i + 1)}>
                  {LAYERS[n.data.type].name}
                  <span>{a.nodeInfo[n.id].outShape ? fmtShape(a.nodeInfo[n.id].outShape) : "?"}</span>
                </button>
              </li>
            ))}
          </ol>

          <div className="tour-actions">
            <button className="btn outline" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
              이전
            </button>
            <button className="btn primary grow" onClick={() => setStep((s) => Math.min(chain.length, s + 1))} disabled={step === chain.length}>
              {step === 0 ? "둘러보기 시작" : "계속"}
            </button>
          </div>
        </aside>
        <div className="three-stage">
          <div ref={mountRef} className="three-mount" />
          <div className="stage-tools">
            <button className="btn outline small" onClick={() => setStep(0)}>
              전체 보기
            </button>
          </div>
          <p className="stage-hint small">드래그해서 돌리고 스크롤로 확대해요. 블록을 누르면 설명이 나와요.</p>
        </div>
      </div>
    </div>
  );
}

function Overview({ a, chain, live, trained }) {
  const first = a.nodeInfo[chain[0]?.id]?.outShape;
  const lastNode = chain[chain.length - 1];
  return (
    <>
      <h1 className="tour-h">모델을 3D로 보기</h1>
      {a.kind === "vlm" ? (
        <p>
          입력이 두 개예요. 사진은 위쪽 줄에서 SigLIP을 지나 토큰 줄이 되고, 질문은 아래쪽 줄에서 글 토큰이 돼요. 두 줄이 Concat에서 한 줄로 이어진 다음, 언어
          모델이 읽고 끝에서 다음 단어의 확률을 내요.
        </p>
      ) : (
        <p>
          사진 한 장이 블록 {chain.length}개를 왼쪽에서 오른쪽으로 지나가며 모양이 바뀌어요. 처음엔 {first ? fmtShape(first) : "?"} 숫자 표였다가, 끝에서는
          클래스별 확률이 돼요.
        </p>
      )}
      <ul className="legend">
        <li>
          <strong>판이 겹친 상자</strong>: 이미지 모양 데이터. 가로·세로는 이미지 크기, 깊이는 채널 수예요.
        </li>
        <li>
          <strong>세로 막대</strong>: 한 줄로 펴진 벡터. 길수록 숫자가 많아요.
        </li>
        {a.kind === "vlm" && (
          <li>
            <strong>얇고 깊은 벽</strong>: 토큰 줄. 깊이는 토큰 개수, 높이는 토큰 하나의 칸 수예요.
          </li>
        )}
        <li>
          <strong>어두운 상자</strong>: 미리 학습된 모델(고정)이에요.
        </li>
      </ul>
      <p className="small muted">
        크기는 실제 숫자에 비례하지 않아요. 224와 1,280이 한 화면에 들어오도록 완만하게 줄였어요. 정확한 모양은 블록 아래 숫자를 보세요.
      </p>
      {!a.ok && <p className="callout danger small">모델에 고칠 곳이 있어요. 빨간 점선 상자에서 모양이 안 맞아요.</p>}
      {!live && a.exportOnly && a.ok && (
        <p className="callout accent small">이 모델은 브라우저에서 돌리지 않아서 모양만 보여요. 실제로 돌려 보려면 위의 Colab 코드 탭으로 가세요.</p>
      )}
      {!live && !a.exportOnly && (
        <p className="callout accent small">
          {trained ? "모델이 바뀌어서 지금은 모양만 보여요. 다시 학습하면" : "학습을 마치면"} 각 블록에 실제 사진이 지나가며 만든 특징 지도가 그려져요.
        </p>
      )}
      {live && lastNode && <p className="small">지금 블록에 그려진 그림은 아래 사진을 실제로 넣었을 때 각 블록이 만든 결과예요.</p>}
    </>
  );
}

function Step({ node, info, joined, act, live, classes }) {
  const def = LAYERS[node.data.type];
  const t = node.data.type;
  const torch = (info.inShape || def.source) && info.outShape ? torchLines(t, node.data.params, info.inShape, info.outShape) : [];
  let picture = null;
  if (live && t === "input") picture = "빨강·초록·파랑 판 3장이 겹쳐서 사진 한 장이 돼요.";
  else if (live && act?.kind === "maps")
    picture = `채널 ${act.c.toLocaleString("en-US")}개 중 이 사진에 가장 강하게 반응한 ${act.maps.length}개를 판으로 그렸어요. 진할수록 강하게 반응한 곳이에요.`;
  else if (live && t === "output" && act) {
    const best = act.values.indexOf(Math.max(...act.values));
    picture = `이 사진을 '${classes[best]?.name}'(으)로 판단했어요 (${Math.round(act.values[best] * 100)}%).`;
  } else if (live && act?.kind === "vec") picture = "칸이 진할수록 큰 값이에요. 숫자가 많으면 몇 칸씩 묶어서 보여 줘요.";

  return (
    <>
      <h1 className="tour-h">
        {def.name} <span className="muted tour-ko">{def.ko}</span>
      </h1>
      <p>{def.about}</p>
      <dl className="shape-io">
        <dt>들어오는 모양</dt>
        <dd className="mono">
          {info.inShape ? fmtShape(info.inShape) : "–"}
          {joined && ` + ${fmtShape(joined)}`}
        </dd>
        <dt>나가는 모양</dt>
        <dd className="mono">{info.outShape ? fmtShape(info.outShape) : "–"}</dd>
        <dt>{def.frozen ? "고정된 값" : "학습할 값"}</dt>
        <dd className="mono">{info.params ? fmtCount(info.params) : "없음"}</dd>
      </dl>
      {info.issue && <p className={`callout ${info.issue.level === "error" ? "danger" : "warning"} small`}>{info.issue.text}</p>}
      {picture && <p className="small tour-picture">{picture}</p>}
      {torch.length > 0 && (
        <pre className="code-snippet">
          <code>{`# ${info.inShape ? torchShape(info.inShape) : ""} → ${torchShape(info.outShape)}\n${torch.join("\n")}`}</code>
        </pre>
      )}
    </>
  );
}

function randomSample(classes, not) {
  const all = classes.flatMap((c) => c.samples);
  if (!all.length) return null;
  let s = all[Math.floor(Math.random() * all.length)];
  if (not && all.length > 1) while (s === not) s = all[Math.floor(Math.random() * all.length)];
  return s;
}

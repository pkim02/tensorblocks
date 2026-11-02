import { useApp } from "../store.js";
import { analyze } from "../model/graph.js";
import { LAYERS } from "../model/layers.js";
import { trainModel, disposeTrained } from "./train.js";

export const MIN_PER_CLASS = 5;

let stopRequested = false;
export const stopTraining = () => (stopRequested = true);

// What still has to happen before the Train button can work, in plain words.
export function blockers(analysis, classes) {
  const out = [];
  if (!analysis.ok) out.push({ text: "모델에 고칠 곳이 있어요", canvas: true });
  if (analysis.exportOnly)
    out.push({
      text:
        analysis.kind === "vlm"
          ? "VLM은 너무 커서 브라우저에서 돌릴 수 없어요. Colab 코드로 가져가서 돌려 주세요."
          : "코드 전용 블록이 있어서 브라우저에서 학습할 수 없어요. Colab 코드로 가져가서 돌려 주세요.",
      code: true,
    });
  if (analysis.kind === "vlm") return out; // No classes: its data is photo + question + answer.
  if (classes.length < 2) out.push({ text: "클래스가 2개 이상 있어야 해요" });
  const thin = classes.filter((c) => c.samples.length < MIN_PER_CLASS);
  if (thin.length) out.push({ text: `${thin.map((c) => `'${c.name}'`).join(", ")}에 사진이 ${MIN_PER_CLASS}장 이상 필요해요` });
  return out;
}

export async function startTraining() {
  const s = useApp.getState();
  const a = analyze(s.nodes, s.edges, s.classes.length);
  if (blockers(a, s.classes).length) return;

  const byId = new Map(s.nodes.map((n) => [n.id, n]));
  const chainNodes = a.chain.map((id) => byId.get(id));
  const classes = s.classes;
  stopRequested = false;
  s.setTrain({ status: "training", logs: [], message: "준비 중…", epoch: 0, total: +s.settings.epochs });

  const update = (patch) => useApp.getState().setTrain(patch);
  try {
    const result = await trainModel({
      chainNodes,
      classes,
      settings: s.settings,
      onStatus: (message) => update({ message }),
      onEpoch: (epoch, logs) => update({ epoch: epoch + 1, logs: [...useApp.getState().train.logs, { epoch: epoch + 1, ...logs }] }),
      shouldStop: () => stopRequested,
    });
    disposeTrained(useApp.getState().trained);
    useApp.getState().setTrained({
      ...result,
      signature: a.signature,
      classNames: classes.map((c) => c.name),
      blockNames: Object.fromEntries(chainNodes.map((n) => [n.id, LAYERS[n.data.type].name])),
    });
    update({ status: "done", message: stopRequested ? "중간에 멈췄어요. 여기까지 배운 모델로 시험해 볼 수 있어요." : "학습이 끝났어요." });
  } catch (e) {
    console.error(e);
    const offline = /fetch|network|Failed to load/i.test(String(e?.message));
    update({
      status: "error",
      message: offline ? "MobileNet 가중치를 내려받지 못했어요. 인터넷 연결을 확인해 주세요." : `학습 중에 문제가 생겼어요: ${e?.message || e}`,
    });
  }
}

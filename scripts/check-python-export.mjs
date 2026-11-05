// Runs the exported PyTorch and Keras scripts for real (1 epoch on a few made-up photos) and checks
// that the trainable parameter count each framework prints equals the canvas's count.
// Every Colab notebook code cell is also parsed for syntax errors.
//   node scripts/check-python-export.mjs [--no-keras] [model ...]
// Needs python with torch + torchvision (and tensorflow for Keras). PYTHON=path overrides python.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { TEMPLATES, buildTemplate, analyze } from "../src/model/graph.js";
import { LAYERS } from "../src/model/layers.js";
import { pytorchCode, kerasCode, notebookCells } from "../src/export/code.js";

const PYTHON = process.env.PYTHON || "python";
const args = process.argv.slice(2);
const keras = !args.includes("--no-keras");
const only = args.filter((a) => !a.startsWith("--"));

// Models beyond the templates, so every new block and option is exported at least once.
const EXTRA = {
  "cls-sine-causal": {
    spec: [
      ["input", { size: 32 }],
      ["patchembed", { patch: 4, dim: 48 }],
      ["posembed", { mode: "sine" }],
      ["attention", { heads: 8, mask: "causal" }],
      ["dense", { units: 48, activation: "relu" }],
      ["dropout"],
      ["activation", { fn: "gelu" }],
      ["layernorm"],
      ["seqpool", { mode: "cls" }],
      ["dense", { units: 32 }],
      ["output"],
    ],
  },
  "tokens-skip-fix": {
    spec: [["input", { size: 32 }], ["patchembed", { patch: 16, dim: 32 }], ["posembed"], ["dense", { units: 40, activation: "linear" }], ["add"], ["seqpool"], ["output"]],
    sides: [{ into: 4, from: 2, spec: [["dense", { units: 40, activation: "linear" }]] }],
  },
  "small-llm-block": {
    spec: [
      ["input", { size: 32 }],
      ["patchembed", { patch: 8, dim: 32 }],
      ["posembed"],
      ["rmsnorm"],
      ["attention", { heads: 2, mask: "causal" }],
      ["add"],
      ["rmsnorm"],
      ["moe", { experts: 4, topk: 2, hidden: 48 }],
      ["add"],
      ["seqpool"],
      ["rmsnorm"],
      ["output"],
    ],
    sides: [
      { into: 5, from: 2, spec: [] },
      { into: 8, from: 5, spec: [] },
    ],
  },
  "upsample-convtranspose": {
    spec: [
      ["input", { size: 32 }],
      ["conv2d", { filters: 8, stride: 2 }],
      ["upsample", { factor: 2 }],
      ["convtranspose", { filters: 8, kernel: 3, stride: 2, padding: "same" }],
      ["convtranspose", { filters: 4, kernel: 5, stride: 2, padding: "valid", activation: "gelu" }],
      ["gap"],
      ["output"],
    ],
  },
};

const models = [
  ...["cnn", "resnet", "vit", "convnext"].map((id) => [id, TEMPLATES.find((t) => t.id === id)]),
  ...Object.entries(EXTRA),
].filter(([id]) => !only.length || only.includes(id));

const classes = [
  { id: "a", name: "circle", samples: [] },
  { id: "b", name: "square", samples: [] },
];
const settings = { epochs: 1, batchSize: 4, learningRate: 0.001 };
const py = (args, opts = {}) => execFileSync(PYTHON, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], ...opts });

const work = mkdtempSync(path.join(tmpdir(), "tb-export-"));
// 6 photos per class: a bright circle or square on a dark background.
py([
  "-c",
  `import os
from PIL import Image, ImageDraw
for k in ["circle", "square"]:
    os.makedirs(os.path.join("data", k), exist_ok=True)
    for i in range(6):
        im = Image.new("RGB", (96, 96), (20, 20, 40))
        d = ImageDraw.Draw(im)
        box = (16 + i * 2, 20, 70 + i * 2, 74)
        (d.ellipse if k == "circle" else d.rectangle)(box, fill=(230, 200, 60))
        im.save(os.path.join("data", k, f"{i}.png"))`,
], { cwd: work });

let failed = false;
const report = (ok, text) => {
  failed ||= !ok;
  console.log(`${ok ? "✔" : "✖"} ${text}`);
};

for (const [id, t] of models) {
  const { nodes, edges } = buildTemplate(t);
  const a = analyze(nodes, edges, classes.length);
  if (!a.ok) {
    report(false, `${id}: 캔버스에 오류가 있어요 ${a.issues.map((i) => i.title).join(", ")}`);
    continue;
  }
  const types = [...new Set(a.order.map((i) => nodes.find((n) => n.id === i).data.type))].map((x) => LAYERS[x].name).join(", ");
  console.log(`\n${id}: ${types}`);
  const dir = path.join(work, id);
  mkdirSync(dir);

  writeFileSync(path.join(dir, "train_pytorch.py"), pytorchCode(a, nodes, classes, settings));
  try {
    const out = py([path.join(dir, "train_pytorch.py")], { cwd: work });
    const n = +out.match(/학습할 값 ([\d,]+)개/)[1].replace(/,/g, "");
    report(n === a.trainableParams, `PyTorch: 학습할 값 ${n.toLocaleString()} (캔버스 ${a.trainableParams.toLocaleString()}), 1 에폭 학습 완료`);
  } catch (e) {
    report(false, `PyTorch 실행 실패\n${e.stderr || e.message}`);
  }

  if (keras) {
    writeFileSync(path.join(dir, "train_keras.py"), kerasCode(a, nodes, classes, settings));
    try {
      const out = py([path.join(dir, "train_keras.py")], { cwd: work, env: { ...process.env, TF_CPP_MIN_LOG_LEVEL: "3" } });
      const n = +out.match(/Trainable params: ([\d,]+)/)[1].replace(/,/g, "");
      report(n === a.trainableParams, `Keras: Trainable params ${n.toLocaleString()} (캔버스 ${a.trainableParams.toLocaleString()}), 1 에폭 학습 완료`);
    } catch (e) {
      report(false, `Keras 실행 실패\n${(e.stderr || e.message).split("\n").slice(-15).join("\n")}`);
    }
  }

  const cells = notebookCells(a, nodes, classes, settings, { photos: "none" }).filter((c) => c.cell_type === "code");
  writeFileSync(path.join(dir, "cells.json"), JSON.stringify(cells.map((c) => c.source.join(""))));
  try {
    py(["-c", "import ast, json, sys\nfor i, c in enumerate(json.load(open(sys.argv[1], encoding='utf-8'))):\n    ast.parse(c, f'cell {i}')", path.join(dir, "cells.json")]);
    report(true, `Colab 노트북: 코드 셀 ${cells.length}개 문법 확인`);
  } catch (e) {
    report(false, `Colab 노트북 문법 오류\n${e.stderr}`);
  }
}

rmSync(work, { recursive: true, force: true });
process.exit(failed ? 1 : 0);

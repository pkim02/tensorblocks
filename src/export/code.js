// Turns the block chain into training code that runs as-is: a PyTorch script, a Keras
// script, and a Colab notebook (PyTorch) with the user's photos packed inside.
// Every in_channels / in_features is filled in from the shapes the canvas already worked out.
import JSZip from "jszip";
import { LAYERS, torchLines, kerasLines, torchShape, fmtShape, fmtCount } from "../model/layers.js";

const pad = (lines, n) => lines.map((l) => " ".repeat(n) + l);
const pyStr = (s) => JSON.stringify(s);
const IMAGENET = "transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])";

// How each pretrained block is written. `out` is the comment for its output shape.
const BACKBONES = {
  mobilenet: {
    torch: [
      "# MobileNetV2 (구글): ImageNet으로 미리 학습된 특징 추출기 → 1280×7×7",
      "# 브라우저에서는 가벼운 0.35 버전을, 여기서는 표준 버전을 써요. 출력 모양은 같아요.",
      "self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features",
    ],
    norm: IMAGENET,
    keras: {
      make: 'keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet", alpha=0.35)',
      pre: "keras.applications.mobilenet_v2.preprocess_input",
    },
  },
  resnet18: {
    torch: [
      "# ResNet-18: ImageNet으로 미리 학습된 특징 추출기 → 512×7×7",
      "resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)",
      "self.backbone = nn.Sequential(*list(resnet.children())[:-2])",
    ],
    norm: IMAGENET,
    keras: {
      note: "# ResNet-18은 Keras에 기본으로 없어서 ResNet50으로 바꿔 넣었어요 (출력 7×7×2048).",
      make: 'keras.applications.ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet")',
      pre: "keras.applications.resnet.preprocess_input",
    },
  },
  resnet50: {
    torch: [
      "# ResNet-50: ImageNet으로 미리 학습된 특징 추출기 → 2048×7×7",
      "resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)",
      "self.backbone = nn.Sequential(*list(resnet.children())[:-2])",
    ],
    norm: IMAGENET,
    keras: {
      make: 'keras.applications.ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet")',
      pre: "keras.applications.resnet.preprocess_input",
    },
  },
  efficientnet: {
    torch: [
      "# EfficientNet-B0 (구글): ImageNet으로 미리 학습된 특징 추출기 → 1280×7×7",
      "self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features",
    ],
    norm: IMAGENET,
    keras: { make: 'keras.applications.EfficientNetB0(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet")', pre: null },
  },
  vgg16: {
    torch: ["# VGG16: ImageNet으로 미리 학습된 특징 추출기 → 512×7×7", "self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features"],
    norm: IMAGENET,
    keras: {
      make: 'keras.applications.VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet")',
      pre: "keras.applications.vgg16.preprocess_input",
    },
  },
  vit: {
    torch: [
      "# ViT-B/16 (구글, Hugging Face): 사진을 16×16 조각 196개로 읽는 Transformer → 768",
      'self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")',
    ],
    norm: "transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])",
    hf: true,
    keras: null,
  },
};

function describe(analysis, nodes) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const chain = analysis.chain.map((id) => ({ node: byId.get(id), info: analysis.nodeInfo[id] }));
  const size = chain[0].node.data.params.size;
  const bbType = chain[1]?.node.data.type;
  const backbone = BACKBONES[bbType] ? { type: bbType, ...BACKBONES[bbType] } : null;
  return { chain, size, backbone, head: chain.slice(backbone ? 2 : 1) };
}

export function folderName(name) {
  return name.trim().replace(/[\\/:*?"<>|]/g, "_") || "class";
}

// Unique folder per class, in class order (ImageFolder later sorts them by name).
function folders(classes) {
  const used = new Set();
  return classes.map((c) => {
    let name = folderName(c.name);
    while (used.has(name)) name += "_";
    used.add(name);
    return name;
  });
}

// ── PyTorch pieces (shared by the .py script and the notebook) ──────────

function torchParts(analysis, nodes, classes, settings) {
  const { size, backbone, head } = describe(analysis, nodes);
  const body = [];
  for (const { node, info } of head) {
    const t = node.data.type;
    body.push(`# ${LAYERS[t].name} (${LAYERS[t].ko}) → ${torchShape(info.outShape)}`);
    body.push(...torchLines(t, node.data.params, info.inShape, info.outShape));
  }

  const imports = [
    "import torch",
    "from torch import nn",
    "from torch.utils.data import DataLoader, random_split",
    "from torchvision import datasets, transforms, models",
    ...(backbone?.hf ? ["from transformers import ViTModel"] : []),
  ].join("\n");

  const config = `DATA_DIR = "data"
IMAGE_SIZE = ${size}
EPOCHS = ${settings.epochs}
BATCH_SIZE = ${settings.batchSize}
LEARNING_RATE = ${settings.learningRate}
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("쓰는 장치:", device)`;

  const data = `transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # 0~255 → 0~1, 모양은 채널×세로×가로
${backbone ? `    ${backbone.norm},  # 사전학습 때와 같은 기준으로 맞추기\n` : ""}])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
CLASSES = dataset.classes  # 폴더 이름순
NUM_CLASSES = len(CLASSES)
n_val = max(1, int(len(dataset) * 0.15))
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
print(f"사진 {len(dataset)}장, 클래스 {CLASSES}")`;

  const init = backbone
    ? [
        ...backbone.torch,
        "for p in self.backbone.parameters():",
        "    p.requires_grad = False  # 사전학습 가중치는 고정",
        "self.head = nn.Sequential(",
        ...pad(body, 4),
        ")",
      ]
    : ["self.layers = nn.Sequential(", ...pad(body, 4), ")"];
  const forward = backbone
    ? backbone.hf
      ? ["return self.head(self.backbone(pixel_values=x).pooler_output)"]
      : ["return self.head(self.backbone(x))"]
    : ["return self.layers(x)"];
  // A frozen backbone stays in eval mode so its BatchNorm statistics don't drift.
  const trainOverride = backbone
    ? `

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self`
    : "";

  const model = `class TensorBlocksNet(nn.Module):
    # 입력: 3×${size}×${size} (채널×세로×가로)
    def __init__(self):
        super().__init__()
${pad(init, 8).join("\n")}

    def forward(self, x):
${pad(forward, 8).join("\n")}${trainOverride}


model = TensorBlocksNet().to(device)
# Softmax는 CrossEntropyLoss 안에 들어 있어서 모델 끝에는 넣지 않아요.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
print(f"학습할 값 {sum(p.numel() for p in model.parameters() if p.requires_grad):,}개")`;

  const target = backbone ? "model.head" : "model.layers";
  const shapes = `# 블록마다 나오는 모양을 출력해요. TensorBlocks 캔버스의 모양과 같아야 해요.
def show_shape(module, inputs, output):
    print(f"{module.__class__.__name__:<20} → {tuple(output.shape[1:])}")

hooks = [m.register_forward_hook(show_shape) for m in ${target}]
with torch.no_grad():
    model.eval()
    model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
for h in hooks:
    h.remove()`;

  const train = `history = {"train_acc": [], "val_acc": [], "loss": []}
for epoch in range(EPOCHS):
    model.train()
    total, correct, running = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)

    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            v_correct += (model(x).argmax(1) == y).sum().item()
            v_total += len(y)
    history["loss"].append(running / total)
    history["train_acc"].append(correct / total)
    history["val_acc"].append(v_correct / max(v_total, 1))
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {running / total:.3f}  "
          f"train acc {correct / total:.1%}  val acc {v_correct / max(v_total, 1):.1%}")`;

  const save = `torch.save(model.state_dict(), "tensorblocks_model.pt")
print("저장했어요: tensorblocks_model.pt")`;

  return { imports, config, data, model, shapes, train, save, backbone, size, head };
}

export function pytorchCode(analysis, nodes, classes, settings) {
  const t = torchParts(analysis, nodes, classes, settings);
  const names = folders(classes);
  return `"""
TensorBlocks에서 만든 모델을 그대로 옮긴 PyTorch 학습 코드예요.

준비:
  pip install torch torchvision${t.backbone?.hf ? " transformers" : ""}
  data/
    ${classes.map((c, i) => `${names[i]}/   ← '${c.name}' 사진들`).join("\n    ")}
실행:
  python train_pytorch.py
"""
${t.imports}

${t.config}

${t.data}


${t.model}

${t.train}

${t.save}
`;
}

// ── Keras ───────────────────────────────────────────────────

export function kerasCode(analysis, nodes, classes, settings) {
  const { size, backbone, head } = describe(analysis, nodes);
  if (backbone && !backbone.keras) {
    return `# ${LAYERS[backbone.type].name}은(는) Hugging Face의 PyTorch 모델이라 Keras 코드로는 내보낼 수 없어요.\n# PyTorch 코드나 Colab 노트북을 써 주세요.\n`;
  }
  const layersOut = [];
  if (backbone) {
    if (backbone.keras.note) layersOut.push(backbone.keras.note);
    if (backbone.keras.pre) layersOut.push(`layers.Lambda(${backbone.keras.pre}),`);
    else layersOut.push("# EfficientNet은 0~255 사진을 그대로 받아요. 값 맞추기는 모델 안에서 해요.");
    layersOut.push("base,");
  } else {
    layersOut.push("layers.Rescaling(1.0 / 255),");
  }
  for (const { node, info } of head) {
    const t = node.data.type;
    layersOut.push(`# ${LAYERS[t].name} → ${fmtShape(info.outShape)}`, ...kerasLines(t, node.data.params));
  }
  const baseDef = backbone ? `base = ${backbone.keras.make}\nbase.trainable = False  # 사전학습 가중치는 고정\n\n` : "";

  return `"""
TensorBlocks에서 만든 모델을 그대로 옮긴 Keras 학습 코드예요.

준비:
  pip install tensorflow
  data/<클래스 이름>/ 폴더마다 사진을 넣어 주세요.
실행:
  python train_keras.py
"""
import keras
from keras import layers

DATA_DIR = "data"
IMAGE_SIZE = ${size}
EPOCHS = ${settings.epochs}
BATCH_SIZE = ${settings.batchSize}
LEARNING_RATE = ${settings.learningRate}

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    validation_split=0.15, subset="both", seed=42, label_mode="categorical",
)
NUM_CLASSES = len(train_ds.class_names)

${baseDef}model = keras.Sequential([
    keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
${pad(layersOut, 4).join("\n")}
])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("tensorblocks_model.keras")
`;
}

// ── Colab notebook ──────────────────────────────────────────

let cellCount = 0;
const cellId = () => `tb${(cellCount++).toString(36)}${Math.random().toString(36).slice(2, 8)}`;
const lines = (text) => {
  const parts = text.split("\n");
  return parts.map((l, i) => (i < parts.length - 1 ? `${l}\n` : l));
};
const md = (text) => ({ cell_type: "markdown", metadata: { id: cellId() }, source: lines(text) });
const code = (text, { form = false } = {}) => ({
  cell_type: "code",
  execution_count: null,
  metadata: { id: cellId(), ...(form ? { cellView: "form" } : {}) },
  outputs: [],
  source: lines(text),
});

// Photos as JPEG base64, grouped by class folder. At least 128px so IMAGE_SIZE can still be
// raised in Colab without the photos going blurry.
function packPhotos(classes, size) {
  const names = folders(classes);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const out = {};
  classes.forEach((cls, i) => {
    out[names[i]] = cls.samples.map((s) => {
      g.drawImage(s.canvas, 0, 0, size, size);
      return c.toDataURL("image/jpeg", 0.85).split(",")[1];
    });
  });
  return { photos: out };
}

// photos: "embed" packs every photo in, "preview" leaves a stand-in (for showing the notebook
// on screen), "none" gives an upload cell instead.
export function notebookCells(analysis, nodes, classes, settings, { photos = "embed" } = {}) {
  const t = torchParts(analysis, nodes, classes, settings);
  const { chain } = describe(analysis, nodes);
  const count = classes.reduce((n, c) => n + c.samples.length, 0);

  const table = [
    "| 블록 | 나가는 모양 | 학습할 값 |",
    "|---|---|---|",
    ...chain.map(({ node, info }) => {
      const def = LAYERS[node.data.type];
      const p = info.params ? `${fmtCount(info.params)}${def.frozen ? " (고정)" : ""}` : "–";
      return `| ${def.name} (${def.ko}) | ${info.outShape ? fmtShape(info.outShape) : "?"} | ${p} |`;
    }),
  ].join("\n");

  const cells = [
    md(`# TensorBlocks 모델, Colab에서 학습하기

[TensorBlocks](https://tensorblocks.netlify.app)에서 블록으로 만든 모델을 그대로 옮긴 노트북이에요. 위에서부터 한 칸씩 실행하거나, 메뉴의 **런타임 → 모두 실행**을 누르세요.

${table}`),
    md(`## 1. 준비

이 노트북은 GPU(T4)로 열리도록 설정돼 있어요. GPU가 안 잡히면 **런타임 → 런타임 유형 변경 → T4 GPU**를 고르세요.`),
    code(`${t.imports}\n\n${t.config}`),
  ];

  if (photos !== "none" && count > 0) {
    const packed = photos === "embed" ? packPhotos(classes, Math.min(224, Math.max(t.size, 128))) : null;
    const listed = packed
      ? JSON.stringify(packed.photos)
      : `{${folders(classes)
          .map((f, i) => `${pyStr(f)}: [...${classes[i].samples.length}장...]`)
          .join(", ")}}`;
    cells.push(
      md(`## 2. 사진 데이터

TensorBlocks에서 모은 사진 ${count}장(클래스 ${classes.length}개)이 이 노트북 안에 들어 있어요. 아래 칸을 실행하면 \`data/\` 폴더에 풀려요.
내 사진으로 바꾸고 싶으면 왼쪽 파일 창에서 \`data/클래스이름/\` 폴더에 사진을 올리면 돼요.`),
      code(
        `#@title 사진 풀기 (사진 데이터가 들어 있어서 코드는 접어 뒀어요)
import base64, os
PHOTOS = ${listed}
for folder, items in PHOTOS.items():
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)
    for i, b64 in enumerate(items):
        with open(os.path.join(DATA_DIR, folder, f"{i + 1:04d}.jpg"), "wb") as f:
            f.write(base64.b64decode(b64))
print({k: len(v) for k, v in PHOTOS.items()})`,
        { form: true },
      ),
    );
  } else {
    cells.push(
      md(`## 2. 사진 데이터

아직 사진이 없어요. 왼쪽 파일 창에서 \`data/클래스이름/\` 폴더를 만들고 사진을 올리거나, 아래 칸으로 \`data.zip\`을 올려 주세요.`),
      code(`from google.colab import files
import zipfile
up = files.upload()  # data.zip 안에 data/클래스이름/사진들
for name in up:
    zipfile.ZipFile(name).extractall(".")`),
    );
  }

  cells.push(
    code(t.data),
    md(`## 3. 모델

\`in_channels\`, \`in_features\`는 TensorBlocks가 계산해 둔 값 그대로예요. 아래 두 번째 칸은 빈 사진 한 장을 넣어서 블록마다 나오는 모양을 보여 줘요. PyTorch는 채널이 앞에 와서 \`64×64×16\`이 \`(16, 64, 64)\`로 보여요.`),
    code(t.model),
    code(t.shapes),
    md("## 4. 학습"),
    code(t.train),
    code(`import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
# Colab 기본 글꼴에는 한글이 없어서 그래프 글자는 영어로 써요.
plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label="train accuracy")
plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], label="validation accuracy")
plt.xlabel("epoch"); plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3)
plt.show()`),
    md(`## 5. 내 사진으로 시험하기

아래 칸을 실행하고 사진을 올려 보세요. 클래스별 확률이 나와요.`),
    code(`from google.colab import files
from PIL import Image
up = files.upload()
model.eval()
for name in up:
    x = transform(Image.open(name).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    print(name)
    for c, p in sorted(zip(CLASSES, probs), key=lambda z: -z[1]):
        print(f"  {c:<12} {p:.1%}")`),
    md("## 6. 저장하기"),
    code(`${t.save}
from google.colab import files
files.download("tensorblocks_model.pt")`),
  );
  return cells;
}

export function notebookJSON(analysis, nodes, classes, settings, opts) {
  return JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 0,
      metadata: {
        colab: { provenance: [], gpuType: "T4", toc_visible: true },
        kernelspec: { name: "python3", display_name: "Python 3" },
        language_info: { name: "python" },
        accelerator: "GPU",
      },
      cells: notebookCells(analysis, nodes, classes, settings, opts),
    },
    null,
    1,
  );
}

// ── Downloads ───────────────────────────────────────────────

function save(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export function downloadNotebook(analysis, nodes, classes, settings) {
  save(new Blob([notebookJSON(analysis, nodes, classes, settings)], { type: "application/x-ipynb+json" }), "tensorblocks.ipynb");
}

export function downloadText(text, filename) {
  save(new Blob([text], { type: "text/x-python" }), filename);
}

const canvasToBlob = (c) => new Promise((res) => c.toBlob(res, "image/png"));

export async function downloadZip(analysis, nodes, classes, settings) {
  const zip = new JSZip();
  const root = zip.folder("tensorblocks-project");
  const { backbone } = describe(analysis, nodes);
  root.file("train_pytorch.py", pytorchCode(analysis, nodes, classes, settings));
  root.file("train_keras.py", kerasCode(analysis, nodes, classes, settings));
  root.file("tensorblocks.ipynb", notebookJSON(analysis, nodes, classes, settings, { photos: "none" }));
  root.file("requirements.txt", `torch\ntorchvision\n${backbone?.hf ? "transformers\n" : ""}# Keras로 돌릴 때만:\n# tensorflow\n`);
  const names = folders(classes);
  for (let ci = 0; ci < classes.length; ci++) {
    const dir = root.folder(`data/${names[ci]}`);
    const samples = classes[ci].samples;
    for (let i = 0; i < samples.length; i++) {
      dir.file(`${String(i + 1).padStart(4, "0")}.png`, await canvasToBlob(samples[i].canvas));
    }
  }
  save(await zip.generateAsync({ type: "blob" }), "tensorblocks-project.zip");
}

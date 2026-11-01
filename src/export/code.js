// Turns the block chain into a training script that runs as-is on the user's own computer.
// Every in_channels / in_features is filled in from the shapes the canvas already worked out.
import JSZip from "jszip";
import { LAYERS, torchLines, kerasLines, torchShape, fmtShape } from "../model/layers.js";

const pad = (lines, n) => lines.map((l) => " ".repeat(n) + l);

function chainOf(analysis, nodes) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  return analysis.chain.map((id) => ({ node: byId.get(id), info: analysis.nodeInfo[id] }));
}

const pyStr = (s) => JSON.stringify(s);

export function pytorchCode(analysis, nodes, classes, settings) {
  const chain = chainOf(analysis, nodes);
  const size = chain[0].node.data.params.size;
  const backbone = chain[1]?.node.data.type;
  const hasBackbone = backbone === "mobilenet" || backbone === "resnet18";
  const head = chain.slice(hasBackbone ? 2 : 1);

  const body = [];
  for (const { node, info } of head) {
    const t = node.data.type;
    body.push(`# ${LAYERS[t].name} (${LAYERS[t].ko}) → ${torchShape(info.outShape)}`);
    body.push(...torchLines(t, node.data.params, info.inShape, info.outShape));
  }

  const backboneInit = {
    mobilenet: [
      "# MobileNetV2: ImageNet으로 미리 학습된 특징 추출기 (출력 1280×7×7)",
      "# 브라우저에서는 가벼운 0.35 버전을, 여기서는 표준 버전을 써요. 출력 모양은 같아요.",
      "self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features",
    ],
    resnet18: [
      "# ResNet-18: ImageNet으로 미리 학습된 특징 추출기 (출력 512×7×7)",
      "resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)",
      "self.backbone = nn.Sequential(*list(resnet.children())[:-2])",
    ],
  }[backbone];

  const init = hasBackbone
    ? [
        ...backboneInit,
        "for p in self.backbone.parameters():",
        "    p.requires_grad = False  # 사전학습 가중치는 고정",
        "self.head = nn.Sequential(",
        ...pad(body, 4),
        ")",
      ]
    : ["self.layers = nn.Sequential(", ...pad(body, 4), ")"];

  const forward = hasBackbone ? ["return self.head(self.backbone(x))"] : ["return self.layers(x)"];
  const normalize = hasBackbone ? "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 기준으로 맞추기\n" : "";

  return `"""
TensorBlocks에서 만든 모델을 그대로 옮긴 PyTorch 학습 코드예요.

준비:
  pip install torch torchvision
  data/
    ${classes.map((c) => `${folderName(c.name)}/   ← '${c.name}' 사진들`).join("\n    ")}
실행:
  python train_pytorch.py
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DATA_DIR = "data"
IMAGE_SIZE = ${size}
EPOCHS = ${settings.epochs}
BATCH_SIZE = ${settings.batchSize}
LEARNING_RATE = ${settings.learningRate}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # 0~255 → 0~1, 모양은 채널×세로×가로
${normalize}])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES = len(dataset.classes)  # 폴더 이름순: ${pyStr(classes.map((c) => folderName(c.name)).sort())}
n_val = max(1, int(len(dataset) * 0.15))
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)


class TensorBlocksNet(nn.Module):
    # 입력: 3×${size}×${size} (채널×세로×가로)
    def __init__(self):
        super().__init__()
${pad(init, 8).join("\n")}

    def forward(self, x):
${pad(forward, 8).join("\n")}


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = TensorBlocksNet().to(device)
# Softmax는 CrossEntropyLoss 안에 들어 있어서 모델 끝에는 넣지 않아요.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

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
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {running / total:.3f}  "
          f"train acc {correct / total:.1%}  val acc {v_correct / max(v_total, 1):.1%}")

torch.save(model.state_dict(), "tensorblocks_model.pt")
print("저장했어요: tensorblocks_model.pt")
`;
}

export function kerasCode(analysis, nodes, classes, settings) {
  const chain = chainOf(analysis, nodes);
  const size = chain[0].node.data.params.size;
  const backbone = chain[1]?.node.data.type;
  const hasBackbone = backbone === "mobilenet" || backbone === "resnet18";
  const head = chain.slice(hasBackbone ? 2 : 1);

  const layersOut = [];
  if (hasBackbone && backbone === "mobilenet") {
    layersOut.push("layers.Lambda(keras.applications.mobilenet_v2.preprocess_input),", "base,  # → 7×7×1280");
  } else if (hasBackbone) {
    layersOut.push(
      "# ResNet-18은 Keras에 기본으로 없어서 ResNet50으로 바꿔 넣었어요 (출력 7×7×2048).",
      "layers.Lambda(keras.applications.resnet.preprocess_input),",
      "base,",
    );
  } else {
    layersOut.push("layers.Rescaling(1.0 / 255),");
  }
  for (const { node, info } of head) {
    const t = node.data.type;
    layersOut.push(`# ${LAYERS[t].name} → ${fmtShape(info.outShape)}`, ...kerasLines(t, node.data.params));
  }

  const baseDef = !hasBackbone
    ? ""
    : backbone === "mobilenet"
      ? `base = keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet", alpha=0.35)
base.trainable = False  # 사전학습 가중치는 고정

`
      : `base = keras.applications.ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

`;

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

export function folderName(name) {
  return name.trim().replace(/[\\/:*?"<>|]/g, "_") || "class";
}

const canvasToBlob = (c) => new Promise((res) => c.toBlob(res, "image/png"));

export async function downloadZip(analysis, nodes, classes, settings) {
  const zip = new JSZip();
  const root = zip.folder("tensorblocks-project");
  root.file("train_pytorch.py", pytorchCode(analysis, nodes, classes, settings));
  root.file("train_keras.py", kerasCode(analysis, nodes, classes, settings));
  root.file("requirements.txt", "torch\ntorchvision\n# Keras로 돌릴 때만:\n# tensorflow\n");
  const used = new Set();
  for (const c of classes) {
    let name = folderName(c.name);
    while (used.has(name)) name += "_";
    used.add(name);
    const dir = root.folder(`data/${name}`);
    for (let i = 0; i < c.samples.length; i++) {
      dir.file(`${String(i + 1).padStart(4, "0")}.png`, await canvasToBlob(c.samples[i].canvas));
    }
  }
  const blob = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "tensorblocks-project.zip";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

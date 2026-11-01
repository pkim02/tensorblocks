// Block catalog: what each block is, how it changes the tensor shape, what can go wrong,
// and how it is written in TF.js / PyTorch / Keras.
// Shapes are channels-last arrays: [H, W, C] for images, [N] for vectors.

export const fmtShape = (s) => (!s ? "?" : s.length === 1 ? s[0].toLocaleString("en-US") : s.join("×"));

export const fmtCount = (n = 0) => {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}K`;
  return n.toLocaleString("en-US");
};

const is3D = (s) => s && s.length === 3;
const is1D = (s) => s && s.length === 1;

const ACTIVATIONS = [
  { value: "relu", label: "ReLU" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "tanh", label: "Tanh" },
  { value: "linear", label: "없음" },
];

const torchAct = { relu: "nn.ReLU()", sigmoid: "nn.Sigmoid()", tanh: "nn.Tanh()" };

// Shared errors -----------------------------------------------------------

function needsImage(name, inShape) {
  return {
    level: "error",
    title: "여기 모양이 안 맞아요",
    text: `${name}은(는) 세로×가로×채널 모양의 이미지를 받아야 해요. 그런데 이미 한 줄로 펴진 벡터(${fmtShape(inShape)})가 들어오고 있어요.`,
    fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
  };
}

function needsVector(name, inShape) {
  const [h, w, c] = inShape;
  return {
    level: "error",
    title: "여기 모양이 안 맞아요",
    text: `${name}은(는) 한 줄짜리 벡터만 받아요. 지금은 ${fmtShape(inShape)} 모양(3차원)이 들어와요. PyTorch였다면 "mat1 and mat2 shapes cannot be multiplied" 에러가 났을 거예요.`,
    fixes: [
      {
        label: `GlobalAvgPool 넣기 → ${fmtShape([c])}`,
        note: "추천: 가볍고 과적합이 적어요",
        action: { kind: "insertBefore", type: "gap" },
      },
      {
        label: `Flatten 넣기 → ${fmtShape([h * w * c])}`,
        action: { kind: "insertBefore", type: "flatten" },
      },
    ],
  };
}

function tooSmall(name, inShape, extraFixes = []) {
  return {
    level: "error",
    title: "이미지가 사라져요",
    text: `${name}을(를) 지나면 ${fmtShape(inShape)} 이미지의 크기가 0 이하가 돼요. 앞에서 너무 많이 줄였어요.`,
    fixes: [...extraFixes, { label: "이 블록 지우기", action: { kind: "remove" } }],
  };
}

function pretrainedRules(name, size, inShape, ctx) {
  if (!ctx.afterInput) {
    return {
      level: "error",
      title: "자리가 틀렸어요",
      text: `${name}은(는) 원본 사진을 받도록 미리 학습됐어요. '이미지 입력' 바로 다음에 놓아 주세요.`,
      fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
    };
  }
  if (inShape[0] !== size || inShape[1] !== size) {
    return {
      level: "error",
      title: "사진 크기가 안 맞아요",
      text: `${name}은(는) ${size}×${size} 사진을 받아요. 지금 입력은 ${fmtShape(inShape)}이에요.`,
      fixes: [{ label: `입력 크기를 ${size}로 바꾸기`, action: { kind: "setInputSize", value: size } }],
    };
  }
  return null;
}

// Catalog -----------------------------------------------------------------

export const LAYERS = {
  input: {
    name: "Image Input",
    ko: "이미지 입력",
    group: "io",
    defaults: { size: 96 },
    fields: [
      {
        key: "size",
        label: "이미지 크기",
        type: "select",
        options: [32, 64, 96, 128, 224].map((v) => ({ value: v, label: `${v}×${v}` })),
        hint: "클수록 자세히 보지만 학습이 느려져요.",
      },
    ],
    about: "사진을 숫자 표로 바꿔요. 64×64×3은 세로 64칸, 가로 64칸, 색(빨강·초록·파랑) 3칸이라는 뜻이에요.",
    summary: (p) => `${p.size}×${p.size} 컬러 사진`,
    infer: (_, p) => ({ out: [p.size, p.size, 3] }),
    params: () => 0,
  },

  mobilenet: {
    name: "MobileNetV2",
    ko: "사전학습 모델",
    group: "pretrained",
    defaults: {},
    fields: [],
    pretrained: true,
    about:
      "구글이 이미지 100만 장 이상으로 미리 학습시킨 '눈'이에요. Teachable Machine도 이걸 써요. 사진을 7×7 칸마다 1,280가지 특징으로 바꿔 줘요. 가중치는 고정해서 학습이 빨라요.",
    summary: () => "ImageNet으로 학습됨, 고정",
    infer: (inShape, _, ctx) => {
      if (!is3D(inShape)) return { error: needsImage("MobileNetV2", inShape) };
      const bad = pretrainedRules("MobileNetV2", 224, inShape, ctx);
      return bad ? { error: bad } : { out: [7, 7, 1280] };
    },
    params: () => 410208,
    frozen: true,
  },

  resnet18: {
    name: "ResNet-18",
    ko: "사전학습 모델",
    group: "pretrained",
    defaults: {},
    fields: [],
    pretrained: true,
    exportOnly: true,
    about: "마이크로소프트 연구팀이 만든 유명한 이미지 모델이에요. 브라우저에서 돌리기엔 무거워서, PyTorch 코드로 내보내 내 컴퓨터에서 학습해요.",
    summary: () => "코드 내보내기 전용",
    infer: (inShape, _, ctx) => {
      if (!is3D(inShape)) return { error: needsImage("ResNet-18", inShape) };
      const bad = pretrainedRules("ResNet-18", 224, inShape, ctx);
      return bad ? { error: bad } : { out: [7, 7, 512] };
    },
    params: () => 11176512,
    frozen: true,
  },

  conv2d: {
    name: "Conv2D",
    ko: "특징 찾기",
    group: "image",
    defaults: { filters: 16, kernel: 3, stride: 1, padding: "same", activation: "relu" },
    fields: [
      { key: "filters", label: "필터 수", type: "number", min: 1, max: 256, hint: "찾을 특징의 종류 수. 출력 채널이 돼요." },
      {
        key: "kernel",
        label: "커널 크기",
        type: "select",
        options: [1, 3, 5, 7].map((v) => ({ value: v, label: `${v}×${v}` })),
        hint: "한 번에 들여다보는 창의 크기",
      },
      {
        key: "stride",
        label: "보폭 (stride)",
        type: "select",
        options: [1, 2].map((v) => ({ value: v, label: String(v) })),
        hint: "2로 하면 출력 크기가 절반이 돼요.",
      },
      {
        key: "padding",
        label: "가장자리",
        type: "select",
        options: [
          { value: "same", label: "same (크기 유지)" },
          { value: "valid", label: "valid (조금 줄어듦)" },
        ],
      },
      { key: "activation", label: "활성화 함수", type: "select", options: ACTIVATIONS },
    ],
    about: "작은 창(커널)으로 사진을 훑으면서 선, 모서리, 무늬 같은 특징을 찾아요. 필터 수만큼 '특징 지도'가 생겨요.",
    summary: (p) => `필터 ${p.filters}개, ${p.kernel}×${p.kernel}${p.stride > 1 ? `, 보폭 ${p.stride}` : ""}`,
    infer: (inShape, p) => {
      if (!is3D(inShape)) return { error: needsImage("Conv2D", inShape) };
      const [h, w] = inShape;
      const k = +p.kernel,
        s = +p.stride;
      const size = (n) => (p.padding === "same" ? Math.ceil(n / s) : Math.floor((n - k) / s) + 1);
      const oh = size(h),
        ow = size(w);
      if (oh < 1 || ow < 1) {
        const fixes = p.padding === "valid" ? [{ label: "가장자리를 same으로", action: { kind: "setParam", key: "padding", value: "same" } }] : [];
        return { error: tooSmall("Conv2D", inShape, fixes) };
      }
      return { out: [oh, ow, +p.filters] };
    },
    params: (inShape, p) => p.kernel * p.kernel * inShape[2] * p.filters + +p.filters,
  },

  maxpool: {
    name: "MaxPool2D",
    ko: "크기 줄이기",
    group: "image",
    defaults: { pool: 2 },
    fields: [
      {
        key: "pool",
        label: "창 크기",
        type: "select",
        options: [2, 3, 4].map((v) => ({ value: v, label: `${v}×${v}` })),
        hint: "2×2면 가로·세로가 절반이 돼요.",
      },
    ],
    about: "2×2 칸마다 가장 큰 값만 남겨서 이미지를 작게 만들어요. 계산이 줄고, 위치가 조금 달라도 같은 걸로 알아봐요.",
    summary: (p) => `${p.pool}×${p.pool}마다 최댓값`,
    infer: (inShape, p) => {
      if (!is3D(inShape)) return { error: needsImage("MaxPool2D", inShape) };
      const [h, w, c] = inShape;
      const oh = Math.floor(h / p.pool),
        ow = Math.floor(w / p.pool);
      if (oh < 1 || ow < 1) return { error: tooSmall("MaxPool2D", inShape) };
      return { out: [oh, ow, c] };
    },
    params: () => 0,
  },

  gap: {
    name: "GlobalAvgPool",
    ko: "평균으로 요약",
    group: "image",
    defaults: {},
    fields: [],
    about: "특징 지도 하나를 평균 내서 숫자 하나로 만들어요. 7×7×1280이면 1,280칸짜리 벡터가 돼요. 학습할 값이 없어서 가벼워요.",
    summary: () => "채널마다 평균 1개",
    infer: (inShape) => {
      if (!is3D(inShape)) return { error: needsImage("GlobalAvgPool", inShape) };
      return { out: [inShape[2]] };
    },
    params: () => 0,
  },

  flatten: {
    name: "Flatten",
    ko: "한 줄로 펴기",
    group: "image",
    defaults: {},
    fields: [],
    about: "3차원 표를 한 줄로 쭉 펴요. 7×7×64면 3,136칸이 돼요. Dense에 넣기 전에 필요해요.",
    summary: () => "3차원 → 1차원",
    infer: (inShape) => {
      if (is1D(inShape)) return { out: inShape };
      const n = inShape.reduce((a, b) => a * b, 1);
      const out = [n];
      if (n > 20000) {
        return {
          out,
          warn: {
            level: "warn",
            title: "너무 길게 펴졌어요",
            text: `펴진 벡터가 ${fmtShape(out)}칸이에요. 다음 Dense의 학습할 값이 아주 많아져서 브라우저 학습이 느리고 외워버리기(과적합) 쉬워요.`,
            fixes: [{ label: "GlobalAvgPool로 바꾸기", note: `→ ${fmtShape([inShape[2]])}`, action: { kind: "replace", type: "gap" } }],
          },
        };
      }
      return { out };
    },
    params: () => 0,
  },

  dense: {
    name: "Dense",
    ko: "판단하기",
    group: "vector",
    defaults: { units: 64, activation: "relu" },
    fields: [
      { key: "units", label: "뉴런 수", type: "number", min: 1, max: 2048, hint: "출력 벡터의 길이" },
      { key: "activation", label: "활성화 함수", type: "select", options: ACTIVATIONS },
    ],
    about: "들어온 모든 숫자를 모든 뉴런과 연결해서 판단해요. PyTorch에서는 nn.Linear예요. 한 줄짜리 벡터만 받아요.",
    summary: (p) => `뉴런 ${p.units}개`,
    infer: (inShape, p) => {
      if (!is1D(inShape)) return { error: needsVector("Dense", inShape) };
      return { out: [+p.units] };
    },
    params: (inShape, p) => inShape[0] * p.units + +p.units,
  },

  dropout: {
    name: "Dropout",
    ko: "일부러 잊기",
    group: "vector",
    defaults: { rate: 0.3 },
    fields: [
      {
        key: "rate",
        label: "끄는 비율",
        type: "select",
        options: [0.1, 0.2, 0.3, 0.5].map((v) => ({ value: v, label: `${Math.round(v * 100)}%` })),
      },
    ],
    about: "학습할 때마다 뉴런 일부를 무작위로 꺼요. 데이터를 통째로 외워버리는 것(과적합)을 막아 줘요. 모양은 그대로예요.",
    summary: (p) => `${Math.round(p.rate * 100)}% 끄기`,
    infer: (inShape) => ({ out: inShape }),
    params: () => 0,
  },

  batchnorm: {
    name: "BatchNorm",
    ko: "값 고르게",
    group: "vector",
    defaults: {},
    fields: [],
    about: "값들의 크기를 고르게 맞춰서 학습이 흔들리지 않게 해요. 이미지 뒤에도, 벡터 뒤에도 쓸 수 있어요.",
    summary: () => "평균 0, 분산 1로",
    infer: (inShape) => ({ out: inShape }),
    params: (inShape) => 2 * inShape[inShape.length - 1],
  },

  output: {
    name: "Output",
    ko: "분류 출력",
    group: "io",
    defaults: {},
    fields: [],
    about: "클래스 수만큼 점수를 내고, 합이 100%인 확률로 바꿔요(Softmax). 클래스 수는 데이터에서 자동으로 맞춰져요.",
    summary: (_, ctx) => `클래스 ${ctx.numClasses}개, Softmax`,
    infer: (inShape, _, ctx) => {
      if (!is1D(inShape)) return { error: needsVector("분류 출력", inShape) };
      return { out: [ctx.numClasses] };
    },
    params: (inShape, _, ctx) => inShape[0] * ctx.numClasses + ctx.numClasses,
  },
};

export const PALETTE = [
  { title: "사전학습 모델", items: ["mobilenet", "resnet18"] },
  { title: "이미지 층", items: ["conv2d", "maxpool", "gap", "flatten"] },
  { title: "벡터 층", items: ["dense", "dropout", "batchnorm"] },
  { title: "입력과 출력", items: ["input", "output"] },
];

// Code lines --------------------------------------------------------------

// PyTorch is channels-first; shapes in comments are written C×H×W to match.
export const torchShape = (s) => (s.length === 3 ? `${s[2]}×${s[0]}×${s[1]}` : String(s[0]));

export function torchLines(type, p, inShape, outShape) {
  const act = (a) => (torchAct[a] ? [`${torchAct[a]},`] : []);
  switch (type) {
    case "conv2d": {
      const pad = p.padding === "same" ? Math.floor(p.kernel / 2) : 0;
      return [`nn.Conv2d(${inShape[2]}, ${p.filters}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${pad}),`, ...act(p.activation)];
    }
    case "maxpool":
      return [`nn.MaxPool2d(${p.pool}),`];
    case "gap":
      return ["nn.AdaptiveAvgPool2d(1),", "nn.Flatten(),"];
    case "flatten":
      return ["nn.Flatten(),"];
    case "dense":
      return [`nn.Linear(${inShape[0]}, ${p.units}),`, ...act(p.activation)];
    case "dropout":
      return [`nn.Dropout(${p.rate}),`];
    case "batchnorm":
      return [inShape.length === 3 ? `nn.BatchNorm2d(${inShape[2]}),` : `nn.BatchNorm1d(${inShape[0]}),`];
    case "output":
      return [`nn.Linear(${inShape[0]}, NUM_CLASSES),`];
    default:
      return [];
  }
}

export function kerasLines(type, p) {
  const act = (a) => (a === "linear" ? "" : `, activation="${a}"`);
  switch (type) {
    case "conv2d":
      return [`layers.Conv2D(${p.filters}, ${p.kernel}, strides=${p.stride}, padding="${p.padding}"${act(p.activation)}),`];
    case "maxpool":
      return [`layers.MaxPooling2D(${p.pool}),`];
    case "gap":
      return ["layers.GlobalAveragePooling2D(),"];
    case "flatten":
      return ["layers.Flatten(),"];
    case "dense":
      return [`layers.Dense(${p.units}${act(p.activation)}),`];
    case "dropout":
      return [`layers.Dropout(${p.rate}),`];
    case "batchnorm":
      return ["layers.BatchNormalization(),"];
    case "output":
      return ['layers.Dense(NUM_CLASSES, activation="softmax"),'];
    default:
      return [];
  }
}

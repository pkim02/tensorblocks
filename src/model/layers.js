// Block catalog: what each block is, how it changes the tensor shape, what can go wrong,
// and how it is written in TF.js / PyTorch / Keras.
// Shapes are channels-last arrays: [H, W, C] for images, [N] for vectors, and [T, D] for a
// row of T tokens that are D numbers long (what a VLM passes around).

export const fmtShape = (s) => (!s ? "?" : s.length === 1 ? s[0].toLocaleString("en-US") : s.join("×"));

export const fmtCount = (n = 0) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}K`;
  return n.toLocaleString("en-US");
};

const is3D = (s) => s && s.length === 3;
const is1D = (s) => s && s.length === 1;
const is2D = (s) => s && s.length === 2;
const describeShape = (s) => (is2D(s) ? `토큰 줄(${fmtShape(s)})` : is1D(s) ? `한 줄로 펴진 벡터(${fmtShape(s)})` : `이미지 모양(${fmtShape(s)})`);

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
    text: `${name}은(는) 세로×가로×채널 모양의 이미지를 받아야 해요. 그런데 ${describeShape(inShape)}가 들어오고 있어요.`,
    fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
  };
}

function needsVector(name, inShape) {
  if (!is3D(inShape)) {
    return {
      level: "error",
      title: "여기 모양이 안 맞아요",
      text: `${name}은(는) 한 줄짜리 벡터만 받아요. 지금은 ${describeShape(inShape)}가 들어와요.`,
      fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
    };
  }
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

function needsTokens(name, inShape) {
  const fixes = is3D(inShape)
    ? [
        {
          label: `To Tokens 넣기 → ${fmtShape([inShape[0] * inShape[1], inShape[2]])}`,
          note: "칸 하나를 토큰 하나로",
          action: { kind: "insertBefore", type: "totokens" },
        },
      ]
    : [];
  return {
    level: "error",
    title: "토큰 줄이 아니에요",
    text: `${name}은(는) 토큰 개수×칸 수 모양의 토큰 줄을 받아요. 지금은 ${describeShape(inShape)}가 들어와요.`,
    fixes: [...fixes, { label: "이 블록 지우기", action: { kind: "remove" } }],
  };
}

// Gemma 2 2B, the language half of PaliGemma 2 (3B). Numbers from the Hugging Face model.
export const GEMMA = { dim: 2304, vocab: 257216 };

// Roughly how many tokens a prompt becomes, counting the start token and the "\n" PaliGemma adds.
export function promptTokens(prompt) {
  let n = 2;
  for (const w of String(prompt).trim().split(/\s+/).filter(Boolean)) n += /^[\x00-\x7f]+$/.test(w) ? Math.ceil(w.length / 6) : Math.ceil(w.length / 2);
  return n;
}

function needsTokenIds(inShape) {
  return {
    level: "error",
    title: "글이 아니에요",
    text: `Embedding은 '글 입력'에서 나온 단어 조각 번호를 받아요. 지금은 ${describeShape(inShape)}가 들어와요.`,
    fixes: [{ label: "이 블록 지우기", action: { kind: "remove" } }],
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

// Pretrained models that are too big for the browser: shape checks and code export only.
function exportBackbone({ name, ko = "사전학습 모델", about, out, params }) {
  return {
    name,
    ko,
    group: "pretrained",
    defaults: {},
    fields: [],
    pretrained: true,
    exportOnly: true,
    about: `${about} 브라우저에서 돌리기엔 무거워서, 코드로 내보내 Colab이나 내 컴퓨터에서 학습해요.`,
    summary: () => "코드 내보내기 전용",
    infer: (inShape, _, ctx) => {
      if (!is3D(inShape)) return { error: needsImage(name, inShape) };
      const bad = pretrainedRules(name, 224, inShape, ctx);
      return bad ? { error: bad } : { out };
    },
    params: () => params,
    frozen: true,
  };
}

const POOL_FIELD = {
  key: "pool",
  label: "창 크기",
  type: "select",
  options: [2, 3, 4].map((v) => ({ value: v, label: `${v}×${v}` })),
  hint: "2×2면 가로·세로가 절반이 돼요.",
};

const KERNEL_FIELDS = [
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
];

// Spatial size after a k×k window with stride s (TF "same"/"valid" rules).
function convOut(name, inShape, p, channels) {
  const [h, w] = inShape;
  const k = +p.kernel,
    s = +p.stride;
  const size = (n) => (p.padding === "same" ? Math.ceil(n / s) : Math.floor((n - k) / s) + 1);
  const oh = size(h),
    ow = size(w);
  if (oh < 1 || ow < 1) {
    const fixes = p.padding === "valid" ? [{ label: "가장자리를 same으로", action: { kind: "setParam", key: "padding", value: "same" } }] : [];
    return { error: tooSmall(name, inShape, fixes) };
  }
  return { out: [oh, ow, channels] };
}

function poolOut(name, inShape, p) {
  if (!is3D(inShape)) return { error: needsImage(name, inShape) };
  const [h, w, c] = inShape;
  const oh = Math.floor(h / p.pool),
    ow = Math.floor(w / p.pool);
  if (oh < 1 || ow < 1) return { error: tooSmall(name, inShape) };
  return { out: [oh, ow, c] };
}

export const ACTIVATION_FNS = [
  { value: "relu", label: "ReLU", torch: "nn.ReLU()" },
  { value: "leakyrelu", label: "LeakyReLU", torch: "nn.LeakyReLU(0.2)" },
  { value: "elu", label: "ELU", torch: "nn.ELU()" },
  { value: "selu", label: "SELU", torch: "nn.SELU()" },
  { value: "swish", label: "Swish (SiLU)", torch: "nn.SiLU()" },
  { value: "sigmoid", label: "Sigmoid", torch: "nn.Sigmoid()" },
  { value: "tanh", label: "Tanh", torch: "nn.Tanh()" },
];

// Catalog -----------------------------------------------------------------

export const LAYERS = {
  input: {
    name: "Image Input",
    ko: "이미지 입력",
    source: true,
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
    about: "마이크로소프트 연구팀이 만든 유명한 이미지 모델이에요. 브라우저에서 돌리기엔 무거워서, 코드로 내보내 Colab이나 내 컴퓨터에서 학습해요.",
    summary: () => "코드 내보내기 전용",
    infer: (inShape, _, ctx) => {
      if (!is3D(inShape)) return { error: needsImage("ResNet-18", inShape) };
      const bad = pretrainedRules("ResNet-18", 224, inShape, ctx);
      return bad ? { error: bad } : { out: [7, 7, 512] };
    },
    params: () => 11176512,
    frozen: true,
  },

  resnet50: exportBackbone({
    name: "ResNet-50",
    about: "ResNet-18보다 훨씬 깊은 50층짜리 버전이에요. 사진을 7×7 칸마다 2,048가지 특징으로 바꿔요.",
    out: [7, 7, 2048],
    params: 23508032,
  }),

  efficientnet: exportBackbone({
    name: "EfficientNet-B0",
    about: "구글이 '크기에 비해 가장 정확하게' 설계한 모델이에요. MobileNet만큼 가볍지만 더 정확해요. 7×7×1,280 특징을 내요.",
    out: [7, 7, 1280],
    params: 4007548,
  }),

  vgg16: exportBackbone({
    name: "VGG16",
    about: "Conv2D와 MaxPool을 차곡차곡 쌓은 고전 모델이에요. 구조가 단순해서 공부하기 좋지만 무거워요. 7×7×512 특징을 내요.",
    out: [7, 7, 512],
    params: 14714688,
  }),

  vit: exportBackbone({
    name: "ViT-B/16",
    ko: "Hugging Face",
    about:
      "구글이 만든 Vision Transformer예요. 사진을 16×16 조각 196개로 잘라 문장처럼 읽고(Self-Attention), 사진 전체를 768칸 벡터 하나로 요약해요. Hugging Face에서 받아오고, PyTorch 코드로만 내보내요.",
    out: [768],
    params: 86389248,
  }),

  conv2d: {
    name: "Conv2D",
    ko: "특징 찾기",
    group: "image",
    defaults: { filters: 16, kernel: 3, stride: 1, padding: "same", activation: "relu" },
    fields: [
      { key: "filters", label: "필터 수", type: "number", min: 1, max: 256, hint: "찾을 특징의 종류 수. 출력 채널이 돼요." },
      ...KERNEL_FIELDS,
      { key: "activation", label: "활성화 함수", type: "select", options: ACTIVATIONS },
    ],
    about: "작은 창(커널)으로 사진을 훑으면서 선, 모서리, 무늬 같은 특징을 찾아요. 필터 수만큼 '특징 지도'가 생겨요.",
    summary: (p) => `필터 ${p.filters}개, ${p.kernel}×${p.kernel}${p.stride > 1 ? `, 보폭 ${p.stride}` : ""}`,
    infer: (inShape, p) => (is3D(inShape) ? convOut("Conv2D", inShape, p, +p.filters) : { error: needsImage("Conv2D", inShape) }),
    params: (inShape, p) => p.kernel * p.kernel * inShape[2] * p.filters + +p.filters,
  },

  depthwise: {
    name: "DepthwiseConv2D",
    ko: "채널별 특징",
    group: "image",
    defaults: { kernel: 3, stride: 1, padding: "same", activation: "relu" },
    fields: [...KERNEL_FIELDS, { key: "activation", label: "활성화 함수", type: "select", options: ACTIVATIONS }],
    about: "채널마다 따로 작은 창을 훑어요. 채널끼리 섞지 않아서 Conv2D보다 계산이 훨씬 적어요. 구글 MobileNet이 가벼운 비결이에요. 채널 수는 그대로예요.",
    summary: (p) => `${p.kernel}×${p.kernel}, 채널마다 따로${p.stride > 1 ? `, 보폭 ${p.stride}` : ""}`,
    infer: (inShape, p) => (is3D(inShape) ? convOut("DepthwiseConv2D", inShape, p, inShape[2]) : { error: needsImage("DepthwiseConv2D", inShape) }),
    params: (inShape, p) => p.kernel * p.kernel * inShape[2] + inShape[2],
  },

  maxpool: {
    name: "MaxPool2D",
    ko: "크기 줄이기",
    group: "image",
    defaults: { pool: 2 },
    fields: [POOL_FIELD],
    about: "2×2 칸마다 가장 큰 값만 남겨서 이미지를 작게 만들어요. 계산이 줄고, 위치가 조금 달라도 같은 걸로 알아봐요.",
    summary: (p) => `${p.pool}×${p.pool}마다 최댓값`,
    infer: (inShape, p) => poolOut("MaxPool2D", inShape, p),
    params: () => 0,
  },

  avgpool: {
    name: "AvgPool2D",
    ko: "평균으로 줄이기",
    group: "image",
    defaults: { pool: 2 },
    fields: [POOL_FIELD],
    about: "2×2 칸마다 평균을 내서 이미지를 작게 만들어요. MaxPool보다 부드럽게 줄여서, 넓게 퍼진 특징을 살려요.",
    summary: (p) => `${p.pool}×${p.pool}마다 평균`,
    infer: (inShape, p) => poolOut("AvgPool2D", inShape, p),
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

  gmp: {
    name: "GlobalMaxPool",
    ko: "최댓값으로 요약",
    group: "image",
    defaults: {},
    fields: [],
    about: "특징 지도마다 가장 큰 값 하나만 남겨요. 그 특징이 사진 어딘가에 '있는지 없는지'를 보는 방식이에요.",
    summary: () => "채널마다 최댓값 1개",
    infer: (inShape) => (is3D(inShape) ? { out: [inShape[2]] } : { error: needsImage("GlobalMaxPool", inShape) }),
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

  layernorm: {
    name: "LayerNorm",
    ko: "한 줄 안에서 고르게",
    group: "vector",
    defaults: {},
    fields: [],
    about: "벡터 하나 안의 값들을 평균 0, 분산 1로 맞춰요. Transformer(ViT, GPT)에 꼭 들어가는 층이에요. 여기서는 1차원 벡터 뒤에 써요.",
    summary: () => "벡터마다 정규화",
    infer: (inShape) => (is1D(inShape) ? { out: inShape } : { error: needsVector("LayerNorm", inShape) }),
    params: (inShape) => 2 * inShape[0],
  },

  activation: {
    name: "Activation",
    ko: "활성화",
    group: "vector",
    defaults: { fn: "relu" },
    fields: [
      { key: "fn", label: "함수", type: "select", options: ACTIVATION_FNS, hint: "Conv2D, Dense 안에도 들어 있지만 PyTorch처럼 따로 놓을 수도 있어요." },
    ],
    about: "숫자를 휘게(비선형) 만들어서, 모델이 직선 하나로는 못 나누는 것도 배우게 해요. 모양은 그대로예요.",
    summary: (p) => ACTIVATION_FNS.find((a) => a.value === p.fn)?.label ?? p.fn,
    infer: (inShape) => ({ out: inShape }),
    params: () => 0,
  },

  // ── Vision-language model (PaliGemma 2 style) ────────────────────
  text: {
    name: "Text Input",
    ko: "글 입력",
    group: "io",
    source: true,
    exportOnly: true,
    defaults: { prompt: "answer en What shape is this?" },
    fields: [
      {
        key: "prompt",
        label: "질문",
        type: "text",
        hint: "PaliGemma는 'answer en 질문', 'caption ko'(한국어로 설명해 줘) 같은 형식으로 물어봐요.",
      },
    ],
    about: "모델에게 할 질문이에요. 글자는 토크나이저가 단어 조각(토큰) 번호로 잘라요. VLM에는 입력이 두 개예요: 사진과 이 글.",
    summary: (p) => `“${String(p.prompt).slice(0, 22)}${String(p.prompt).length > 22 ? "…" : ""}”`,
    infer: (_, p) => ({ out: [promptTokens(p.prompt)] }),
    params: () => 0,
  },

  embedding: {
    name: "Embedding",
    ko: "Gemma 단어 사전",
    group: "pretrained",
    pretrained: true,
    exportOnly: true,
    frozen: true,
    defaults: {},
    fields: [],
    about: `단어 조각 번호 하나를 ${GEMMA.dim.toLocaleString("en-US")}칸짜리 벡터(토큰)로 바꿔요. Gemma가 미리 배운 ${GEMMA.vocab.toLocaleString("en-US")}개짜리 사전이에요.`,
    summary: () => `번호 → ${GEMMA.dim.toLocaleString("en-US")}칸`,
    infer: (inShape) => (is1D(inShape) ? { out: [inShape[0], GEMMA.dim] } : { error: needsTokenIds(inShape) }),
    params: () => GEMMA.vocab * GEMMA.dim,
  },

  siglip: {
    name: "SigLIP",
    ko: "비전 인코더",
    group: "pretrained",
    pretrained: true,
    exportOnly: true,
    frozen: true,
    defaults: {},
    fields: [],
    about:
      "구글이 사진과 설명글 짝으로 학습시킨 Vision Transformer예요. 224×224 사진을 14×14 조각 256개로 잘라서, 조각마다 1,152칸짜리 벡터(토큰)로 바꿔요. PaliGemma의 '눈'이에요.",
    summary: () => "사진 → 토큰 256개",
    infer: (inShape, _, ctx) => {
      if (!is3D(inShape)) return { error: needsImage("SigLIP", inShape) };
      const bad = pretrainedRules("SigLIP", 224, inShape, ctx);
      return bad ? { error: bad } : { out: [256, 1152] };
    },
    params: () => 427680704,
  },

  totokens: {
    name: "To Tokens",
    ko: "칸을 토큰으로",
    group: "vlm",
    exportOnly: true,
    defaults: {},
    fields: [],
    about: "세로×가로×채널 특징 지도를 칸마다 하나의 토큰으로 펼쳐요. 7×7×1,280이면 1,280칸짜리 토큰 49개가 돼요. CNN의 특징도 언어 모델에 넘길 수 있어요.",
    summary: () => "H×W×C → (H·W)×C",
    infer: (inShape) => (is3D(inShape) ? { out: [inShape[0] * inShape[1], inShape[2]] } : { error: needsImage("To Tokens", inShape) }),
    params: () => 0,
  },

  projector: {
    name: "Projector",
    ko: "이미지 → 글 공간",
    group: "vlm",
    exportOnly: true,
    defaults: { units: GEMMA.dim },
    fields: [
      {
        key: "units",
        label: "토큰 칸 수",
        type: "number",
        min: 1,
        max: 4096,
        hint: `언어 모델이 읽는 크기에 맞춰요. Gemma 2 2B는 ${GEMMA.dim.toLocaleString("en-US")}칸이에요.`,
      },
    ],
    about:
      "이미지 토큰을 언어 모델이 읽는 크기로 바꾸는 Linear 층이에요. 이게 있어야 이미지 토큰과 글 토큰을 이어 붙일 수 있어요. VLM을 만들 때 가장 먼저 학습하는 부분이에요.",
    summary: (p) => `토큰마다 → ${(+p.units).toLocaleString("en-US")}칸`,
    infer: (inShape, p) => (is2D(inShape) ? { out: [inShape[0], +p.units] } : { error: needsTokens("Projector", inShape) }),
    params: (inShape, p) => inShape[1] * p.units + +p.units,
  },

  concat: {
    name: "Concat",
    ko: "토큰 이어 붙이기",
    group: "vlm",
    merge: true,
    exportOnly: true,
    defaults: {},
    fields: [],
    about:
      "이미지 토큰 뒤에 글 토큰을 이어 붙여서 한 줄로 만들어요. 그래서 언어 모델이 사진과 질문을 한꺼번에 읽어요. 두 갈래가 합쳐지는 블록이에요(입력 2개).",
    summary: () => "이미지 토큰 + 글 토큰",
    infer: (inShape, _, ctx) => {
      if (!ctx.side) {
        return {
          error: {
            level: "error",
            title: "글 쪽이 비어 있어요",
            text: "Concat은 입력이 두 개예요. 이미지 토큰 말고 글(질문) 토큰도 이어 줘야 해요.",
            fixes: [{ label: "글 입력 + Embedding 붙이기", action: { kind: "addSide" } }],
          },
        };
      }
      if (!is2D(inShape)) return { error: needsTokens("Concat", inShape) };
      const side = ctx.side.shape;
      if (is1D(side)) {
        return {
          error: {
            level: "error",
            title: "글이 아직 번호예요",
            text: `글 쪽에서 단어 조각 번호 ${side[0]}개가 그대로 들어와요. 이어 붙이려면 번호마다 벡터(토큰)로 바꿔야 해요.`,
            fixes: [{ label: `Embedding 넣기 → ${fmtShape([side[0], GEMMA.dim])}`, action: { kind: "insertBefore", type: "embedding", side: true } }],
          },
        };
      }
      if (!is2D(side)) return { error: needsTokens("Concat(글 쪽)", side) };
      if (inShape[1] !== side[1]) {
        return {
          error: {
            level: "error",
            title: "토큰 길이가 달라서 못 붙여요",
            text: `이미지 토큰은 ${inShape[1].toLocaleString("en-US")}칸, 글 토큰은 ${side[1].toLocaleString("en-US")}칸이에요. 한 줄로 이으려면 토큰 하나의 길이가 같아야 해요. PyTorch였다면 torch.cat에서 "Sizes of tensors must match" 에러가 났을 거예요.`,
            fixes: [
              {
                label: `Projector 넣기 → ${fmtShape([inShape[0], side[1]])}`,
                note: "이미지 토큰을 글 토큰 길이로",
                action: { kind: "insertBefore", type: "projector", params: { units: side[1] } },
              },
            ],
          },
        };
      }
      return { out: [inShape[0] + side[0], inShape[1]] };
    },
    params: () => 0,
  },

  gemma: {
    name: "Gemma 2",
    ko: "언어 모델",
    group: "pretrained",
    pretrained: true,
    exportOnly: true,
    frozen: true,
    defaults: {},
    fields: [],
    about:
      "구글의 언어 모델 Gemma 2 (2B)예요. 이미지 토큰과 글 토큰을 한 줄로 읽고, 토큰마다 앞의 모든 토큰을 살펴봐요(어텐션). 26층짜리 Transformer라서 모양은 그대로예요.",
    summary: () => "26층 Transformer",
    infer: (inShape) => {
      if (!is2D(inShape)) return { error: needsTokens("Gemma 2", inShape) };
      if (inShape[1] !== GEMMA.dim) {
        return {
          error: {
            level: "error",
            title: "Gemma가 못 읽는 길이예요",
            text: `Gemma 2 2B는 ${GEMMA.dim.toLocaleString("en-US")}칸짜리 토큰만 읽어요. 지금 토큰은 ${inShape[1].toLocaleString("en-US")}칸이에요.`,
            fixes: [
              {
                label: `Projector 넣기 → ${fmtShape([inShape[0], GEMMA.dim])}`,
                action: { kind: "insertBefore", type: "projector", params: { units: GEMMA.dim } },
              },
            ],
          },
        };
      }
      return { out: inShape };
    },
    params: () => 2024517888,
  },

  textout: {
    name: "Text Output",
    ko: "글 출력",
    group: "io",
    terminal: true,
    exportOnly: true,
    defaults: {},
    fields: [],
    about: `마지막 토큰을 단어 조각 ${GEMMA.vocab.toLocaleString("en-US")}개의 점수로 바꿔서 가장 그럴듯한 다음 단어를 골라요. 이걸 반복해서 답을 한 조각씩 써요. Embedding 사전을 거꾸로 쓰기 때문에 따로 학습할 값이 없어요.`,
    summary: () => "다음 단어 고르기",
    infer: (inShape) => (is2D(inShape) ? { out: [GEMMA.vocab] } : { error: needsTokens("Text Output", inShape) }),
    params: () => 0,
  },

  output: {
    name: "Output",
    ko: "분류 출력",
    group: "io",
    terminal: true,
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
  { title: "비전-언어 모델 (VLM)", items: ["text", "embedding", "siglip", "totokens", "projector", "concat", "gemma", "textout"] },
  { title: "사전학습 모델", items: ["mobilenet", "efficientnet", "vit", "resnet18", "resnet50", "vgg16"] },
  { title: "이미지 층", items: ["conv2d", "depthwise", "maxpool", "avgpool", "gap", "gmp", "flatten"] },
  { title: "벡터 층", items: ["dense", "layernorm"] },
  { title: "활성화와 규제", items: ["activation", "dropout", "batchnorm"] },
  { title: "입력과 출력", items: ["input", "output"] },
];

// Code lines --------------------------------------------------------------

// PyTorch is channels-first; shapes in comments are written C×H×W to match.
export const torchShape = (s) => (s.length === 3 ? `${s[2]}×${s[0]}×${s[1]}` : s.join("×"));

export function torchLines(type, p, inShape, outShape) {
  const act = (a) => (torchAct[a] ? [`${torchAct[a]},`] : []);
  switch (type) {
    case "conv2d": {
      const pad = p.padding === "same" ? Math.floor(p.kernel / 2) : 0;
      return [`nn.Conv2d(${inShape[2]}, ${p.filters}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${pad}),`, ...act(p.activation)];
    }
    case "depthwise": {
      const pad = p.padding === "same" ? Math.floor(p.kernel / 2) : 0;
      const c = inShape[2];
      return [`nn.Conv2d(${c}, ${c}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${pad}, groups=${c}),`, ...act(p.activation)];
    }
    case "maxpool":
      return [`nn.MaxPool2d(${p.pool}),`];
    case "avgpool":
      return [`nn.AvgPool2d(${p.pool}),`];
    case "gap":
      return ["nn.AdaptiveAvgPool2d(1),", "nn.Flatten(),"];
    case "gmp":
      return ["nn.AdaptiveMaxPool2d(1),", "nn.Flatten(),"];
    case "layernorm":
      return [`nn.LayerNorm(${inShape[0]}),`];
    case "activation":
      return [`${ACTIVATION_FNS.find((a) => a.value === p.fn)?.torch ?? "nn.ReLU()"},`];
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
    case "text":
      return [`input_ids = processor(text=${JSON.stringify(p.prompt)}, ...)["input_ids"]`];
    case "embedding":
      return ["text_tokens = model.model.language_model.embed_tokens(input_ids)"];
    case "siglip":
      return ["image_tokens = model.model.vision_tower(pixel_values).last_hidden_state"];
    case "totokens":
      return ["x = x.flatten(2).transpose(1, 2)  # (N, C, H, W) → (N, H·W, C)"];
    case "projector":
      return [`nn.Linear(${inShape[1]}, ${p.units})  # multi_modal_projector`];
    case "concat":
      return ["x = torch.cat([image_tokens, text_tokens], dim=1)"];
    case "gemma":
      return ["x = model.model.language_model(inputs_embeds=x).last_hidden_state"];
    case "textout":
      return ["logits = model.lm_head(x[:, -1])  # 다음 단어 점수", "next_id = logits.argmax(-1)"];
    default:
      return [];
  }
}

export function kerasLines(type, p) {
  const act = (a) => (a === "linear" ? "" : `, activation="${a}"`);
  switch (type) {
    case "conv2d":
      return [`layers.Conv2D(${p.filters}, ${p.kernel}, strides=${p.stride}, padding="${p.padding}"${act(p.activation)}),`];
    case "depthwise":
      return [`layers.DepthwiseConv2D(${p.kernel}, strides=${p.stride}, padding="${p.padding}"${act(p.activation)}),`];
    case "maxpool":
      return [`layers.MaxPooling2D(${p.pool}),`];
    case "avgpool":
      return [`layers.AveragePooling2D(${p.pool}),`];
    case "gap":
      return ["layers.GlobalAveragePooling2D(),"];
    case "gmp":
      return ["layers.GlobalMaxPooling2D(),"];
    case "layernorm":
      return ["layers.LayerNormalization(),"];
    case "activation":
      return [p.fn === "leakyrelu" ? "layers.LeakyReLU(negative_slope=0.2)," : `layers.Activation("${p.fn}"),`];
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

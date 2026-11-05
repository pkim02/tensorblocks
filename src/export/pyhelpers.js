// Small Python classes the exported code needs for blocks that PyTorch / Keras don't ship
// as one layer. Only the ones the model uses are written out, in a fixed order.
// Parameter counts must match the canvas: tests/transformer.test.js checks them.

const TORCH = {
  LayerNorm2d: `class LayerNorm2d(nn.Module):
    # 이미지의 채널마다 정규화해요 (ConvNeXt와 같은 방식).
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)`,

  PatchEmbed: `class PatchEmbed(nn.Module):
    # 사진을 patch_size×patch_size 조각으로 자르고, 조각마다 dim칸 토큰 하나로 바꿔요.
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (N, D, H', W') → (N, H'·W', D)`,

  PositionEmbedding: `class PositionEmbedding(nn.Module):
    # 토큰마다 '몇 번째 자리인지'를 더해요. cls=True면 맨 앞에 CLS 토큰 1개를 붙여요.
    def __init__(self, tokens, dim, learned=True, cls=False):
        super().__init__()
        n = tokens + (1 if cls else 0)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls else None
        if learned:
            self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        else:  # 사인·코사인 물결 (학습하지 않아요)
            position = torch.arange(n).unsqueeze(1)
            freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
            pe = torch.zeros(1, n, dim)
            pe[0, :, 0::2] = torch.sin(position * freq)
            pe[0, :, 1::2] = torch.cos(position * freq)[:, : dim // 2]
            self.register_buffer("pos", pe, persistent=False)

    def forward(self, x):
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos`,

  SelfAttention: `class SelfAttention(nn.Module):
    # 토큰마다 다른 토큰을 살펴봐요. causal=True면 자기보다 앞쪽 토큰만 봐요.
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.causal = causal

    def forward(self, x):
        mask = None
        if self.causal:
            n = x.shape[1]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        return self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]`,

  TokenPool: `class TokenPool(nn.Module):
    # 토큰 줄을 벡터 하나로 요약해요. "mean"은 평균, "cls"는 맨 앞 CLS 토큰.
    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return x[:, 0] if self.mode == "cls" else x.mean(dim=1)`,

  RMSNorm: `if not hasattr(nn, "RMSNorm"):  # PyTorch 2.4부터 들어 있어요. 옛 버전이면 같은 것을 만들어 써요.
    class _RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    nn.RMSNorm = _RMSNorm`,

  MoEFFN: `class MoEFFN(nn.Module):
    # 문지기(gate)가 토큰마다 전문가 top_k개를 골라 그 답을 섞어요.
    # 배우기 쉽게 모든 전문가를 다 계산한 뒤 고른 것만 섞어요. 진짜 MoE는 고른 전문가만 계산해서 빨라요.
    def __init__(self, dim, hidden, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)) for _ in range(num_experts)
        )
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
        weights = torch.softmax(logits.masked_fill(logits < kth, float("-inf")), dim=-1)
        outs = torch.stack([e(x) for e in self.experts], dim=-1)  # (..., dim, 전문가 수)
        return (outs * weights.unsqueeze(-2)).sum(-1)`,
};

const KERAS = {
  PositionEmbedding: `class PositionEmbedding(layers.Layer):
    # 토큰마다 '몇 번째 자리인지'를 더해요. cls=True면 맨 앞에 CLS 토큰 1개를 붙여요.
    def __init__(self, learned=True, cls=False, **kwargs):
        super().__init__(**kwargs)
        self.learned, self.cls = learned, cls

    def build(self, input_shape):
        n, dim = int(input_shape[1]) + (1 if self.cls else 0), int(input_shape[2])
        if self.cls:
            self.cls_token = self.add_weight(name="cls_token", shape=(1, 1, dim), initializer="zeros")
        if self.learned:
            self.pos = self.add_weight(name="pos", shape=(1, n, dim), initializer=keras.initializers.RandomNormal(stddev=0.02))
        else:  # 사인·코사인 물결 (학습하지 않아요)
            position = np.arange(n)[:, None]
            freq = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
            pe = np.zeros((1, n, dim), "float32")
            pe[0, :, 0::2] = np.sin(position * freq)
            pe[0, :, 1::2] = np.cos(position * freq)[:, : dim // 2]
            self.pos = tf.constant(pe)

    def call(self, x):
        if self.cls:
            x = tf.concat([tf.tile(self.cls_token, [tf.shape(x)[0], 1, 1]), x], axis=1)
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "learned": self.learned, "cls": self.cls}`,

  SelfAttention: `class SelfAttention(layers.Layer):
    # 토큰마다 다른 토큰을 살펴봐요. causal=True면 자기보다 앞쪽 토큰만 봐요.
    def __init__(self, dim, num_heads, causal=False, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads, self.causal = dim, num_heads, causal
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)

    def call(self, x):
        return self.attn(x, x, use_causal_mask=self.causal)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "num_heads": self.num_heads, "causal": self.causal}`,

  TokenPool: `class TokenPool(layers.Layer):
    # 토큰 줄을 벡터 하나로 요약해요. "mean"은 평균, "cls"는 맨 앞 CLS 토큰.
    def __init__(self, mode="mean", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, x):
        return x[:, 0] if self.mode == "cls" else tf.reduce_mean(x, axis=1)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}`,

  RMSNorm: `class RMSNorm(layers.Layer):
    # 평균은 빼지 않고, 값들의 제곱평균만 1로 맞춰요.
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(name="weight", shape=(int(input_shape[-1]),), initializer="ones")

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps) * self.weight

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}`,

  MoEFFN: `class MoEFFN(layers.Layer):
    # 문지기(gate)가 토큰마다 전문가 top_k개를 골라 그 답을 섞어요.
    # 배우기 쉽게 모든 전문가를 다 계산한 뒤 고른 것만 섞어요. 진짜 MoE는 고른 전문가만 계산해서 빨라요.
    def __init__(self, dim, hidden, num_experts, top_k, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.hidden, self.num_experts, self.top_k = dim, hidden, num_experts, top_k
        self.gate = layers.Dense(num_experts)
        self.experts = [keras.Sequential([layers.Dense(hidden, activation="gelu"), layers.Dense(dim)]) for _ in range(num_experts)]

    def call(self, x):
        logits = self.gate(x)
        kth = tf.math.top_k(logits, k=self.top_k).values[..., -1:]
        weights = tf.nn.softmax(tf.where(logits < kth, -1e9, logits))
        outs = tf.stack([e(x) for e in self.experts], axis=-1)  # (..., dim, 전문가 수)
        return tf.reduce_sum(outs * weights[..., None, :], axis=-1)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "hidden": self.hidden, "num_experts": self.num_experts, "top_k": self.top_k}`,
};

// Which helper classes a block needs, given its params and input shape.
function torchNeeds(type, p, inShape) {
  switch (type) {
    case "layernorm":
      return inShape?.length === 3 ? ["LayerNorm2d"] : [];
    case "patchembed":
      return ["PatchEmbed"];
    case "posembed":
      return ["PositionEmbedding"];
    case "attention":
      return ["SelfAttention"];
    case "seqpool":
      return ["TokenPool"];
    case "rmsnorm":
      return ["RMSNorm"];
    case "moe":
      return ["MoEFFN"];
    default:
      return [];
  }
}

function kerasNeeds(type, p) {
  switch (type) {
    case "posembed":
      return ["PositionEmbedding"];
    case "attention":
      return ["SelfAttention"];
    case "seqpool":
      return p.mode === "cls" ? ["TokenPool"] : [];
    case "rmsnorm":
      return ["RMSNorm"];
    case "moe":
      return ["MoEFFN"];
    default:
      return [];
  }
}

const collect = (blocks, needs, table) => {
  const used = new Set(blocks.flatMap(({ type, params, inShape }) => needs(type, params, inShape)));
  return Object.keys(table).filter((k) => used.has(k));
};

// blocks: [{ type, params, inShape }]. Returns the class source to put above the model, and extra imports.
export function torchHelpers(blocks) {
  const names = collect(blocks, torchNeeds, TORCH);
  return { names, code: names.map((k) => `${TORCH[k]}\n\n\n`).join(""), imports: names.includes("PositionEmbedding") ? ["import math"] : [] };
}

export function kerasHelpers(blocks) {
  const names = collect(blocks, kerasNeeds, KERAS);
  return { names, code: names.map((k) => `${KERAS[k]}\n\n\n`).join(""), imports: names.length ? ["import numpy as np", "import tensorflow as tf"] : [] };
}

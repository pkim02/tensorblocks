# TensorBlocks

Teachable Machine처럼 쉽게 데이터를 모으고 학습시키되, 모델 속을 **블록으로 열어서 고치고**, **3D로 들여다보고**, 완성된 구조를 **Google Colab 노트북**(또는 PyTorch / Keras 코드)으로 가져가는 웹 앱이에요. (해커톤 주제: 안전, 교육)

**바로 써 보기: https://tensorblocks.netlify.app** (설치 없이 브라우저에서 열려요. 웹캠은 https라서 바로 돼요.)

## 실행

Node 20 이상이 필요해요.

```bash
npm install
npm run dev
```

브라우저에서 http://localhost:5181 을 열어요. 웹캠이 없으면 **예제 도형 불러오기**로 동그라미·세모·네모 120장을 바로 불러올 수 있어요.

## 화면 흐름

1. **데이터**: 클래스마다 웹캠(누르고 있으면 연속 촬영)이나 사진 파일로 이미지를 모아요.
2. **모델과 학습**: 모델 틀을 고르고 학습시켜요. **모델 열어 보기**에서 같은 모델을 세 가지로 열 수 있어요: **블록** / **3D** / **Colab 코드** (위쪽 탭으로 오가요).
3. **미리보기**: 웹캠·사진으로 바로 시험하고, **모델 속 보기**에서 블록마다 사진을 어떻게 보는지(특징 지도) 볼 수 있어요.

## 블록 캔버스에서 되는 것

- **VLM 예시 (구글 PaliGemma 2 방식)**: 모델 틀 맨 위의 "VLM: 사진 보고 답하기". 사진(Image Input → SigLIP → Projector)과 질문(Text Input → Embedding) 두 갈래가 **Concat**에서 합쳐져 Gemma 2 → Text Output으로 가요. 블록 수·모양·값 수는 실제 PaliGemma 2 3B와 같아요 (3.05B). "실수 체험: VLM에서 Projector 빠뜨리기"로 토큰 길이가 안 맞는 에러(1,152 vs 2,304)와 자동 수정을 보여 줄 수 있어요.
- 블록 37종: VLM 블록 8개(Text Input, Embedding, SigLIP, To Tokens, Projector, Concat, Gemma 2, Text Output), 사전학습 모델 6개(MobileNetV2, EfficientNet-B0, ViT-B/16 (Hugging Face), ResNet-18/50, VGG16), 이미지 층 10개(Conv2D, DepthwiseConv2D, MaxPool, AvgPool, GlobalAvgPool, GlobalMaxPool, Flatten, Patch Embedding, Upsample2D, ConvTranspose2D), 벡터 층(Dense, LayerNorm, RMSNorm), 트랜스포머 층 4개(Position Embedding, Self-Attention, MoE FFN, 시퀀스 요약), 활성화(ReLU·LeakyReLU·ELU·SELU·Swish·GELU·Sigmoid·Tanh), Dropout, BatchNorm, Add, 입력·출력.
- MobileNetV2와 직접 쌓은 층(Add로 만든 skip connection과 트랜스포머 층 포함)은 브라우저에서 바로 학습돼요. 나머지 사전학습 모델은 무거워서 "코드 전용"이에요 (Colab에서 학습). Upsample2D와 ConvTranspose2D도 아직 분류 출력뿐이라 코드 전용이에요.
- **모양은 세 가지**예요: 이미지 `세로×가로×채널`, 벡터 `N`, 토큰 줄 `토큰 수×칸 수`(마우스를 올리면 "토큰 64개 × 64칸"). Dense·LayerNorm·Dropout·Activation은 토큰 줄에서 토큰마다 마지막 칸들에만 적용돼요.
- **트랜스포머 모델 틀**: "나만의 작은 ViT"(Patch Embedding → Position Embedding → [LayerNorm → Self-Attention → 더하기] → [LayerNorm → Dense 두 개 → 더하기] ×2 → 시퀀스 요약), "ConvNeXt 블록", "실수 체험: 헤드 수가 칸 수로 안 나눠짐". 시퀀스 요약에서 CLS 토큰을 고르면 Position Embedding이 맨 앞에 토큰 1개를 붙여요.
- 블록을 끌어다 놓거나 눌러서 넣기. 선 위에 놓으면 그 사이에 끼워져요.
- 모든 연결선에 텐서 모양(예: `32×32×16`)이 표시돼요.
- 모양이 안 맞으면 선이 빨개지고, 무엇이 틀렸는지와 고치는 버튼이 나와요 (예: Conv 뒤에 바로 Dense → GlobalAvgPool 또는 Flatten 넣기).
- 너무 무거운 구조(예: 62,720칸으로 펴진 Flatten)는 주황색으로 경고해요.
- 블록을 고르면 하는 일 설명, 설정, 그 블록의 PyTorch 코드 한 줄이 보여요.
- 시연용 모델 틀 **실수 체험: Flatten 빠뜨리기**로 차원 에러와 자동 수정을 바로 보여줄 수 있어요.

## 3D로 보기

모델 카드의 **3D** 버튼이나 위쪽 **3D** 탭을 누르면, 블록마다 나오는 텐서를 3D 상자로 보여 줘요 (three.js).

- 이미지 모양 데이터는 판이 겹친 상자(가로·세로 = 이미지 크기, 깊이 = 채널 수), 벡터는 세로 막대예요.
- 학습한 뒤에는 실제 사진 한 장을 넣었을 때 각 블록이 만든 특징 지도가 판에 그려져요. **다른 사진**으로 바꿔 볼 수 있어요.
- 왼쪽 **둘러보기**에서 계속/이전(또는 ← → 키)을 누르면 카메라가 블록마다 날아가며 설명해 줘요. 블록을 눌러도 돼요.
- 크기는 실제 숫자에 비례하지 않고 완만하게 줄였어요 (d^0.42).

## Colab 코드

**Colab 코드** 탭은 모델을 Google Colab 노트북(`.ipynb`)으로 바꿔 줘요. 화면에서 노트북을 칸별로 미리 보고, PyTorch·Keras `.py`로도 볼 수 있어요.

- **노트북 받고 Colab 열기** → Colab의 노트북 열기 창에서 **업로드** → 받은 파일 고르기 → **런타임 → 모두 실행**.
- 모은 사진이 노트북 안에 들어 있어서(접힌 칸) 따로 올릴 필요가 없어요. GPU(T4)로 열리게 설정돼 있어요.
- 노트북 순서: 준비 → 사진 풀기 → 데이터 → 모델 → 블록별 모양 출력(캔버스와 같은지 확인) → 학습 → 정확도 그래프 → 내 사진으로 시험 → 저장.
- VLM 노트북은 구글의 실제 PaliGemma 2 (3B)를 불러와서 블록별 모양 확인 → 내 사진에 질문 → 학습 없이 클래스 이름 맞히기를 해요. 처음 한 번 Hugging Face에서 약관 동의와 토큰(`HF_TOKEN`)이 필요해요.
- 버튼 한 번에 Colab에 바로 열리게 하려면 Google Drive 연동(구글 클라우드 OAuth 설정)이 필요해서 아직은 파일 업로드 방식이에요.

## 코드 구조

| 파일 | 하는 일 |
|---|---|
| `src/model/layers.js` | 블록 목록: 모양 계산, 에러와 고치는 방법, PyTorch/Keras 코드 한 줄 |
| `src/model/graph.js` | 캔버스 그래프를 따라가며 모든 연결의 모양과 문제를 계산 (VLM처럼 두 갈래가 합쳐지는 구조 포함), 모델 틀 |
| `src/store.js` | 앱 상태 (zustand): 클래스·사진, 블록·연결, 고치기 동작, 학습 상태 |
| `src/train/train.js` | TF.js 학습. 캔버스 그래프를 함수형 모델로 만들어요(갈래·Add 포함). MobileNetV2(Teachable Machine과 같은 가중치)는 고정하고 뒤쪽 블록만 학습 |
| `src/train/blocks.js` | TF.js에 없는 층: Position Embedding, Self-Attention, 시퀀스 요약, RMSNorm, MoE FFN |
| `src/export/pyhelpers.js` | 내보낸 PyTorch·Keras 코드에 들어가는 보조 클래스 (PatchEmbed, SelfAttention 등) |
| `src/three/scene.js` | 3D 장면: 텐서 상자, 특징 지도 판, 카메라 이동, 블록 클릭 |
| `src/export/code.js` | Colab 노트북(.ipynb), PyTorch / Keras 학습 스크립트 생성, zip 내보내기 |
| `src/components/` | 화면: 데이터, 모델·학습, 미리보기, 블록 캔버스, 3D, Colab 코드 |
| `DESIGN.md` | 디자인 방향 (색, 글꼴, 레이아웃, 해도 되는 것과 안 되는 것) |
| `public/kit/` | 공용 UI 키트 (직접 고치지 말고 `src/styles.css`에서 덮어쓰기) |

## 같이 작업할 때

- `main`에 바로 올리지 말고 브랜치를 만들어서 PR로 합쳐요.
- 올리기 전에 `npx prettier@3.9.10 --write src`로 코드 모양을 맞춰요 (`.prettierrc`에 설정 있음). 버전(`@3.9.10`)을 붙여야 모두 같은 모양이 나와요. `src/styles.css`는 일부러 빼 두었어요 (`.prettierignore`).
  - 실행 후 `git status`로 확인해서, 내가 작업하지 않은 파일이 바뀌었으면 `git restore 파일경로`로 되돌리고 올려 주세요.
- 화면은 라이트 모드 전용이에요 (`index.html`의 `data-mode="light"`).
- 배포: `main`에 push(또는 PR 합치기)하면 1~2분 뒤 https://tensorblocks.netlify.app 에 자동으로 올라가요. 그러니 `main`에는 잘 돌아가는 것만 합쳐 주세요. 급할 때는 `npm run deploy`로 직접 올릴 수도 있어요 (Netlify HAPE 팀 로그인 필요).

### 블록을 바꿨을 때 더 확인할 것

- `npm test`: 모양·값 수가 PyTorch 공식과 같은지, TF.js 모델의 값 수가 캔버스와 같은지 봐요 (`tests/transformer.test.js`).
- `node scripts/check-python-export.mjs`: 내보낸 PyTorch·Keras 코드를 사진 몇 장으로 1 에폭 실제로 돌리고, 프레임워크가 센 학습할 값 수를 캔버스와 비교해요. 노트북 코드 칸의 문법도 확인해요. torch·torchvision이 필요하고, Keras까지 보려면 tensorflow도 필요해요 (`--no-keras`로 건너뛰기).
- `node scripts/check-browser-training.mjs "vit|60|0.0003" "cnn|20"`: 크롬(헤드리스)에서 예제 도형 데이터로 모델 틀을 학습해서 loss가 줄어드는지 봐요. 인자는 `모델 틀|에폭|학습률`이에요.

### 머지하기 전 확인할 것

PR은 다른 팀원 한 명이 보고 **Approve**하면, 올린 사람이 직접 머지해요.

1. 최신 `main`을 내 브랜치에 받아요 (`git pull origin main`). 충돌이 나면 내 브랜치에서 풀고 다시 올려요.
2. `npm run build`가 에러 없이 끝나요.
3. `npm run dev`로 열어서 내가 바꾼 부분이 되고, 다른 화면(데이터·학습·미리보기, 블록·3D·Colab 코드 탭)도 하얀 화면 없이 열려요. 브라우저 콘솔에 빨간 에러가 없어요.
4. PR의 **Files changed**에 내가 작업한 파일만 있어요 (prettier가 바꾼 남의 파일이나 `.env`가 섞이지 않게).
5. `package.json`을 바꿨으면 `package-lock.json`도 같이 올려요.
6. 머지하고 1~2분 뒤 https://tensorblocks.netlify.app 를 열어 확인해요. 깨졌으면 PR 페이지 아래의 **Revert** 버튼으로 바로 되돌리고 팀 채팅에 알려 주세요.

## 기술

React 19 + Vite, [React Flow](https://reactflow.dev) (블록 캔버스), TensorFlow.js (브라우저 학습), JSZip.
학습은 전부 브라우저 안에서 일어나고, 사진은 서버로 보내지 않아요. MobileNet 가중치만 Google 저장소에서 한 번 내려받아요.

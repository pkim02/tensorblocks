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

- 블록 20종: 사전학습 모델 6개(MobileNetV2, EfficientNet-B0, ViT-B/16 (Hugging Face), ResNet-18/50, VGG16), 이미지 층 7개(Conv2D, DepthwiseConv2D, MaxPool, AvgPool, GlobalAvgPool, GlobalMaxPool, Flatten), 벡터 층(Dense, LayerNorm), 활성화(ReLU·LeakyReLU·ELU·SELU·Swish·Sigmoid·Tanh), Dropout, BatchNorm, 입력·출력.
- MobileNetV2와 직접 쌓은 층은 브라우저에서 바로 학습돼요. 나머지 사전학습 모델은 무거워서 "코드 전용"이에요 (Colab에서 학습).
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
- 버튼 한 번에 Colab에 바로 열리게 하려면 Google Drive 연동(구글 클라우드 OAuth 설정)이 필요해서 아직은 파일 업로드 방식이에요.

## 코드 구조

| 파일 | 하는 일 |
|---|---|
| `src/model/layers.js` | 블록 목록: 모양 계산, 에러와 고치는 방법, PyTorch/Keras 코드 한 줄 |
| `src/model/graph.js` | 캔버스 그래프를 따라가며 모든 연결의 모양과 문제를 계산, 모델 틀 |
| `src/store.js` | 앱 상태 (zustand): 클래스·사진, 블록·연결, 고치기 동작, 학습 상태 |
| `src/train/train.js` | TF.js 학습. MobileNetV2(Teachable Machine과 같은 가중치)는 고정하고 뒤쪽 블록만 학습 |
| `src/three/scene.js` | 3D 장면: 텐서 상자, 특징 지도 판, 카메라 이동, 블록 클릭 |
| `src/export/code.js` | Colab 노트북(.ipynb), PyTorch / Keras 학습 스크립트 생성, zip 내보내기 |
| `src/components/` | 화면: 데이터, 모델·학습, 미리보기, 블록 캔버스, 3D, Colab 코드 |
| `DESIGN.md` | 디자인 방향 (색, 글꼴, 레이아웃, 해도 되는 것과 안 되는 것) |
| `public/kit/` | 공용 UI 키트 (직접 고치지 말고 `src/styles.css`에서 덮어쓰기) |

## 같이 작업할 때

- `main`에 바로 올리지 말고 브랜치를 만들어서 PR로 합쳐요.
- 올리기 전에 `npx prettier --write src`로 코드 모양을 맞춰요 (`.prettierrc`에 설정 있음).
- 화면은 라이트 모드 전용이에요 (`index.html`의 `data-mode="light"`).
- 배포: `main`에 push(또는 PR 합치기)하면 1~2분 뒤 https://tensorblocks.netlify.app 에 자동으로 올라가요. 그러니 `main`에는 잘 돌아가는 것만 합쳐 주세요. 급할 때는 `npm run deploy`로 직접 올릴 수도 있어요 (Netlify HAPE 팀 로그인 필요).

## 기술

React 19 + Vite, [React Flow](https://reactflow.dev) (블록 캔버스), TensorFlow.js (브라우저 학습), JSZip.
학습은 전부 브라우저 안에서 일어나고, 사진은 서버로 보내지 않아요. MobileNet 가중치만 Google 저장소에서 한 번 내려받아요.

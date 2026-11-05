// Words a beginner may not know yet. The text keeps the real term; pointing at it shows
// a plain one-line meaning (see Glossed in components/Term.jsx).
export const GLOSSARY = {
  채널: "사진의 한 점에 담긴 숫자의 개수예요. 컬러 사진은 빨강·초록·파랑 3개이고, 블록을 지나면 찾은 특징의 수만큼 늘어나요.",
  "특징 지도": "특징 하나가 사진의 어느 위치에서 얼마나 강하게 보이는지 적어 둔 표예요.",
  가중치: "모델이 학습하면서 조금씩 고치는 숫자예요. 모델이 배운 내용이 여기에 저장돼요.",
  뉴런: "들어온 숫자마다 가중치를 곱해서 더한 뒤, 숫자 하나를 내는 작은 계산 단위예요.",
  과적합: "학습한 사진만 외워서, 처음 보는 사진은 잘 못 맞히는 상태예요.",
  커널: "사진을 훑을 때 한 번에 보는 작은 창이에요. 3이면 3×3 크기예요.",
  필터: "특징 한 가지를 찾는 창이에요. 필터가 16개면 특징 16가지를 찾아요.",
  stride: "창이 한 번에 몇 칸씩 옮겨 가는지예요.",
  비선형: "곱하고 더하기만으로는 만들 수 없는 꺾인 계산이에요. 이게 있어야 복잡한 구분을 배울 수 있어요.",
  Softmax: "점수들을 모두 더하면 100%가 되는 확률로 바꾸는 계산이에요.",
  토큰: "언어 모델이 읽는 한 조각이에요. 단어 조각이나 사진 조각 하나가 토큰 하나예요.",
  정규화: "숫자들의 크기를 비슷한 범위로 맞추는 것이에요.",
  분산: "숫자들이 평균에서 얼마나 흩어져 있는지 나타내는 값이에요.",
  ImageNet: "사진 약 120만 장, 1,000가지 종류로 된 유명한 학습용 사진 모음이에요.",
  사전학습: "큰 데이터로 미리 학습시켜 둔 것이에요. 가져와서 바로 쓸 수 있어요.",
  어텐션: "토큰마다 앞의 어떤 토큰이 중요한지 골라서 보는 계산이에요.",
  Transformer: "어텐션을 여러 층 쌓은 모델 구조예요. ChatGPT와 Gemma가 이 구조예요.",
  "skip connection": "앞 블록의 출력을 몇 블록 건너뛰어 뒤에 바로 더하는 연결이에요.",
  클래스: "모델이 구분할 종류예요. '데이터' 탭에서 만든 목록이에요.",
  에폭: "가진 사진 전체를 한 번 다 보고 학습하는 것이 1에폭이에요.",
  배치: "가중치를 한 번 고칠 때 함께 보는 사진 묶음이에요.",
  학습률: "가중치를 한 번에 얼마나 크게 고칠지예요. 너무 크면 학습이 들쭉날쭉해요.",
  손실: "모델의 답이 정답과 얼마나 다른지 나타내는 값이에요. 낮을수록 좋아요.",
};

// Picks the right Korean particle after a word: josa("Dense", "은는") → "Dense는".
// English names go by how they are read aloud (Flatten → 은, Conv2D → 는).
export function josa(word, pair) {
  const [withFinal, without] = [...pair];
  return word + (hasFinal(word) ? withFinal : without);
}

function hasFinal(word) {
  const w = String(word).replace(/\s*\([^)]*\)\s*$/, "").trim();
  const c = w[w.length - 1] ?? "";
  const code = c.charCodeAt(0);
  if (code >= 0xac00 && code <= 0xd7a3) return (code - 0xac00) % 28 !== 0;
  if (/\d/.test(c)) return "013678".includes(c); // 영 일 삼 육 칠 팔
  // A lone capital after a digit or a lowercase letter is read by its letter name (Conv2D → 디);
  // a run of capitals is read as a word (SigLIP → 립).
  if (/[A-Z]/.test(c) && !/[A-Z]/.test(w[w.length - 2] ?? "")) return "LMN".includes(c); // 엘 엠 엔
  const low = w.toLowerCase();
  if (low.endsWith("ng")) return true;
  return /[lmnpkt]$/.test(low);
}

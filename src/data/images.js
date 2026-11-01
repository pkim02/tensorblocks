import { newId } from "../model/graph.js";

export const SAMPLE_SIZE = 224;
const THUMB = 72;

// Centre-crops any image/video/canvas into a square SAMPLE_SIZE canvas.
export function toSquareCanvas(source, srcW, srcH, { mirror = false } = {}) {
  const c = document.createElement("canvas");
  c.width = c.height = SAMPLE_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const side = Math.min(srcW, srcH);
  const sx = (srcW - side) / 2,
    sy = (srcH - side) / 2;
  if (mirror) {
    ctx.translate(SAMPLE_SIZE, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source, sx, sy, side, side, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  return c;
}

export function makeSample(canvas) {
  const t = document.createElement("canvas");
  t.width = t.height = THUMB;
  t.getContext("2d").drawImage(canvas, 0, 0, THUMB, THUMB);
  return { id: newId("s"), canvas, thumb: t.toDataURL("image/jpeg", 0.8) };
}

export function grabFrame(video) {
  if (!video || video.readyState < 2) return null;
  return toSquareCanvas(video, video.videoWidth, video.videoHeight, { mirror: true });
}

export function fileToCanvas(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      resolve(toSquareCanvas(img, img.naturalWidth, img.naturalHeight));
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`${file.name}을(를) 열 수 없어요`));
    };
    img.src = url;
  });
}

export async function filesToSamples(files) {
  const out = [];
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    try {
      out.push(makeSample(await fileToCanvas(f)));
    } catch {
      /* skip unreadable files */
    }
  }
  return out;
}

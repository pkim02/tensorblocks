// Reads a Hugging Face dataset's structure and first rows for the VLM data panel.
// The dataset itself is never downloaded here; photos load straight from Hugging Face.
const API = "https://datasets-server.huggingface.co";
const PREVIEW_ROWS = 20;

const QUESTION = ["question", "query", "prompt", "user"];
const ANSWER = ["answer", "answers", "response", "assistant"];

// Accepts "owner/name" or a dataset page URL.
export function repoName(text) {
  return text
    .trim()
    .replace(/^https?:\/\/huggingface\.co\/datasets\//, "")
    .replace(/\/+$/, "");
}

async function get(path, params, signal) {
  const res = await fetch(`${API}/${path}?${new URLSearchParams(params)}`, { signal });
  if (res.status === 401 || res.status === 404)
    throw new Error("데이터셋을 찾지 못했어요. 이름을 확인해 주세요. 비공개이거나 동의가 필요한 데이터셋은 읽을 수 없어요.");
  if (!res.ok) throw new Error(`Hugging Face에서 데이터를 읽지 못했어요 (${res.status}). 잠시 후 다시 시도해 주세요.`);
  return res.json();
}

// Subsets of a dataset, each with its splits, e.g. [{ config: "cocoqa", splits: ["train"] }].
export async function fetchConfigs(repo, signal) {
  const { splits } = await get("splits", { dataset: repo }, signal);
  const byConfig = new Map();
  for (const s of splits) (byConfig.get(s.config) || byConfig.set(s.config, []).get(s.config)).push(s.split);
  return [...byConfig].map(([config, list]) => ({ config, splits: list }));
}

const isImage = (t) => t?._type === "Image" || t?.feature?._type === "Image" || (Array.isArray(t) && t[0]?._type === "Image");
const pick = (names, wanted) => wanted.find((w) => names.includes(w));

// Which columns hold the photo, question and answer: either flat columns or a
// conversation list like the_cauldron's texts[{ user, assistant }].
export function findFields(features) {
  const image = features.find((f) => isImage(f.type))?.name;
  const names = features.map((f) => f.name);
  const flat = { question: pick(names, QUESTION), answer: pick(names, ANSWER) };
  if (image && flat.question && flat.answer) return { image, ...flat, list: null };
  for (const f of features) {
    const item = Array.isArray(f.type) ? f.type[0] : f.type?.feature;
    const keys = Object.keys(item || {});
    const nested = { question: pick(keys, QUESTION), answer: pick(keys, ANSWER) };
    if (image && nested.question && nested.answer) return { image, ...nested, list: f.name };
  }
  return null;
}

const firstText = (v) => String(Array.isArray(v) ? (v[0] ?? "") : (v ?? ""));

// One example per question: a photo in a conversation list can have several.
export function rowsToExamples(rows, fields) {
  return rows.flatMap(({ row }) => {
    const photo = row[fields.image];
    const image = (Array.isArray(photo) ? photo[0] : photo)?.src ?? null;
    const pairs = fields.list ? row[fields.list] || [] : [row];
    return pairs.map((p) => ({ image, question: firstText(p[fields.question]).trim(), answer: firstText(p[fields.answer]).trim(), source: p.source ?? "" }));
  });
}

export async function fetchPreview(repo, config, split, signal) {
  // /rows, not /first-rows: the photo links /first-rows gives for the_cauldron answer 403.
  const { features, rows } = await get("rows", { dataset: repo, config, split, offset: 0, length: PREVIEW_ROWS }, signal);
  const fields = findFields(features);
  return { fields, examples: fields ? rowsToExamples(rows, fields) : [] };
}

export const fieldPath = (fields, key) => (fields.list ? `${fields.list}.${fields[key]}` : fields[key]);

// Trains templates in a real (headless) Chrome on the shapes dataset and checks the loss goes down.
//   node scripts/check-browser-training.mjs "vit|60|0.0003" "convnext|20"
// Each argument is template|epochs|learning rate (or a template JSON instead of a name). CHROME=path overrides the browser.
import { createServer } from "vite";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const CHROME =
  process.env.CHROME ||
  [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ].find(existsSync);
if (!CHROME) throw new Error("Chrome을 찾지 못했어요. CHROME=경로 로 알려 주세요.");

const runs = (process.argv.slice(2).length ? process.argv.slice(2) : ["vit|60|0.0003"]).map((a) => {
  const [t, epochs = "20", lr = "0.001"] = a.split("|");
  return { t, epochs, lr };
});

const server = await createServer({ server: { host: "127.0.0.1", port: 5191, strictPort: true }, logLevel: "error" });
await server.listen();
const profile = mkdtempSync(path.join(tmpdir(), "tb-chrome-"));
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${profile}`, "--no-first-run", "--enable-unsafe-swiftshader", "about:blank"], {
  stdio: "ignore",
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  target = await fetch("http://127.0.0.1:9333/json")
    .then((r) => r.json())
    .then((list) => list.find((t) => t.type === "page"))
    .catch(() => null);
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let msgId = 0;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (pending.has(m.id)) pending.get(m.id)(m.result);
});
const send = (method, params = {}) =>
  new Promise((resolve) => {
    pending.set(++msgId, resolve);
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result.value;

let failed = false;
try {
  for (const { t, epochs, lr } of runs) {
    await send("Page.navigate", { url: `http://127.0.0.1:5191/tests/train-browser.html?${t.startsWith("{") ? `spec=${encodeURIComponent(t)}` : `t=${t}`}&epochs=${epochs}&lr=${lr}` });
    let result = null;
    for (let i = 0; i < 1800 && !result; i++) {
      await sleep(1000);
      result = await evaluate("window.__result ?? null").catch(() => null);
    }
    const losses = result?.logs?.map((l) => l.loss) ?? [];
    const accs = result?.logs?.map((l) => l.acc) ?? [];
    const ok = result?.status === "done" && losses.length > 1 && losses.at(-1) < losses[0];
    failed ||= !ok;
    console.log(`${ok ? "✔" : "✖"} ${t.slice(0, 60)} (lr ${lr}): ${result?.status} on ${result?.backend}, ${result?.seconds}s`);
    console.log(`  loss ${losses.map((v) => v.toFixed(3)).join(" ")}`);
    console.log(`  train acc ${accs.map((v) => v.toFixed(2)).join(" ")}`);
    console.log(`  val acc ${(result?.logs ?? []).map((l) => (l.valAcc ?? 0).toFixed(2)).join(" ")}`);
    if (result?.inside) console.log(`  모델 속 보기: ${result.inside.join(", ")}`);
    if (!ok) console.log(`  ${result?.message}`);
  }
} finally {
  ws.close();
  chrome.kill();
  await server.close();
  await sleep(500);
  rmSync(profile, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);

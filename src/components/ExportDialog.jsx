import { useEffect, useMemo, useRef, useState } from "react";
import { useApp, useAnalysis } from "../store.js";
import { pytorchCode, kerasCode, downloadZip } from "../export/code.js";

export default function ExportDialog() {
  const open = useApp((s) => s.exportOpen);
  const setOpen = useApp((s) => s.setExportOpen);
  const nodes = useApp((s) => s.nodes);
  const classes = useApp((s) => s.classes);
  const settings = useApp((s) => s.settings);
  const a = useAnalysis();
  const ref = useRef(null);
  const [lang, setLang] = useState("torch");
  const [zipping, setZipping] = useState(false);

  useEffect(() => {
    const d = ref.current;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const code = useMemo(() => {
    if (!open || !a.ok) return "";
    return lang === "torch" ? pytorchCode(a, nodes, classes, settings) : kerasCode(a, nodes, classes, settings);
  }, [open, lang, a, nodes, classes, settings]);

  const photos = classes.reduce((n, c) => n + c.samples.length, 0);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      window.kit?.toast("코드를 복사했어요");
    } catch {
      window.kit?.toast("복사하지 못했어요. 코드를 직접 선택해 주세요.", { tone: "danger" });
    }
  };

  const zip = async () => {
    setZipping(true);
    try {
      await downloadZip(a, nodes, classes, settings);
    } finally {
      setZipping(false);
    }
  };

  return (
    <dialog ref={ref} className="dialog export" closedby="any" onClose={() => setOpen(false)} aria-labelledby="export-h">
      <h2 id="export-h">코드 내보내기</h2>
      <p className="small muted">캔버스에서 계산한 모양 그대로 in_channels, in_features를 채워 넣었어요. 내 컴퓨터나 Colab에서 바로 돌아가요.</p>
      <div className="segmented" role="radiogroup" aria-label="프레임워크">
        <label>
          <input type="radio" name="lang" checked={lang === "torch"} onChange={() => setLang("torch")} /> PyTorch
        </label>
        <label>
          <input type="radio" name="lang" checked={lang === "keras"} onChange={() => setLang("keras")} /> Keras
        </label>
      </div>
      <pre className="code-block" tabIndex={0} aria-label="생성된 코드">
        <code>{code}</code>
      </pre>
      <div className="actions">
        <button className="btn ghost" onClick={() => setOpen(false)}>
          닫기
        </button>
        <button className="btn outline" onClick={copy}>
          코드 복사
        </button>
        <button className="btn primary" onClick={zip} aria-busy={zipping}>
          {photos ? `zip 받기 (코드 + 사진 ${photos}장)` : "zip 받기 (코드)"}
        </button>
      </div>
    </dialog>
  );
}

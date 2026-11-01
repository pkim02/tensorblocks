import { Fragment, useMemo, useState } from "react";
import { useApp, useAnalysis } from "../store.js";
import { notebookCells, downloadNotebook, pytorchCode, kerasCode, downloadText, downloadZip } from "../export/code.js";
import ViewTabs from "./ViewTabs.jsx";

const COLAB = "https://colab.research.google.com/";

// "Colab 코드": the model as a Colab notebook (shown cell by cell) or as plain .py scripts.
export default function CodeView() {
  const a = useAnalysis();
  const nodes = useApp((s) => s.nodes);
  const classes = useApp((s) => s.classes);
  const settings = useApp((s) => s.settings);
  const setView = useApp((s) => s.setView);
  const [fmt, setFmt] = useState("nb");
  const [zipping, setZipping] = useState(false);
  const photos = classes.reduce((n, c) => n + c.samples.length, 0);
  const vlm = a.kind === "vlm";

  const cells = useMemo(() => (a.ok ? notebookCells(a, nodes, classes, settings, { photos: "preview" }) : []), [a, nodes, classes, settings]);
  const script = useMemo(() => {
    if (!a.ok || fmt === "nb") return "";
    return fmt === "torch" ? pytorchCode(a, nodes, classes, settings) : kerasCode(a, nodes, classes, settings);
  }, [a, fmt, nodes, classes, settings]);

  const getNotebook = () => downloadNotebook(a, nodes, classes, settings);
  const openColab = () => {
    getNotebook();
    window.open(COLAB, "_blank", "noopener");
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(script);
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
    <div className="code-view">
      <header className="work-bar">
        <ViewTabs />
      </header>
      <div className="code-body">
        <aside className="code-side">
          <h1 className="tour-h">{vlm ? "Colab에서 VLM 돌려 보기" : "Colab에서 학습하기"}</h1>
          <p>
            Colab은 구글이 만든 브라우저 속 파이썬 노트북이에요. 설치 없이 GPU를 무료로 쓸 수 있어요. 블록으로 만든 모델이 노트북 한 개가 되고
            {photos ? `, 모은 사진 ${photos}장도 노트북 안에 함께 들어가요.` : "요. 사진은 Colab에서 올리면 돼요."}
          </p>

          {a.ok ? (
            <>
              <ol className="colab-steps">
                <li>
                  <button className="btn primary block" onClick={openColab}>
                    노트북 받고 Colab 열기
                  </button>
                  <span className="small muted">tensorblocks.ipynb 파일을 받고 Colab을 새 탭으로 열어요.</span>
                </li>
                <li>
                  Colab의 <strong>노트북 열기</strong> 창에서 <strong>업로드</strong>를 누르고 받은 파일을 골라요. 창이 안 보이면 메뉴의{" "}
                  <strong>파일 → 노트북 업로드</strong>예요.
                </li>
                {vlm && <li>처음 한 번은 Hugging Face에서 PaliGemma 약관에 동의하고 토큰을 넣어야 해요. 방법은 노트북 1번 칸에 있어요.</li>}
                <li>
                  메뉴의 <strong>런타임 → 모두 실행</strong>을 누르면 {vlm ? "구글 PaliGemma 2를 받아서 내 사진에 질문해요." : "학습이 시작돼요."}
                </li>
              </ol>
              <p className="small muted">
                Colab에서는 숫자(에폭, 학습률)나 층을 직접 고쳐서 다시 돌려 볼 수 있어요. 고친 노트북은 <strong>파일 → 드라이브에 사본 저장</strong>으로 내 구글
                드라이브에 남겨요.
              </p>
              <div className="code-more">
                <h2 className="small">다른 방법으로 받기</h2>
                <button className="btn outline small" onClick={getNotebook}>
                  노트북만 받기 (.ipynb)
                </button>
                <button
                  className="btn outline small"
                  onClick={() => downloadText(pytorchCode(a, nodes, classes, settings), a.kind === "vlm" ? "vlm_paligemma.py" : "train_pytorch.py")}
                >
                  PyTorch 코드 받기 (.py)
                </button>
                <button className="btn outline small" onClick={zip} aria-busy={zipping}>
                  {photos ? `전체 zip 받기 (코드 + 사진 ${photos}장)` : "전체 zip 받기 (코드)"}
                </button>
              </div>
            </>
          ) : (
            <div className="callout danger">
              <p>
                <strong>모델에 모양이 안 맞는 곳이 있어요.</strong> 고쳐야 코드를 만들 수 있어요.
              </p>
              <button className="btn outline small" onClick={() => setView("canvas")}>
                블록에서 고치기
              </button>
            </div>
          )}
        </aside>

        <section className="code-main" aria-label="코드 미리보기">
          <div className="code-tools">
            <div className="segmented" role="radiogroup" aria-label="코드 형식">
              {[
                ["nb", "Colab 노트북"],
                ["torch", "PyTorch .py"],
                ["keras", "Keras .py"],
              ].map(([id, label]) => (
                <label key={id}>
                  <input type="radio" name="code-fmt" checked={fmt === id} onChange={() => setFmt(id)} /> {label}
                </label>
              ))}
            </div>
            {fmt !== "nb" && a.ok && (
              <button className="btn outline small" onClick={copy}>
                코드 복사
              </button>
            )}
          </div>
          {!a.ok ? (
            <p className="muted">모델을 고치면 여기에 코드가 나와요.</p>
          ) : fmt === "nb" ? (
            <Notebook cells={cells} />
          ) : (
            <pre className="code-block" tabIndex={0} aria-label="생성된 코드">
              <code>{script}</code>
            </pre>
          )}
        </section>
      </div>
    </div>
  );
}

// Looks like Colab: markdown cells as text, code cells with an empty run gutter.
function Notebook({ cells }) {
  return (
    <div className="nb">
      {cells.map((c) => {
        const src = c.source.join("");
        if (c.cell_type === "markdown")
          return (
            <div key={c.metadata.id} className="nb-md">
              <Markdown text={src} />
            </div>
          );
        const form = c.metadata.cellView === "form";
        return (
          <div key={c.metadata.id} className="nb-cell">
            <span className="nb-run" aria-hidden="true">
              [ ]
            </span>
            {form && <p className="nb-form">{src.split("\n")[0].replace("#@title ", "")}</p>}
            <pre className="nb-code">
              <code>{form ? src.split("\n").slice(1).join("\n") : src}</code>
            </pre>
          </div>
        );
      })}
    </div>
  );
}

// Just enough Markdown for the cells we write: headings, paragraphs, one table, **bold**, `code`, links.
function Markdown({ text }) {
  const blocks = text.split(/\n\n+/);
  return blocks.map((b, i) => {
    if (b.startsWith("# ")) return <h2 key={i}>{inline(b.slice(2))}</h2>;
    if (b.startsWith("## ")) return <h3 key={i}>{inline(b.slice(3))}</h3>;
    if (b.startsWith("|")) {
      const rows = b
        .split("\n")
        .filter((r) => !/^\|[-| ]+\|$/.test(r))
        .map((r) =>
          r
            .slice(1, -1)
            .split("|")
            .map((x) => x.trim()),
        );
      return (
        <div key={i} className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                {rows[0].map((h, j) => (
                  <th key={j}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(1).map((r, j) => (
                <tr key={j}>
                  {r.map((x, k) => (
                    <td key={k} className={k ? "num" : ""}>
                      {x}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    return (
      <p key={i}>
        {b.split("\n").map((line, j) => (
          <Fragment key={j}>
            {j > 0 && <br />}
            {inline(line)}
          </Fragment>
        ))}
      </p>
    );
  });
}

function inline(s) {
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/).map((part, i) => {
    if (part.startsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link)
      return (
        <a key={i} href={link[2]} target="_blank" rel="noreferrer">
          {link[1]}
        </a>
      );
    return part;
  });
}

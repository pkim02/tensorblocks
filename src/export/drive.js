// Saves a notebook to the user's Google Drive so Colab can open it directly.
// Uses a resumable upload because notebooks with embedded photos can pass the 5 MB multipart limit.
const UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id";
const NOTEBOOK_MIME = "application/x-ipynb+json";

export const colabUrl = (fileId) => `https://colab.research.google.com/drive/${encodeURIComponent(fileId)}`;

// needsSignIn: the token is no good, so the user has to sign in again before retrying.
function failure(message, needsSignIn = false) {
  return Object.assign(new Error(message), { needsSignIn });
}

async function send(url, options) {
  try {
    return await fetch(url, options);
  } catch (err) {
    throw Object.assign(failure("인터넷 연결을 확인하고 다시 시도해 주세요."), { cause: err });
  }
}

async function responseError(res) {
  if (res.status === 401) return failure("로그인이 만료됐어요. 다시 로그인해 주세요.", true);
  const body = await res.json().catch(() => null);
  const reason = body?.error?.errors?.[0]?.reason ?? body?.error?.status;
  if (reason === "insufficientPermissions" || reason === "insufficientScopes" || reason === "PERMISSION_DENIED")
    return failure("Drive 저장 권한이 필요해요. 다시 로그인할 때 권한에 동의해 주세요.", true);
  if (reason === "storageQuotaExceeded") return failure("Google Drive 저장 공간이 부족해요. 공간을 비우고 다시 시도해 주세요.");
  if (reason === "accessNotConfigured") return failure("Drive 연결 설정이 아직 안 됐어요. 노트북을 받아서 Colab에 직접 올려 주세요.");
  return failure("노트북을 Drive에 올리지 못했어요. 다시 시도해 주세요.");
}

// Returns the new file's id. Throws an Error with a message to show.
export async function uploadNotebook(json, token, name = "tensorblocks.ipynb") {
  const start = await send(UPLOAD_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json; charset=UTF-8",
      "X-Upload-Content-Type": NOTEBOOK_MIME,
    },
    body: JSON.stringify({ name, mimeType: NOTEBOOK_MIME }),
  });
  if (!start.ok) throw await responseError(start);
  const session = start.headers.get("Location");
  if (!session) throw failure("노트북을 Drive에 올리지 못했어요. 다시 시도해 주세요.");

  const done = await send(session, { method: "PUT", headers: { "Content-Type": NOTEBOOK_MIME }, body: new Blob([json], { type: NOTEBOOK_MIME }) });
  if (!done.ok) throw await responseError(done);
  const { id } = await done.json();
  if (!id) throw failure("노트북을 Drive에 올리지 못했어요. 다시 시도해 주세요.");
  return id;
}

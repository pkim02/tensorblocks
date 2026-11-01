import { useEffect, useRef, useState } from "react";

// One camera stream shared by every component that asks for it.
let stream = null;
let users = 0;
let pending = null;

async function acquire() {
  users++;
  if (stream) return stream;
  if (!pending) {
    pending = navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false })
      .then((s) => (stream = s))
      .finally(() => (pending = null));
  }
  return pending;
}

function release() {
  users = Math.max(0, users - 1);
  if (users === 0 && stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// Attaches the shared camera to a <video> while `on` is true.
export function useWebcam(on) {
  const videoRef = useRef(null);
  const [state, setState] = useState("off"); // off | starting | live | error
  const [error, setError] = useState("");

  useEffect(() => {
    if (!on) {
      setState("off");
      return;
    }
    let cancelled = false;
    let got = false;
    setState("starting");
    acquire()
      .then((s) => {
        if (cancelled) return release();
        got = true;
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          v.play().catch(() => {});
        }
        setState("live");
      })
      .catch((e) => {
        users = Math.max(0, users - 1);
        if (cancelled) return;
        setError(e?.name === "NotAllowedError" ? "카메라 권한이 막혀 있어요. 주소창 옆 카메라 아이콘에서 허용해 주세요." : "카메라를 켤 수 없어요. 다른 앱이 쓰고 있지 않은지 확인해 주세요.");
        setState("error");
      });
    return () => {
      cancelled = true;
      if (got) release();
    };
  }, [on]);

  return { videoRef, state, error };
}

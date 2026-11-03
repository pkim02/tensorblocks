import { useState } from "react";
import { isAuthConfigured, signIn, signOutUser, useAuth } from "../auth/google.js";

export default function AuthButton() {
  const user = useAuth((s) => s.user);
  const ready = useAuth((s) => s.ready);
  const [busy, setBusy] = useState(false);

  const run = async (action) => {
    setBusy(true);
    try {
      await action();
    } catch (err) {
      window.kit?.toast(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (user) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="hint">{user.displayName ?? user.email}</span>
        <button className="btn outline small" onClick={() => run(signOutUser)} aria-busy={busy}>
          로그아웃
        </button>
      </div>
    );
  }

  return (
    <button
      className="btn outline small"
      onClick={() => run(signIn)}
      disabled={!isAuthConfigured || !ready}
      aria-busy={busy}
      title={isAuthConfigured ? undefined : "로그인 설정이 아직 안 됐어요"}
    >
      Google 로그인
    </button>
  );
}

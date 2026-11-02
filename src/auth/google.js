// Google sign-in in one place: who is signed in (useAuth) and the token for Drive uploads (getDriveToken).
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, onAuthStateChanged, signInWithPopup, signOut } from "firebase/auth";
import { create } from "zustand";

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

export const isAuthConfigured = Object.values(config).every(Boolean);

// Shared with other Firebase features (Firestore). Null until the VITE_FIREBASE_* values are set.
export const firebaseApp = isAuthConfigured ? initializeApp(config) : null;

const auth = firebaseApp && getAuth(firebaseApp);

const provider = new GoogleAuthProvider();
provider.addScope("https://www.googleapis.com/auth/drive.file");

// user: the Firebase user or null. ready: false until Firebase has restored the last session.
export const useAuth = create(() => ({ user: null, ready: !auth }));

if (auth) onAuthStateChanged(auth, (user) => useAuth.setState({ user, ready: true }));

// Google's access token lasts about an hour and Firebase does not keep it across a page reload,
// so a signed-in user can still have no token. Stop using it a little before it expires.
const TOKEN_LIFETIME_MS = 55 * 60 * 1000;
let driveToken = null;
let driveTokenExpiresAt = 0;

// The token for Drive uploads, or null if the user has to sign in again.
export function getDriveToken() {
  return Date.now() < driveTokenExpiresAt ? driveToken : null;
}

const CANCELLED = ["auth/popup-closed-by-user", "auth/cancelled-popup-request"];

const MESSAGES = {
  "auth/popup-blocked": "로그인 창이 차단됐어요. 브라우저에서 팝업을 허용해 주세요.",
  "auth/unauthorized-domain": "이 주소에서는 아직 로그인할 수 없어요.",
  "auth/network-request-failed": "인터넷 연결을 확인하고 다시 시도해 주세요.",
};

// Returns the signed-in user, or null if the user closed the popup. Throws an Error with a message to show.
export async function signIn() {
  if (!auth) throw new Error("로그인 설정이 아직 안 됐어요.");
  try {
    const result = await signInWithPopup(auth, provider);
    driveToken = GoogleAuthProvider.credentialFromResult(result)?.accessToken ?? null;
    driveTokenExpiresAt = driveToken ? Date.now() + TOKEN_LIFETIME_MS : 0;
    return result.user;
  } catch (err) {
    if (CANCELLED.includes(err.code)) return null;
    throw new Error(MESSAGES[err.code] ?? "로그인하지 못했어요. 다시 시도해 주세요.", { cause: err });
  }
}

export async function signOutUser() {
  driveToken = null;
  driveTokenExpiresAt = 0;
  if (auth) await signOut(auth);
}

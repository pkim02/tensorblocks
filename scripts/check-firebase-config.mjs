import { loadEnv } from "vite";
import { randomUUID } from "node:crypto";

// Read-only diagnostics. Never print configuration values, tokens, or server bodies.
const env = loadEnv("development", process.cwd(), "VITE_FIREBASE_");
const keys = ["API_KEY", "AUTH_DOMAIN", "PROJECT_ID", "APP_ID"];
const config = Object.fromEntries(keys.map((key) => [key, env[`VITE_FIREBASE_${key}`]?.trim()]));
const missing = keys.filter((key) => !config[key]);
if (missing.length) {
  console.log(JSON.stringify({ configured: false, missing }));
  process.exitCode = 1;
} else {
  try {
    const response = await fetch(`https://identitytoolkit.googleapis.com/v1/projects?key=${encodeURIComponent(config.API_KEY)}`, {
      signal: AbortSignal.timeout(15000),
    });
    const result = await response.json();
    const domains = Array.isArray(result.authorizedDomains) ? result.authorizedDomains : [];
    console.log(
      JSON.stringify(
        {
          configured: true,
          authConfigHttpStatus: response.status,
          authConfigAccessible: response.ok,
          localhostAuthorized: domains.includes("localhost"),
          loopbackAuthorized: domains.includes("127.0.0.1"),
          productionDomainAuthorized: domains.includes("tensorblocks.netlify.app"),
          configuredAuthDomainAuthorized: domains.includes(config.AUTH_DOMAIN),
          projectIdFormatValid: /^[a-z][a-z0-9-]{4,29}$/.test(config.PROJECT_ID),
          // Identity Toolkit commonly returns the numeric project NUMBER, not the textual project ID.
          projectNumberMatchesAppId: /^\d+$/.test(String(result.projectId)) ? String(result.projectId) === config.APP_ID.split(":")[1] : "not returned",
          textualProjectIdMatches:
            result.projectId && !/^\d+$/.test(String(result.projectId)) ? String(result.projectId) === config.PROJECT_ID : "not returned",
          // These checks do not authenticate a user, validate appId, inspect rules, or write documents.
        },
        null,
        2,
      ),
    );
    if (!response.ok) process.exitCode = 1;
    const probe = `codex-readonly-${randomUUID()}`;
    const firestoreResponse = await fetch(
      `https://firestore.googleapis.com/v1/projects/${encodeURIComponent(config.PROJECT_ID)}/databases/(default)/documents/users/${probe}/projects/${probe}`,
      { headers: { "X-Goog-Api-Key": config.API_KEY }, signal: AbortSignal.timeout(15000) },
    );
    const probeResult = await firestoreResponse.json();
    console.log(
      JSON.stringify(
        {
          firestoreReadOnlyProbeHttpStatus: firestoreResponse.status,
          unauthenticatedReadDenied: firestoreResponse.status === 403 || firestoreResponse.status === 401,
          firestoreRulesPermissionMessage: /missing or insufficient permissions/i.test(probeResult.error?.message ?? ""),
          firestoreErrorStatus: probeResult.error?.status ?? null,
          note: "GET of a fresh random document path only. No writes and no existing documents requested. This does not verify deployed owner rules.",
        },
        null,
        2,
      ),
    );
  } catch {
    console.log(JSON.stringify({ configured: true, authConfigAccessible: false, reason: "network-or-response-error" }));
    process.exitCode = 1;
  }
}

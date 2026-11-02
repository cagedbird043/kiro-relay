import { describe, expect, test } from "bun:test";
import type { FetchFn } from "../src/kiro/base.ts";
import { refreshKiroToken } from "../src/kiro/refresh.ts";

const oidc = { authMethod: "idc", clientId: "client", clientSecret: "secret", region: "eu-north-1" };
const external = {
  authMethod: "external_idp", clientId: "public-client",
  tokenEndpoint: "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
  scopes: ["offline_access", "api://client/conversations"],
};
const flows = [
  { name: "OIDC", data: oidc, endpoint: "https://oidc.eu-north-1.amazonaws.com/token", oauth: false },
  { name: "social", data: { authMethod: "imported", clientId: "unused", clientSecret: "unused" }, endpoint: "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken", oauth: false },
  { name: "external IdP", data: external, endpoint: external.tokenEndpoint, oauth: true },
];

describe("typed Kiro refresh grants", () => {
  for (const flow of flows) {
    test(`${flow.name} rotates tokens through one injected request`, async () => {
      let calls = 0;
      const fetchImpl: FetchFn = async (input, init) => {
        calls++;
        expect(String(input)).toBe(flow.endpoint);
        expect(init?.signal).toBeInstanceOf(AbortSignal);
        expect(init?.signal?.aborted).toBe(false);
        if (flow.oauth) {
          const body = new URLSearchParams(String(init?.body));
          expect(body.get("grant_type")).toBe("refresh_token");
          expect(body.get("refresh_token")).toBe("old-refresh");
          expect(body.has("client_secret")).toBe(false);
        } else {
          expect(JSON.parse(String(init?.body)).refreshToken).toBe("old-refresh");
        }
        return Response.json(flow.oauth
          ? { access_token: "new-access", refresh_token: "rotated", expires_in: 7200 }
          : { accessToken: "new-access", refreshToken: "rotated", expiresIn: 7200 });
      };
      expect(await refreshKiroToken("old-refresh", flow.data, null, fetchImpl)).toEqual({
        ok: true, accessToken: "new-access", refreshToken: "rotated", expiresIn: 7200,
      });
      expect(calls).toBe(1);
    });

    for (const code of ["InvalidGrantException", "ExpiredTokenException", "invalid_grant", "invalid_client"]) {
      test(`${flow.name} classifies ${code} without registration or retries`, async () => {
        let calls = 0;
        const fetchImpl: FetchFn = async () => {
          calls++;
          return Response.json(flow.oauth ? { error: code } : { __type: code }, { status: 400 });
        };
        expect(await refreshKiroToken("refresh", flow.data, null, fetchImpl)).toEqual({ ok: false, kind: "unrecoverable", code });
        expect(calls).toBe(1);
      });
    }

    for (const status of [429, 500, 503]) {
      test(`${flow.name} treats HTTP ${status} as transient`, async () => {
        const result = await refreshKiroToken("refresh", flow.data, null,
          async () => Response.json({ error: "invalid_grant" }, { status }));
        expect(result).toEqual({ ok: false, kind: "transient", code: `http_${status}` });
      });
    }
  }

  test("rejects invalid region and missing config before sending credentials", async () => {
    const fetchImpl: FetchFn = async () => { throw new Error("must not fetch"); };
    for (const region of ["us-east-1.attacker.example", "us-east-1/path", "", 42]) {
      expect(await refreshKiroToken("refresh", { ...oidc, region }, null, fetchImpl)).toEqual({ ok: false, kind: "invalid_config", code: "invalid_region" });
    }
    expect(await refreshKiroToken(undefined, oidc, null, fetchImpl)).toEqual({ ok: false, kind: "invalid_config", code: "missing_refresh_token" });
    expect(await refreshKiroToken("refresh", { authMethod: "idc", clientId: "client" }, null, fetchImpl)).toEqual({ ok: false, kind: "invalid_config", code: "missing_oidc_client" });
    expect(await refreshKiroToken("refresh", { authMethod: "external_idp" }, null, fetchImpl)).toEqual({ ok: false, kind: "invalid_config", code: "invalid_external_idp_config" });
  });

  test("accepts the common government-region shape and retains an unrotated refresh token", async () => {
    expect(await refreshKiroToken("refresh", { ...oidc, region: "us-gov-west-1" }, null, async (input) => {
      expect(String(input)).toBe("https://oidc.us-gov-west-1.amazonaws.com/token");
      return Response.json({ accessToken: "new", expiresIn: 60 });
    })).toEqual({ ok: true, accessToken: "new", refreshToken: "refresh", expiresIn: 60 });
  });

  test("network and malformed token responses are transient", async () => {
    expect(await refreshKiroToken("refresh", oidc, null, async () => { throw new TypeError("private network error"); })).toEqual({ ok: false, kind: "transient", code: "refresh_network_error" });
    for (const body of [{}, { accessToken: "access", expiresIn: -1 }, { accessToken: "access", expiresIn: 1e100 }]) {
      expect(await refreshKiroToken("refresh", oidc, null, async () => Response.json(body))).toEqual({ ok: false, kind: "transient", code: "invalid_token_response" });
    }
  });

  test("15 second timeout remains active while reading the response body", async () => {
    let signal: AbortSignal | undefined;
    const start = Date.now();
    const result = await refreshKiroToken("refresh", oidc, null, async (_input, init) => {
      signal = init?.signal ?? undefined;
      return new Response(new ReadableStream({
        start(controller) {
          signal!.addEventListener("abort", () => controller.error(signal!.reason), { once: true });
          controller.enqueue(new TextEncoder().encode('{"accessToken":"unfinished'));
        },
      }));
    });
    expect(signal?.aborted).toBe(true);
    expect(Date.now() - start).toBeGreaterThanOrEqual(14_900);
    expect(result).toEqual({ ok: false, kind: "transient", code: "refresh_timeout" });
  }, 20_000);
});

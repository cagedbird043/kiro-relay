import type { FetchFn } from "../src/kiro/base.ts";

/**
 * 在 `run` 期间用 `fake` 替换全局 fetch，结束后恢复。
 * Bun 的 `typeof fetch` 额外要求 `preconnect`，替身只需调用签名，因此在这里统一补齐。
 */
export async function withFetch<T>(fake: FetchFn, run: () => Promise<T>): Promise<T> {
  const original = globalThis.fetch;
  globalThis.fetch = Object.assign(fake, { preconnect: original.preconnect });
  try {
    return await run();
  } finally {
    globalThis.fetch = original;
  }
}

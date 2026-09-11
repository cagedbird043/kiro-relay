/**
 * Live model catalog.
 *
 * The relay used to expose exactly the models hand-listed in `config.yaml`,
 * which made that file the sole gate: an id missing from it was invisible to
 * `GET /v1/models` and rejected outright by every request. This module replaces
 * that gate with Kiro's own `ListAvailableModels` response, so the exposed set
 * is whatever the account is actually entitled to.
 *
 * Guarantees:
 *   - Discovery is best-effort. Every failure path falls back to a usable
 *     catalog; nothing here ever throws into a request.
 *   - Stale-while-error: once a live snapshot has been seen it keeps serving,
 *     so an upstream blip never makes models vanish mid-session.
 *   - Refresh is non-blocking. Requests are served from the current snapshot
 *     while a stale one is refreshed in the background.
 */

import type { AppConfig } from "./config.ts";
import type { ProviderCredentials } from "./kiro/base.ts";
import { MODEL_CONTEXT_LENGTHS } from "./kiro/config.ts";
import { fetchKiroAvailableModels, type KiroModel } from "./kiro/models.ts";

export type CatalogEntry = { id: string; contextLength: number };

export type Catalog = {
  /** Current snapshot; triggers a background refresh when stale. */
  list(): CatalogEntry[];
  has(id: string): boolean;
  contextLengthFor(id: string): number | undefined;
  /** Await a discovery pass. Used by tests and by startup warm-up. */
  refresh(): Promise<void>;
};

export type CatalogOptions = {
  /** Manual entries from config.yaml: fallback set plus context-window overrides. */
  models: AppConfig["models"];
  /** Reads the account credentials. Failures are treated as "no token". */
  loadCredentials: () => Promise<ProviderCredentials | null | undefined>;
  /** Injectable for tests; defaults to the real Kiro discovery call. */
  fetchModels?: typeof fetchKiroAvailableModels;
  ttlMs?: number;
  errorRetryMs?: number;
};

const DEFAULT_TTL_MS = 5 * 60_000;
const DEFAULT_ERROR_RETRY_MS = 30_000;

/**
 * `auto` is Kiro's server-side router rather than a real model: the id that
 * actually serves a request is chosen upstream per turn, so its context window
 * and capabilities cannot be known here. Exposing it would produce unpredictable
 * usage accounting, so it is deliberately withheld.
 */
const EXCLUDED_MODEL_IDS: Record<string, true> = { auto: true };

function entriesFrom(lengths: Map<string, number>): CatalogEntry[] {
  const entries: CatalogEntry[] = [];
  for (const [id, contextLength] of lengths) entries.push({ id, contextLength });
  return entries;
}

/**
 * Offline catalog: the static table mirrors Kiro's published windows, and
 * config.yaml may both add ids and override windows. Used before the first
 * successful discovery and whenever discovery has never succeeded.
 */
function fallbackLengths(models: AppConfig["models"]): Map<string, number> {
  const lengths = new Map<string, number>();
  for (const [id, contextLength] of Object.entries(MODEL_CONTEXT_LENGTHS)) {
    if (!EXCLUDED_MODEL_IDS[id]) lengths.set(id, contextLength);
  }
  for (const [id, spec] of Object.entries(models)) {
    if (!EXCLUDED_MODEL_IDS[id]) lengths.set(id, spec.contextLength);
  }
  return lengths;
}

/**
 * Live catalog: exactly the ids Kiro routes on. Synthetic `-thinking` aliases
 * produced by the discovery expander are dropped — they are request-time
 * selectors, not separate models, and listing them would clutter the catalog
 * with ids that do not exist upstream.
 *
 * config.yaml may override a discovered window (an escape hatch for a wrong
 * upstream value) but may not introduce ids the account lacks.
 */
function discoveredLengths(
  models: AppConfig["models"],
  discovered: KiroModel[]
): Map<string, number> {
  const lengths = new Map<string, number>();
  for (const model of discovered) {
    if (!model.id || EXCLUDED_MODEL_IDS[model.id]) continue;
    if (model.upstreamModelId && model.upstreamModelId !== model.id) continue;
    const contextLength =
      typeof model.contextLength === "number" && model.contextLength > 0
        ? model.contextLength
        : MODEL_CONTEXT_LENGTHS[model.id];
    if (contextLength) lengths.set(model.id, contextLength);
  }
  for (const [id, spec] of Object.entries(models)) {
    if (lengths.has(id)) lengths.set(id, spec.contextLength);
  }
  return lengths;
}

export function createCatalog(options: CatalogOptions): Catalog {
  const {
    models,
    loadCredentials,
    fetchModels = fetchKiroAvailableModels,
    ttlMs = DEFAULT_TTL_MS,
    errorRetryMs = DEFAULT_ERROR_RETRY_MS,
  } = options;

  let lengths = fallbackLengths(models);
  let snapshot = entriesFrom(lengths);
  let freshUntil = 0;
  let inFlight: Promise<void> | undefined;

  const refresh = async (): Promise<void> => {
    let discovered: KiroModel[] | undefined;
    try {
      const credentials = await loadCredentials();
      const result = await fetchModels({
        accessToken: credentials?.accessToken,
        providerSpecificData: credentials?.providerSpecificData,
      });
      if (result.source === "api") discovered = result.models;
    } catch {
      // Discovery is advisory. A credential read or network failure must never
      // surface to a request, so the previous snapshot simply keeps serving.
    }

    if (discovered && discovered.length > 0) {
      const next = discoveredLengths(models, discovered);
      if (next.size > 0) {
        lengths = next;
        snapshot = entriesFrom(next);
        freshUntil = Date.now() + ttlMs;
        return;
      }
    }
    freshUntil = Date.now() + errorRetryMs;
  };

  const ensureFresh = () => {
    if (inFlight || Date.now() < freshUntil) return;
    inFlight = refresh().finally(() => {
      inFlight = undefined;
    });
  };

  return {
    list() {
      ensureFresh();
      return snapshot;
    },
    has(id: string) {
      ensureFresh();
      return lengths.has(id);
    },
    contextLengthFor(id: string) {
      ensureFresh();
      return lengths.get(id);
    },
    refresh,
  };
}

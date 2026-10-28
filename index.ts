/**
 * OpenClaw Memory (SynapCores) Plugin
 *
 * Long-term memory with vector search for AI conversations.
 * Uses SynapCores AIDB for storage via the engine-side
 * `MEMORY_STORE` / `MEMORY_RECALL` / `MEMORY_FORGET` primitives, and the
 * engine-native `client.embed()` for the relevance-extension features that
 * need a client-side embedding (cosine feature in `predictRelevance`,
 * graph-node embedding when `autoLinkSimilar` is on). No external
 * embedding provider is required.
 *
 * Provides seamless auto-recall and auto-capture via lifecycle hooks,
 * plus four SynapCores-only extensions (SQL-filtered recall, graph-relation
 * walks, AutoML relevance scoring, and a model-training helper) — see
 * `recallFiltered`, `recallRelated`, `predictRelevance`, and
 * `trainRelevanceModel`.
 *
 * v0.4.0 migrates the core memory ops to `@synapcores/sdk@^0.5.0`'s
 * `client.memory` surface. Requires SynapCores gateway v1.8.5-ce+.
 *
 * v0.7.0 adds a SECOND backend alongside that one: the v1.14.3+ `CREATE MEMORY`
 * object (SRS v1.2) and its five verbs — REMEMBER / RECALL / CURRENT / FORGET /
 * TRACE — over the gateway's `/v1/memory/*` REST surface. The plugin probes the
 * gateway once and picks; engines that predate the feature keep running on the
 * legacy `MEMORY_*` path unchanged. See MIGRATION.md for the full mapping, the
 * identity design, and what the new backend cannot do.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { Type } from "typebox";
import { stringEnum } from "openclaw/plugin-sdk/core";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { SILENT_REPLY_TOKEN } from "openclaw/plugin-sdk/reply-runtime";
import { SynapCores } from "@synapcores/sdk";
import type { MemoryRecord } from "@synapcores/sdk";
import {
  MEMORY_CATEGORIES,
  type MemoryCategory,
  type MemoryObjectConfig,
  type MemoryCreateRequest,
  memoryConfigSchema,
} from "./config.js";

// ============================================================================
// Types
// ============================================================================

/**
 * Internal memory record shape used throughout the plugin. The legacy v0.3.x
 * surface guaranteed `vector: number[]` on every entry because storage went
 * through `client.vectorCollection(...)`. v0.4.0's `client.memory.recall`
 * does not return embeddings — `vector` is filled lazily by callers that
 * need it (currently only `predictRelevance` / `trainRelevanceModel`).
 */
export type MemoryEntry = {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: MemoryCategory;
  createdAt: number;
};

export type MemorySearchResult = {
  entry: MemoryEntry;
  /**
   * RELEVANCE — the retrieval score, in [0, 1]. On the legacy backend this is
   * `MEMORY_RECALL`'s cosine `similarity`; on the memory-object backend it is
   * the envelope's `relevance`.
   *
   * This is NOT confidence. The engine keeps the two apart on purpose
   * (SRS v1.2): relevance says "this matched your question", confidence says
   * "the evidence for this claim is strong". A highly relevant historical
   * statement can be non-authoritative, so the plugin never folds one into the
   * other — see `confidence` below.
   */
  score: number;
  /**
   * CONFIDENCE — evidence strength in [0, 1], when the backend reports one.
   * Only the memory-object backend does; legacy rows leave it undefined.
   */
  confidence?: number;
};

/**
 * Options for `recallFiltered`: a SynapCores-only extension that pairs a
 * semantic vector search with a SQL `WHERE` clause (date ranges, tags,
 * importance thresholds, category equality, etc.).
 */
export type RecallFilteredOptions = {
  /**
   * SQL fragment applied as the search filter.
   *
   * v0.4.0 note: the WHERE clause is now evaluated against the columns
   * returned by `MEMORY_RECALL(?, ?, ?)` — `id`, `content`, `similarity`,
   * `metadata`, `created_at`. The legacy v0.3.x `category = 'preference'`
   * shorthand still works for callers that wrote category/importance into
   * the metadata blob (the plugin always does so) BUT the path is now
   * engine-side JSON access: `metadata->>'category' = 'preference'` is the
   * portable form. The plugin auto-rewrites a handful of the legacy
   * shorthands (`category`, `importance`, `createdAt`, `text`) into the
   * JSON-extract form so most existing callers keep working unchanged.
   */
  where: string;
  /** Natural-language query that gets embedded engine-side and used as the search vector. */
  semantic: string;
  /** Maximum results to return (default: 5). */
  limit?: number;
};

/**
 * Options for `recallRelated`: a SynapCores-only extension that walks the
 * memory graph from a given memory ID and returns the neighborhood.
 */
export type RecallRelatedOptions = {
  /** Maximum graph hops to traverse (default: 1). Capped at 4 to avoid runaway walks. */
  hops?: number;
  /**
   * Restrict to specific edge kinds (default: `["SIMILAR_TO"]`). The
   * gateway exposes `SIMILAR_TO` as a synthetic edge derived from
   * cosine similarity over the `embedding` property the plugin writes
   * onto every captured Memory node. Other edge kinds (e.g. `MENTIONS`,
   * `RELATES_TO`) are honoured if the caller has populated them.
   */
  edgeKinds?: string[];
  /**
   * Cosine-similarity threshold for synthetic `SIMILAR_TO` edges
   * (default: 0.5). Higher values return fewer, more closely related
   * memories. Ignored for non-synthetic edge kinds.
   */
  similarityThreshold?: number;
  /** Maximum results to return (default: 20). */
  limit?: number;
};

/**
 * One result returned from `recallRelated` — the neighbor memory plus the
 * shortest path length (hops) from the source memory.
 */
export type RelatedMemoryResult = {
  entry: MemoryEntry;
  hops: number;
  via: string[];
};

/**
 * One result returned from `predictRelevance` — the candidate memory plus
 * a model-predicted relevance score in [0, 1].
 */
export type RelevanceScoredMemory = {
  entry: MemoryEntry;
  /** Predicted relevance score in [0, 1]. */
  relevance: number;
};

/**
 * One labelled training sample for `trainRelevanceModel`.
 */
export type RelevanceFeedback = {
  /** UUID of the memory the user actually engaged with (or didn't). */
  memoryId: string;
  /** Natural-language query the user issued. */
  queryText: string;
  /** Engagement score in [0, 1] (1 = highly relevant, 0 = not relevant). */
  score: number;
};

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_COLLECTION = "openclaw_memories";

// ============================================================================
// Connection preflight
//
// This plugin is a CLIENT — it never installs or starts a SynapCores database.
// The most common first-run failure is "installed the plugin, but the gateway
// isn't running", which otherwise surfaces as a raw ECONNREFUSED. Preflight
// turns that into an actionable message pointing at the installer + admin UI.
// ============================================================================

const PREFLIGHT_INSTALL_HINT =
  "This plugin needs a running SynapCores gateway — it does not install one. " +
  "Install + start the free Community Edition:\n" +
  "  curl -fsSL https://synapcores.com/install.sh | sh\n" +
  "  synapcores start\n" +
  "Docs: https://synapcores.com/install";

function httpStatusOf(err: unknown): number | undefined {
  const e = err as {
    status?: number;
    statusCode?: number;
    response?: { status?: number };
  };
  return e?.status ?? e?.statusCode ?? e?.response?.status;
}

// Raw transport-error fallback (used if the error isn't one of the SDK's typed
// classes — e.g. a bare fetch/undici error).
function isRawConnError(err: unknown): boolean {
  const code = String((err as { code?: string })?.code ?? "");
  const msg = String((err as { message?: string })?.message ?? err ?? "");
  return (
    [
      "ECONNREFUSED",
      "ENOTFOUND",
      "ETIMEDOUT",
      "ECONNRESET",
      "EAI_AGAIN",
      "EHOSTUNREACH",
      "EHOSTDOWN",
    ].includes(code) ||
    /ECONNREFUSED|ENOTFOUND|ETIMEDOUT|ECONNRESET|EHOSTUNREACH|getaddrinfo|fetch failed|network error|socket hang up|Failed to connect to SynapCores|Connection refused/i.test(
      msg,
    )
  );
}

/**
 * Classify a gateway error. The @synapcores/sdk normalises failures into typed
 * errors that carry a stable `.code` ("CONNECTION_ERROR" / "AUTH_ERROR") and a
 * `.name` ("ConnectionError" / "AuthenticationError" / "TimeoutError"); we key
 * off those first, then fall back to raw transport heuristics + HTTP status.
 */
function classifyGatewayError(err: unknown): "connection" | "auth" | "other" {
  const code = String((err as { code?: string })?.code ?? "");
  const name = String((err as { name?: string })?.name ?? "");
  const status = httpStatusOf(err);
  if (
    code === "CONNECTION_ERROR" ||
    code === "TIMEOUT" ||
    name === "ConnectionError" ||
    name === "TimeoutError" ||
    isRawConnError(err)
  ) {
    return "connection";
  }
  if (code === "AUTH_ERROR" || name === "AuthenticationError" || status === 401 || status === 403) {
    return "auth";
  }
  return "other";
}

/**
 * One lightweight round-trip to confirm the gateway is reachable AND the API
 * key is accepted, before any memory operation runs. Throws an Error with an
 * actionable message on failure; resolves on success.
 *
 * v0.4.0 uses `client.executeQuery({ sql: "SELECT 1" })` — the lowest-cost
 * authenticated probe available on the gateway and the same path
 * `client.memory.*` rides for every operation.
 */
async function preflightGateway(client: SynapCores): Promise<void> {
  let host = "localhost";
  let port = 8080;
  let scheme = "http";
  try {
    const cfg = (client as unknown as { _getConfig?: () => { host?: string; port?: number; useHttps?: boolean } })._getConfig?.();
    if (cfg) {
      host = cfg.host ?? host;
      port = cfg.port ?? port;
      scheme = cfg.useHttps ? "https" : "http";
    }
  } catch {
    // _getConfig is best-effort — only used to make the message specific.
  }
  const url = `${scheme}://${host}:${port}`;

  // No key configured at all. Since 0.6.4 the plugin installs without config
  // (requiring it made `openclaw plugins install` abort before the user could
  // ever configure it), so an unconfigured first run is expected and deserves
  // its own message rather than a confusing "rejected the API key".
  let configuredKey: string | undefined;
  try {
    configuredKey = (
      client as unknown as { _getConfig?: () => { apiKey?: string } }
    )._getConfig?.()?.apiKey;
  } catch {
    // best-effort
  }
  if (configuredKey !== undefined && configuredKey.trim() === "") {
    throw new Error(
      "SynapCores memory is installed but not configured yet. Set " +
        "plugins.entries.memory-synapcores.config.synapcores.apiKey in openclaw.json, " +
        "or export SYNAPCORES_API_KEY. " +
        PREFLIGHT_INSTALL_HINT,
    );
  }

  try {
    await client.executeQuery({ sql: "SELECT 1" });
  } catch (err) {
    const kind = classifyGatewayError(err);
    if (kind === "connection") {
      throw new Error(`Cannot reach the SynapCores gateway at ${url}. ${PREFLIGHT_INSTALL_HINT}`);
    }
    if (kind === "auth") {
      throw new Error(
        `The SynapCores gateway at ${url} rejected the API key. ` +
          `Create a FullAccess key in the admin UI (http://${host}:8095) and set ` +
          `synapcores.apiKey (or the SYNAPCORES_API_KEY env var).`,
      );
    }
    throw new Error(
      `SynapCores preflight against ${url} failed: ${String(
        (err as { message?: string })?.message ?? err,
      )}`,
    );
  }
}

// ============================================================================
// MemoryDB — thin wrapper around @synapcores/sdk@0.5.0's MemoryClient.
// ============================================================================
//
// v0.4.0 swaps the v0.3.x `client.vectorCollection(name).insert/search/delete`
// path for `client.memory.store/recall/forget`. The engine now owns
// embedding (one round-trip instead of two), the table schema, and the
// vector index — the plugin's job is to translate between OpenClaw's
// `MemoryEntry` shape and the engine's `MemoryRecord` shape.
//
// HARD CUT MIGRATION: the engine-managed table is `_memory_<namespace>`,
// which is a different storage backend from the v0.3.x vector collection.
// Existing v0.3.x installs WILL NOT see their old memories — see the
// README "Migration" section.

class MemoryDB {
  private readonly client: SynapCores;
  private preflightPromise: Promise<void> | null = null;

  constructor(
    client: SynapCores,
    private readonly namespace: string,
  ) {
    this.client = client;
  }

  /**
   * One-time gateway preflight. Lazy — only runs on the first call,
   * matching the v0.3.x init semantics.
   *
   * Public since v0.7.0: capability detection runs this BEFORE probing
   * `/v1/memory`, so "the gateway isn't running" and "the key was rejected"
   * still produce their actionable messages instead of surfacing as a raw
   * transport error from the probe.
   */
  async ensureReady(): Promise<void> {
    if (this.preflightPromise) {
      return this.preflightPromise;
    }
    this.preflightPromise = preflightGateway(this.client);
    try {
      await this.preflightPromise;
    } catch (err) {
      // Reset on failure so the next call can retry.
      this.preflightPromise = null;
      throw err;
    }
  }

  /**
   * Store a memory. `vector` on the input is IGNORED — the engine embeds
   * `text` server-side via the configured embedding model. Returns the
   * fully-populated entry (with engine-assigned id + timestamp).
   */
  async store(entry: Omit<MemoryEntry, "id" | "createdAt">): Promise<MemoryEntry> {
    await this.ensureReady();
    const createdAt = Date.now();
    const id = await this.client.memory.store(this.namespace, entry.text, {
      metadata: {
        importance: entry.importance,
        category: entry.category,
        createdAt,
      },
    });
    return {
      ...entry,
      id,
      createdAt,
    };
  }

  /**
   * Semantic recall by free-text query. `minScore` is applied client-side
   * to the engine's `similarity` field (already in [0, 1]).
   */
  async searchByQuery(
    query: string,
    limit = 5,
    minScore = 0.5,
  ): Promise<MemorySearchResult[]> {
    await this.ensureReady();
    const records = await this.client.memory.recall(this.namespace, query, {
      topK: clampTopK(limit),
    });
    return records
      .map((r) => recordToResult(r))
      .filter((r) => r.score >= minScore);
  }

  /**
   * SQL-filtered recall: a semantic vector search paired with a `WHERE`
   * predicate over category / importance / createdAt / text / `metadata->>'…'`.
   *
   * ENGINE BUG (SynapCores gateway, confirmed v1.6.5.2-ce … v1.9.x-ce):
   * applying ANY `WHERE` to the table-valued `MEMORY_RECALL(?, ?, ?)`
   * function drops every row and returns an empty column set — even a bare
   * `WHERE metadata->>'category' = 'preference'`. The engine cannot filter a
   * table-valued-function result-set in the same SELECT. See the repro in the
   * revalidation report; this needs an engine-side fix (planner should push
   * the predicate as a post-filter over the TVF output, or materialize the
   * TVF before applying WHERE).
   *
   * WORKAROUND (this method): fetch an oversampled, UNFILTERED
   * `SELECT … FROM MEMORY_RECALL(?, ?, ?)` and apply the WHERE predicate
   * CLIENT-SIDE in JS (see {@link compileWherePredicate}). The predicate
   * evaluates against the recall row's category / importance / createdAt /
   * text / id / similarity and its parsed `metadata` blob, so the documented
   * filtering surface works despite the engine gap. Once the engine bug is
   * fixed this can revert to a single WHERE-bearing SQL statement.
   *
   * Legacy shorthand (`category`, `importance`, `createdAt`, `text`) and the
   * portable `metadata->>'…'` JSON-extract form are both accepted.
   */
  async searchFiltered(
    semantic: string,
    where: string,
    limit = 5,
  ): Promise<MemorySearchResult[]> {
    await this.ensureReady();
    // Oversample so the client-side post-filter still has enough rows to return.
    const oversample = clampTopK(Math.max(limit * 5, 25));
    const sql =
      "SELECT id, content, similarity, metadata, created_at " +
      `FROM MEMORY_RECALL($1, $2, $3) LIMIT ${oversample}`;
    let result;
    try {
      result = await this.client.executeQuery({
        sql,
        parameters: [this.namespace, semantic, oversample],
      });
    } catch (err) {
      // Missing namespace == empty result set.
      if (isMissingNamespaceError(err)) {
        return [];
      }
      throw err;
    }
    const { predicate } = compileWherePredicate(where);
    const filtered = mapRecallRowsRich(result).filter((r) =>
      predicate({ entry: r.result.entry, score: r.result.score, metadata: r.metadata }),
    );
    return filtered.slice(0, Number(limit)).map((r) => r.result);
  }

  async delete(id: string): Promise<boolean> {
    await this.ensureReady();
    if (typeof id !== "string" || id.length === 0) {
      throw new Error(`Invalid memory ID: ${String(id)}`);
    }
    return this.client.memory.forget(this.namespace, id);
  }

  /**
   * Count memories in the namespace. The engine table is
   * `_memory_<namespace>`; we count via a direct SQL probe. Best-effort —
   * returns 0 if the namespace hasn't been written to yet.
   */
  async count(): Promise<number> {
    await this.ensureReady();
    const table = `_memory_${this.namespace}`;
    try {
      const result = await this.client.executeQuery({
        sql: `SELECT COUNT(*) FROM ${table}`,
      });
      const first = result.rows?.[0];
      const raw = Array.isArray(first) ? first[0] : undefined;
      if (typeof raw === "number") return raw;
      if (typeof raw === "string") {
        const n = Number(raw);
        return Number.isFinite(n) ? n : 0;
      }
      return 0;
    } catch (err) {
      if (isMissingNamespaceError(err)) return 0;
      return 0;
    }
  }

  /**
   * Fetch a single memory by id. Used by extensions (e.g.
   * `trainRelevanceModel`) to hydrate feedback rows. Returns null if not
   * found.
   *
   * Strategy: oversample MEMORY_RECALL with the id as the query string
   * (any free text works — we only care about the id-filtered result),
   * then look up by id. This avoids reaching into the engine-managed
   * `_memory_<ns>` table directly.
   */
  async get(id: string): Promise<MemoryEntry | null> {
    await this.ensureReady();
    if (typeof id !== "string" || id.length === 0) return null;
    try {
      const result = await this.client.executeQuery({
        sql:
          "SELECT id, content, similarity, metadata, created_at " +
          "FROM MEMORY_RECALL($1, $2, $3) WHERE id = $4 LIMIT 1",
        parameters: [this.namespace, " ", 100, id],
      });
      const mapped = mapRecallRows(result);
      if (mapped.length === 0) return null;
      return mapped[0].entry;
    } catch (err) {
      if (isMissingNamespaceError(err)) return null;
      throw err;
    }
  }

  /** Expose the namespace (used by recallRelated for graph queries). */
  get ns(): string {
    return this.namespace;
  }
}

// ============================================================================
// Shape translators between engine MemoryRecord and plugin MemoryEntry
// ============================================================================

function clampTopK(k: number): number {
  if (!Number.isFinite(k) || k < 1) return 1;
  if (k > 100) return 100;
  return Math.floor(k);
}

function metadataField(
  metadata: Record<string, unknown> | null | undefined,
  key: string,
): unknown {
  if (!metadata) return undefined;
  return metadata[key];
}

function recordToEntry(record: MemoryRecord): MemoryEntry {
  const meta = record.metadata ?? null;
  const importance = metadataField(meta, "importance");
  const category = metadataField(meta, "category");
  const createdAtMeta = metadataField(meta, "createdAt");
  const createdAtTs = record.createdAt instanceof Date
    ? record.createdAt.getTime()
    : Number.NaN;
  return {
    id: record.id,
    text: record.content,
    vector: [],
    importance: typeof importance === "number" ? importance : 0,
    category: (typeof category === "string"
      ? (category as MemoryCategory)
      : "other") as MemoryCategory,
    createdAt: typeof createdAtMeta === "number"
      ? createdAtMeta
      : Number.isFinite(createdAtTs)
        ? createdAtTs
        : 0,
  };
}

function recordToResult(record: MemoryRecord): MemorySearchResult {
  return {
    entry: recordToEntry(record),
    score: clamp01(record.similarity),
  };
}

/**
 * Parse a raw `executeQuery` result-set into MemorySearchResult[].
 * Mirrors @synapcores/sdk@0.5.0's MemoryClient row mapping but operates
 * on `executeQuery` output (which is what we use here for the WHERE-clause
 * pass-through).
 */
function mapRecallRows(result: {
  columns?: Array<{ name?: string } | string>;
  rows?: unknown[][];
}): MemorySearchResult[] {
  return mapRecallRowsRich(result).map((r) => r.result);
}

/**
 * Like {@link mapRecallRows}, but also surfaces the parsed `metadata` blob
 * alongside each result. Used by `searchFiltered` so the WHERE clause can be
 * evaluated client-side against the full metadata object (see
 * {@link compileWherePredicate} and the ENGINE-BUG note on `searchFiltered`).
 */
function mapRecallRowsRich(result: {
  columns?: Array<{ name?: string } | string>;
  rows?: unknown[][];
}): Array<{ result: MemorySearchResult; metadata: Record<string, unknown> | null }> {
  const cols = (result.columns ?? []).map((c) =>
    typeof c === "string" ? c : (c?.name ?? ""),
  );
  const colIndex = (name: string): number => cols.findIndex((c) => c === name);
  const idIdx = colIndex("id");
  const contentIdx = colIndex("content");
  const simIdx = colIndex("similarity");
  const metaIdx = colIndex("metadata");
  const createdIdx = colIndex("created_at");

  const rows = result.rows ?? [];
  const out: Array<{
    result: MemorySearchResult;
    metadata: Record<string, unknown> | null;
  }> = [];
  for (const row of rows) {
    if (!Array.isArray(row)) continue;
    const meta = parseMetadataValue(metaIdx >= 0 ? row[metaIdx] : undefined);
    const createdRaw = createdIdx >= 0 ? row[createdIdx] : undefined;
    const createdAtMs = parseTimestampMs(createdRaw);
    const importance = metadataField(meta, "importance");
    const category = metadataField(meta, "category");
    const similarity = simIdx >= 0 ? toFiniteNumber(row[simIdx]) : 0;
    const id = idIdx >= 0 ? String(row[idIdx] ?? "") : "";
    const content = contentIdx >= 0 ? String(row[contentIdx] ?? "") : "";
    out.push({
      result: {
        entry: {
          id,
          text: content,
          vector: [],
          importance: typeof importance === "number" ? importance : 0,
          category: (typeof category === "string"
            ? (category as MemoryCategory)
            : "other") as MemoryCategory,
          createdAt: typeof metadataField(meta, "createdAt") === "number"
            ? (metadataField(meta, "createdAt") as number)
            : createdAtMs,
        },
        score: clamp01(similarity),
      },
      metadata: meta,
    });
  }
  return out;
}

function parseMetadataValue(value: unknown): Record<string, unknown> | null {
  if (value == null || value === "") return null;
  if (typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // fall through
    }
  }
  return null;
}

function parseTimestampMs(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (value instanceof Date) {
    return value.getTime();
  }
  if (typeof value === "string" && value.length > 0) {
    const candidate = value.includes("T") ? value : value.replace(" ", "T");
    const d = new Date(candidate);
    const t = d.getTime();
    if (!Number.isNaN(t)) return t;
  }
  return 0;
}

function toFiniteNumber(value: unknown): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : 0;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

function isMissingNamespaceError(err: unknown): boolean {
  const e = err as { code?: string; message?: string };
  if (e?.code === "NOT_FOUND") return true;
  const msg = String(e?.message ?? "").toLowerCase();
  return (
    msg.includes("does not exist") ||
    msg.includes("no such table") ||
    msg.includes("unknown namespace") ||
    msg.includes("namespace not found")
  );
}

/**
 * A record the client-side WHERE predicate evaluates against — the mapped
 * recall result plus its parsed `metadata` blob.
 */
type WhereRecord = {
  entry: MemoryEntry;
  score: number;
  /** Evidence strength, when the backend reports one (memory-object only). */
  confidence?: number;
  metadata: Record<string, unknown> | null;
};

/**
 * WHERE fields the memory-object backend can actually supply.
 *
 * The object backend stores no free-form metadata (REMEMBER has no `metadata`
 * option — verified against the engine), so `category` / `importance` /
 * `createdAt` / `metadata->>'…'` have no source there. `recallFiltered` uses
 * this set to fail loudly instead of returning a confidently empty result set.
 */
const MEMORY_OBJECT_WHERE_FIELDS = new Set([
  "id",
  "text",
  "content",
  "similarity",
  "score",
  "confidence",
]);

/**
 * Compile a SQL-ish `WHERE` fragment into a JS predicate over a
 * {@link WhereRecord}. This exists because the engine cannot apply a `WHERE`
 * to the table-valued `MEMORY_RECALL(...)` output (see the ENGINE BUG note on
 * `searchFiltered`), so `searchFiltered` fetches unfiltered rows and filters
 * them here.
 *
 * Supported surface (the plugin's documented filtering contract):
 *   - Fields: `category`, `importance`, `createdAt`, `text`/`content`, `id`,
 *     `similarity`/`score`, and JSON-extract `metadata->>'key'`.
 *   - Comparison ops: `=`/`==`, `!=`/`<>`, `>`, `>=`, `<`, `<=`, `LIKE`
 *     (SQL `%`/`_` wildcards), `IN (...)`.
 *   - Boolean combinators: `AND`, `OR`, `NOT`, and parentheses.
 *   - Literals: single-quoted strings, numbers, `TRUE`/`FALSE`/`NULL`.
 *
 * A trivial always-true clause (empty or `1=1`) short-circuits to pass-all.
 * Unsupported syntax throws a descriptive error rather than silently
 * returning wrong rows.
 *
 * Returns the compiled predicate AND the set of fields it reads, so a caller
 * can refuse a clause the active backend cannot answer (see
 * {@link MEMORY_OBJECT_WHERE_FIELDS}) instead of evaluating it against
 * undefined and returning nothing.
 */
function compileWherePredicate(where: string): {
  predicate: (rec: WhereRecord) => boolean;
  fields: string[];
} {
  const src = (where ?? "").trim();
  if (src === "" || src === "1=1" || src === "1 = 1" || src.toLowerCase() === "true") {
    return { predicate: () => true, fields: [] };
  }

  type Tok = { t: string; v: string };
  const tokens: Tok[] = [];
  const re =
    /\s+|metadata\s*->>\s*'([^']*)'|'((?:[^']|'')*)'|(>=|<=|!=|<>|==|=|>|<)|(\()|(\))|,|([A-Za-z_][A-Za-z0-9_]*)|(-?\d+(?:\.\d+)?)/g;
  let m: RegExpExecArray | null;
  let lastIndex = 0;
  while ((m = re.exec(src)) !== null) {
    if (m.index !== lastIndex) {
      throw new Error(`recallFiltered: unsupported token near "${src.slice(lastIndex, m.index + 1)}"`);
    }
    lastIndex = re.lastIndex;
    const raw = m[0];
    if (/^\s+$/.test(raw)) continue;
    if (m[1] !== undefined) tokens.push({ t: "field", v: `metadata:${m[1]}` });
    else if (m[2] !== undefined) tokens.push({ t: "str", v: m[2].replace(/''/g, "'") });
    else if (m[3] !== undefined) tokens.push({ t: "op", v: m[3] });
    else if (m[4] !== undefined) tokens.push({ t: "lparen", v: "(" });
    else if (m[5] !== undefined) tokens.push({ t: "rparen", v: ")" });
    else if (raw === ",") tokens.push({ t: "comma", v: "," });
    else if (m[6] !== undefined) {
      const kw = m[6].toUpperCase();
      if (["AND", "OR", "NOT", "LIKE", "IN", "TRUE", "FALSE", "NULL"].includes(kw)) {
        tokens.push({ t: "kw", v: kw });
      } else {
        tokens.push({ t: "field", v: m[6] });
      }
    } else if (m[7] !== undefined) tokens.push({ t: "num", v: m[7] });
  }
  if (lastIndex !== src.length) {
    throw new Error(`recallFiltered: unsupported token near "${src.slice(lastIndex)}"`);
  }

  // Recursive-descent parser -> predicate closure.
  let pos = 0;
  const peek = (): Tok | undefined => tokens[pos];
  const next = (): Tok | undefined => tokens[pos++];
  type Pred = (rec: WhereRecord) => boolean;

  const fieldValue = (name: string, rec: WhereRecord): unknown => {
    if (name.startsWith("metadata:")) {
      const key = name.slice("metadata:".length);
      return rec.metadata ? rec.metadata[key] : undefined;
    }
    switch (name.toLowerCase()) {
      case "category": return rec.entry.category;
      case "importance": return rec.entry.importance;
      case "createdat": return rec.entry.createdAt;
      case "text":
      case "content": return rec.entry.text;
      case "id": return rec.entry.id;
      case "similarity":
      case "score": return rec.score;
      case "confidence": return rec.confidence;
      default:
        return rec.metadata ? rec.metadata[name] : undefined;
    }
  };

  const asNumber = (v: unknown): number | null => {
    if (typeof v === "number") return Number.isFinite(v) ? v : null;
    if (typeof v === "string" && v.trim() !== "") {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  };

  const likeToRegExp = (pattern: string): RegExp => {
    let out = "";
    for (const ch of pattern) {
      if (ch === "%") out += ".*";
      else if (ch === "_") out += ".";
      else out += ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
    return new RegExp(`^${out}$`, "i");
  };

  const compareOp = (op: string, left: unknown, right: unknown): boolean => {
    const ln = asNumber(left);
    const rn = asNumber(right);
    const bothNum = ln !== null && rn !== null;
    switch (op) {
      case "=":
      case "==":
        return bothNum ? ln === rn : String(left) === String(right);
      case "!=":
      case "<>":
        return bothNum ? ln !== rn : String(left) !== String(right);
      case ">": return bothNum ? ln > rn : String(left) > String(right);
      case ">=": return bothNum ? ln >= rn : String(left) >= String(right);
      case "<": return bothNum ? ln < rn : String(left) < String(right);
      case "<=": return bothNum ? ln <= rn : String(left) <= String(right);
      default:
        throw new Error(`recallFiltered: unsupported operator "${op}"`);
    }
  };

  // literal value parse (for RHS of a comparison / IN list)
  const parseLiteral = (): unknown => {
    const tk = next();
    if (!tk) throw new Error("recallFiltered: unexpected end of WHERE clause");
    if (tk.t === "str") return tk.v;
    if (tk.t === "num") return Number(tk.v);
    if (tk.t === "kw" && tk.v === "TRUE") return true;
    if (tk.t === "kw" && tk.v === "FALSE") return false;
    if (tk.t === "kw" && tk.v === "NULL") return null;
    throw new Error(`recallFiltered: expected a literal, got "${tk.v}"`);
  };

  const parseComparison = (): Pred => {
    const tk = next();
    if (!tk || tk.t !== "field") {
      throw new Error(`recallFiltered: expected a field name, got "${tk?.v ?? "<end>"}"`);
    }
    const fieldName = tk.v;
    const opTok = peek();
    // IN (...)
    if (opTok && opTok.t === "kw" && opTok.v === "IN") {
      next();
      const lp = next();
      if (!lp || lp.t !== "lparen") throw new Error("recallFiltered: expected '(' after IN");
      const values: unknown[] = [];
      for (;;) {
        values.push(parseLiteral());
        const sep = next();
        if (sep && sep.t === "comma") continue;
        if (sep && sep.t === "rparen") break;
        throw new Error("recallFiltered: malformed IN (...) list");
      }
      return (rec) => {
        const lv = fieldValue(fieldName, rec);
        return values.some((v) => compareOp("=", lv, v));
      };
    }
    // LIKE
    if (opTok && opTok.t === "kw" && opTok.v === "LIKE") {
      next();
      const lit = parseLiteral();
      const rx = likeToRegExp(String(lit));
      return (rec) => rx.test(String(fieldValue(fieldName, rec) ?? ""));
    }
    // comparison operator
    if (!opTok || opTok.t !== "op") {
      throw new Error(`recallFiltered: expected an operator after "${fieldName}", got "${opTok?.v ?? "<end>"}"`);
    }
    next();
    const rhs = parseLiteral();
    return (rec) => compareOp(opTok.v, fieldValue(fieldName, rec), rhs);
  };

  const parsePrimary = (): Pred => {
    const tk = peek();
    if (tk && tk.t === "lparen") {
      next();
      const inner = parseOr();
      const rp = next();
      if (!rp || rp.t !== "rparen") throw new Error("recallFiltered: missing ')'");
      return inner;
    }
    if (tk && tk.t === "kw" && tk.v === "NOT") {
      next();
      const inner = parsePrimary();
      return (rec) => !inner(rec);
    }
    return parseComparison();
  };

  const parseAnd = (): Pred => {
    let left = parsePrimary();
    while (peek() && peek()!.t === "kw" && peek()!.v === "AND") {
      next();
      const right = parsePrimary();
      const l = left;
      left = (rec) => l(rec) && right(rec);
    }
    return left;
  };

  function parseOr(): Pred {
    let left = parseAnd();
    while (peek() && peek()!.t === "kw" && peek()!.v === "OR") {
      next();
      const right = parseAnd();
      const l = left;
      left = (rec) => l(rec) || right(rec);
    }
    return left;
  }

  const predicate = parseOr();
  if (pos !== tokens.length) {
    throw new Error(`recallFiltered: unexpected trailing tokens in WHERE clause near "${peek()?.v ?? ""}"`);
  }
  // Field tokens are collected from the token stream rather than during the
  // parse so the two stay in step no matter which grammar branch consumed them.
  const fields = Array.from(
    new Set(
      tokens
        .filter((t) => t.t === "field")
        .map((t) => (t.v.startsWith("metadata:") ? t.v : t.v.toLowerCase())),
    ),
  );
  return { predicate, fields };
}

// ============================================================================
// CREATE MEMORY backend (SRS v1.2, engine v1.14.3+)
// ============================================================================
//
// The second backend. Where MemoryDB above talks to the flat per-namespace
// `MEMORY_*` primitives, this one talks to the memory OBJECT: one logical
// thing with five real tables behind it, addressed by the five agent-facing
// verbs REMEMBER / RECALL / CURRENT / FORGET / TRACE, every call scoped by an
// identity.
//
// WHY REST AND NOT SQL. The verbs are available as SQL too, but the gateway's
// `/v1/memory/*` module is where identity isolation is enforced
// (`resolve_identity`, MEM-SEC-001): a credential carrying a `memory_identity`
// claim is refused with MEMORY_ACCESS_DENIED before any SQL is composed, and
// every literal — including the free text of a REMEMBER — is escaped
// server-side. Composing `REMEMBER … FOR user_id = '…'` in the plugin would
// move both the escaping and the authorization decision client-side.
//
// The gateway wraps every response in its standard `{data, meta}`; the SDK's
// response interceptor already strips that layer, so `response.data` IS the
// SRS §5.8 envelope: {status, operation, memory, identity, request_id,
// generated_at, data, warnings, error}.

/** The SRS §5.8 response envelope. `data` is verb-specific. */
type MemoryEnvelope = {
  status?: string;
  operation?: string;
  memory?: string;
  identity?: { field?: string; value?: string } | null;
  request_id?: string;
  generated_at?: string;
  data?: Record<string, unknown> | null;
  warnings?: unknown[];
  error?: { code?: string; message?: string; retryable?: boolean } | null;
};

/**
 * Attribute used by the credential-pin probe.
 *
 * No leading underscore on purpose: the engine's statement parser drops leading
 * underscores from an attribute identifier (sending `__openclaw_pin_probe` came
 * back as `openclaw_pin_probe` in the envelope — observed live), so a name that
 * survives the round trip unchanged keeps the probe's log output honest.
 */
const CREDENTIAL_PIN_PROBE_ATTRIBUTE = "openclaw_pin_probe";

/** Minimal shape of the SDK's underlying axios instance that we rely on. */
type HttpLike = {
  get: (path: string) => Promise<{ data: unknown }>;
  post: (path: string, body?: unknown) => Promise<{ data: unknown }>;
  put: (path: string, body?: unknown) => Promise<{ data: unknown }>;
  delete: (path: string) => Promise<{ data: unknown }>;
};

/**
 * The gateway refused this identity.
 *
 * A credential whose token carries a `memory_identity` claim is PINNED: a
 * request naming any other identity is refused with HTTP 403 and
 * `error.code = "MEMORY_ACCESS_DENIED"`. That is an authorization decision
 * about memory, not a bad API key, and it must never be retried against a
 * different identity or blurred into "the gateway is unreachable" — so it gets
 * its own error type and its own message.
 *
 * SDK GAP (@synapcores/sdk@0.6.1): `handleError` maps EVERY 403 onto
 * `AuthenticationError` with `code: "AUTH_ERROR"`, discarding the gateway's
 * own code. The code survives in the response body, which the SDK does pass
 * through as `err.details`, so {@link memoryErrorCode} digs it back out.
 */
export class MemoryAccessDeniedError extends Error {
  readonly code = "MEMORY_ACCESS_DENIED";
  constructor(message: string) {
    super(message);
    this.name = "MemoryAccessDeniedError";
  }
}

/** The active backend cannot serve this operation (and no silent fallback exists). */
export class MemoryFeatureUnavailableError extends Error {
  readonly code = "MEMORY_FEATURE_UNAVAILABLE";
  constructor(message: string) {
    super(message);
    this.name = "MemoryFeatureUnavailableError";
  }
}

/**
 * `memory.identityFrom: "credential"` but the credential carries no pin.
 *
 * The gateway mints `memory_identity` for ordinary interactive logins only.
 * Admins stay unpinned (they must administer other identities) and API keys
 * stay unpinned (one backend credential legitimately serves many end users and
 * supplies the identity per request) — so the two most common OpenClaw
 * credentials, a FullAccess API key and an admin token, both land here. Engines
 * that predate pin minting land here too: the claim is simply absent.
 *
 * The gateway's answer is `400 {"error":{"code":"missing_field","message":
 * "'identity' is required"}}` — verified live. It is NOT a memory error code,
 * which is exactly why it needs translating: left alone it reads as a malformed
 * request rather than "this credential cannot be used in credential mode".
 *
 * This is deliberately fatal rather than a silent fall back to a shared
 * identity: falling back would quietly undo the isolation the mode was chosen
 * for, and writes would land in a scope the operator never asked for.
 */
export class MemoryCredentialUnpinnedError extends Error {
  readonly code = "MEMORY_CREDENTIAL_UNPINNED";
  constructor(message: string) {
    super(message);
    this.name = "MemoryCredentialUnpinnedError";
  }
}

/** The one message every unpinned-credential path emits, so the fix reads the same everywhere. */
function unpinnedCredentialMessage(context: string): string {
  return (
    `memory-synapcores: ${context} sent no identity because memory.identityFrom is "credential" ` +
    "(the default), but the gateway reports this credential carries no memory_identity pin. " +
    "The gateway pins ordinary user logins only — API keys and admin tokens are never pinned, " +
    "and engines older than the pin-minting build never pin anything. Fix it either way: " +
    "configure an ordinary user's token so the gateway resolves the identity itself, or set " +
    'memory.identityFrom to "static" (with memory.identity) or "host".'
  );
}

/**
 * Recover the gateway's stable memory error code from an SDK error.
 *
 * The gateway's error body is `{error: {code, message, fields}, meta}` and the
 * SDK stores the whole body on `err.details` when it has nothing more specific
 * — so `details.error.code` is where MEMORY_ACCESS_DENIED / MEMORY_NOT_FOUND /
 * missing_field actually live. `err.code` is the SDK's own coarse class
 * ("AUTH_ERROR", "NOT_FOUND", …) and is only used as a fallback.
 */
function memoryErrorCode(err: unknown): string | undefined {
  const e = err as { code?: string; details?: { error?: { code?: string } } };
  return e?.details?.error?.code ?? e?.code;
}

/** Single-quote-escape a value for a SQL string literal (engine convention: '' ). */
function sqlLiteral(value: string): string {
  return `'${String(value).replace(/'/g, "''")}'`;
}

/**
 * Where a call's memory identity comes from, resolved once at registration.
 *
 * See MIGRATION.md §3 for the reasoning. The short version: OpenClaw tool
 * `execute()` handlers get no session or user context (the host's signature is
 * `(toolCallId, params, signal?, onUpdate?)`), so deriving a per-user identity
 * in a hook and caching it for a later tool call would be a race — and losing
 * that race means one user reading another user's memory. This resolver
 * therefore never guesses: it returns an identity it can justify, or it throws.
 */
class IdentityResolver {
  constructor(private readonly cfg: MemoryObjectConfig) {}

  get source(): MemoryObjectConfig["identityFrom"] {
    return this.cfg.identityFrom;
  }

  /**
   * Identity for a call with no host context (tools, CLI).
   *
   * `undefined` is a real answer, not a failure: in `credential` mode the
   * plugin deliberately names no identity and lets the gateway resolve the
   * token's pin.
   */
  forTool(operation: string): string | undefined {
    if (this.cfg.identityFrom === "credential") {
      return undefined;
    }
    const identity = this.cfg.identity;
    if (identity) {
      return normalizeIdentity(identity);
    }
    throw new MemoryFeatureUnavailableError(
      `memory-synapcores: ${operation} cannot resolve a memory identity. ` +
        `memory.identityFrom is "${this.cfg.identityFrom}", but OpenClaw tool calls carry no ` +
        "session or user context, so there is nothing to derive one from. Set memory.identity " +
        "to the identity this install should use, or switch memory.identityFrom to " +
        '"credential" and use a token pinned to one identity.',
    );
  }

  /**
   * Identity for a hook-driven call. `ctx` is OpenClaw's
   * PluginHookAgentContext, the only place the host hands a plugin a caller
   * identity. Returns `null` when `host` mode has nothing usable — the caller
   * must SKIP the operation rather than fall back to a shared bucket.
   */
  forHook(ctx: unknown, operation: string): string | undefined | null {
    if (this.cfg.identityFrom !== "host") {
      try {
        return this.forTool(operation);
      } catch {
        return null;
      }
    }
    const c = (ctx ?? {}) as Record<string, unknown>;
    for (const key of ["senderId", "accountId", "sessionKey", "sessionId"]) {
      const raw = c[key];
      if (typeof raw === "string" && raw.trim().length > 0) {
        return normalizeIdentity(raw);
      }
    }
    // Explicitly configured static fallback, if the operator set one.
    return this.cfg.identity ? normalizeIdentity(this.cfg.identity) : null;
  }
}

/**
 * Normalize an identity value. The engine treats it as an opaque string and
 * escapes it itself, so this only trims and bounds it — a 4 KB session key in
 * every row helps nobody.
 */
function normalizeIdentity(value: string): string {
  const trimmed = String(value).trim();
  return trimmed.length > 128 ? trimmed.slice(0, 128) : trimmed;
}

/** Result of `REMEMBER` — one episode plus whatever the engine derived from it. */
export type RememberResult = {
  /**
   * The episode id. REMEMBER also returns `memory_ids` for the derived facts,
   * but the episode is the thing `FORGET … MEMORY '<id>'` and the graph node
   * key have to agree on, so it is the id the plugin reports.
   */
  id: string;
  memoryIds: string[];
  changes: Record<string, number>;
  status: string;
};

/**
 * Thin client for the memory object's REST contract.
 *
 * Every method returns the envelope's `data` (or the envelope itself where the
 * status matters), and lets error envelopes surface as thrown SDK errors that
 * {@link memoryErrorCode} can classify.
 */
class MemoryObjectDB {
  constructor(
    private readonly client: SynapCores,
    private readonly memoryName: string,
  ) {}

  get name(): string {
    return this.memoryName;
  }

  private http(): HttpLike {
    return (
      this.client as unknown as { _getHttpClient: () => HttpLike }
    )._getHttpClient();
  }

  /** `identity: undefined` means "let the gateway resolve the credential's pin". */
  private scope(identity: string | undefined): Record<string, unknown> {
    return identity === undefined ? {} : { identity };
  }

  private async call(
    path: string,
    body: Record<string, unknown>,
    operation: string,
    identity: string | undefined,
  ): Promise<MemoryEnvelope> {
    try {
      const response = await this.http().post(path, body);
      return (response?.data ?? {}) as MemoryEnvelope;
    } catch (err) {
      throw translateMemoryError(err, operation, identity);
    }
  }

  /**
   * `SHOW MEMORIES` — the capability probe.
   *
   * Uses `/memory` WITHOUT a trailing slash: `/v1/memory/` misses the route and
   * lands on the SPA fallback (404), which would read as "feature absent" on an
   * engine that has it.
   */
  async listMemories(): Promise<Array<Record<string, unknown>>> {
    const response = await this.http().get("/memory");
    const body = response?.data;
    // Shape check, not just status check: an older gateway whose SPA fallback
    // answered 200 for an unrouted /v1 path would otherwise look like support.
    return Array.isArray(body) ? (body as Array<Record<string, unknown>>) : [];
  }

  /**
   * `CREATE MEMORY <name> [FOR AGENT <a>] IDENTITY <field> [WITH (...)]`.
   *
   * `create.agent` maps to `FOR AGENT`; `create.options` is the `WITH (...)`
   * map (already engine-keyed). When `create` is omitted or empty the object is
   * created on the engine's own defaults, exactly as before 0.8.0.
   */
  async createMemory(
    identityField: string,
    create?: MemoryCreateRequest,
  ): Promise<MemoryEnvelope> {
    const body: Record<string, unknown> = {
      name: this.memoryName,
      identity_field: identityField,
    };
    if (create?.agent) {
      body.agent = create.agent;
    }
    if (create?.options && Object.keys(create.options).length > 0) {
      body.options = create.options;
    }
    const response = await this.http().post("/memory", body);
    return (response?.data ?? {}) as MemoryEnvelope;
  }

  /** `DESCRIBE MEMORY <name>` — the object's definition + option values. */
  async describe(): Promise<MemoryEnvelope> {
    try {
      const response = await this.http().get(`/memory/${this.memoryName}`);
      return (response?.data ?? {}) as MemoryEnvelope;
    } catch (err) {
      throw translateMemoryError(err, "DESCRIBE", undefined);
    }
  }

  /**
   * `GET /:name/status` — operational health/stats for the object.
   *
   * This replaces the pre-0.8.0 habit of faking stats with a raw `COUNT(*)` on
   * the internal `_mem_<name>_episodes` table, which returned the whole
   * tenant-visible count (wrong) under a credential-pinned identity.
   */
  async status(): Promise<MemoryEnvelope> {
    try {
      const response = await this.http().get(`/memory/${this.memoryName}/status`);
      return (response?.data ?? {}) as MemoryEnvelope;
    } catch (err) {
      throw translateMemoryError(err, "STATUS", undefined);
    }
  }

  /** `HISTORY` — the temporal timeline of how an attribute's value changed. */
  async history(
    identity: string | undefined,
    attribute: string | undefined,
    limit?: number,
  ): Promise<MemoryEnvelope> {
    const options: Record<string, unknown> = {};
    if (typeof limit === "number" && Number.isFinite(limit)) {
      options.limit = limit;
    }
    const body: Record<string, unknown> = { ...this.scope(identity), options };
    if (attribute) {
      body.attribute = attribute;
    }
    return this.call(`/memory/${this.memoryName}/history`, body, "HISTORY", identity);
  }

  /** `SEARCH` — raw semantic search over the object (vs RECALL's distilled view). */
  async search(
    identity: string | undefined,
    query: string,
    options: Record<string, unknown> = {},
  ): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/search`,
      { ...this.scope(identity), query, options },
      "SEARCH",
      identity,
    );
  }

  /** `SUPERSEDE` — explicitly set an attribute's current value, superseding prior claims. */
  async supersede(
    identity: string | undefined,
    attribute: string,
    value: string,
    options: Record<string, unknown> = {},
  ): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/supersede`,
      { ...this.scope(identity), attribute, value, options },
      "SUPERSEDE",
      identity,
    );
  }

  /** `RECONCILE` — re-run conflict resolution across (optionally one) attribute. */
  async reconcile(
    identity: string | undefined,
    attribute?: string,
  ): Promise<MemoryEnvelope> {
    const body: Record<string, unknown> = { ...this.scope(identity), options: {} };
    if (attribute) {
      body.attribute = attribute;
    }
    return this.call(`/memory/${this.memoryName}/reconcile`, body, "RECONCILE", identity);
  }

  /** `EXPLAIN MEMORY <name> ABOUT <q>` — how the engine would answer a recall. */
  async explain(identity: string | undefined, about: string): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/explain`,
      { ...this.scope(identity), about },
      "EXPLAIN",
      identity,
    );
  }

  /** `ALTER MEMORY <name> SET (...)` — change option values in place. */
  async alter(options: Record<string, unknown>): Promise<MemoryEnvelope> {
    try {
      const response = await this.http().put(`/memory/${this.memoryName}`, { options });
      return (response?.data ?? {}) as MemoryEnvelope;
    } catch (err) {
      throw translateMemoryError(err, "ALTER", undefined);
    }
  }

  /** `DROP MEMORY [IF EXISTS] <name>` — remove the object (optionally its stored memories). */
  async drop(options: { cascade?: boolean; ifExists?: boolean } = {}): Promise<MemoryEnvelope> {
    const query: string[] = [];
    if (options.cascade) query.push("cascade=true");
    if (options.ifExists) query.push("if_exists=true");
    const suffix = query.length > 0 ? `?${query.join("&")}` : "";
    try {
      const response = await this.http().delete(`/memory/${this.memoryName}${suffix}`);
      return (response?.data ?? {}) as MemoryEnvelope;
    } catch (err) {
      throw translateMemoryError(err, "DROP", undefined);
    }
  }

  async remember(
    identity: string | undefined,
    content: string,
    options: Record<string, unknown> = {},
  ): Promise<RememberResult> {
    const envelope = await this.call(
      `/memory/${this.memoryName}/remember`,
      { ...this.scope(identity), content, options },
      "REMEMBER",
      identity,
    );
    const data = (envelope.data ?? {}) as Record<string, unknown>;
    return {
      id: typeof data.episode_id === "string" ? data.episode_id : "",
      memoryIds: Array.isArray(data.memory_ids) ? (data.memory_ids as string[]) : [],
      changes: (data.changes ?? {}) as Record<string, number>,
      status: envelope.status ?? "accepted",
    };
  }

  async recall(
    identity: string | undefined,
    about: string,
    options: Record<string, unknown> = {},
  ): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/recall`,
      { ...this.scope(identity), about, options },
      "RECALL",
      identity,
    );
  }

  async current(identity: string | undefined, attribute: string): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/current`,
      { ...this.scope(identity), attribute },
      "CURRENT",
      identity,
    );
  }

  async trace(identity: string | undefined, attribute: string): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/trace`,
      { ...this.scope(identity), attribute },
      "TRACE",
      identity,
    );
  }

  /**
   * `FORGET` — explicit, authorized removal. Exactly one target; the gateway
   * refuses an ambiguous FORGET rather than resolving it by precedence.
   */
  async forget(
    identity: string | undefined,
    target: { memoryId?: string; about?: string; attribute?: string },
  ): Promise<MemoryEnvelope> {
    const body: Record<string, unknown> = { ...this.scope(identity) };
    if (target.memoryId) body.memory_id = target.memoryId;
    else if (target.about) body.about = target.about;
    else if (target.attribute) body.attribute = target.attribute;
    return this.call(`/memory/${this.memoryName}/forget`, body, "FORGET", identity);
  }

  /**
   * `CONSOLIDATE` — optional control, never a prerequisite.
   *
   * Deliberately NOT a model-facing tool. The engine runs consolidation on its
   * own schedule and current state is already correct the moment REMEMBER
   * returns; the manual is explicit that reaching for CONSOLIDATE to make an
   * answer correct is a bug worth reporting. This exists for tests, for a
   * bulk import, and for "run a report right now".
   */
  async consolidate(identity: string | undefined): Promise<MemoryEnvelope> {
    return this.call(
      `/memory/${this.memoryName}/consolidate`,
      this.scope(identity),
      "CONSOLIDATE",
      identity,
    );
  }

  /**
   * Count this identity's episodes.
   *
   * There is no COUNT verb; the memory object materializes five ORDINARY
   * tables (`_mem_<name>_{episodes,facts,state,prov,rels}`) that the manual
   * documents as directly SELECT-able, so this reads the episodes table. With
   * a credential-pinned identity (identity unknown client-side) it counts the
   * whole tenant-visible object instead of one identity.
   */
  async count(identity: string | undefined): Promise<number> {
    const where = identity === undefined ? "" : ` WHERE identity = ${sqlLiteral(identity)}`;
    try {
      const result = await this.client.executeQuery({
        sql: `SELECT COUNT(*) FROM _mem_${this.memoryName}_episodes${where}`,
      });
      const first = (result.rows as unknown[][] | undefined)?.[0];
      const raw = Array.isArray(first) ? first[0] : undefined;
      if (typeof raw === "number") return raw;
      if (typeof raw === "string") {
        const n = Number(raw);
        return Number.isFinite(n) ? n : 0;
      }
      return 0;
    } catch {
      // Object exists but has never been written to — no rows, no table yet.
      return 0;
    }
  }

  /**
   * Fetch one episode by id, for callers that hydrate a memory from an id
   * (`trainRelevanceModel`). Same reasoning as `count`: the episodes table is
   * public. `importance` / `category` come back at their defaults because the
   * memory object stores no free-form metadata (see MIGRATION.md §6).
   */
  async get(id: string, identity: string | undefined): Promise<MemoryEntry | null> {
    if (typeof id !== "string" || id.length === 0) return null;
    const scope = identity === undefined ? "" : ` AND identity = ${sqlLiteral(identity)}`;
    try {
      const result = await this.client.executeQuery({
        sql:
          `SELECT id, content, recorded_at FROM _mem_${this.memoryName}_episodes ` +
          `WHERE id = ${sqlLiteral(id)}${scope} LIMIT 1`,
      });
      const row = (result.rows as unknown[][] | undefined)?.[0];
      if (!Array.isArray(row)) return null;
      return {
        id: String(row[0] ?? id),
        text: String(row[1] ?? ""),
        vector: [],
        importance: 0,
        category: "other",
        createdAt: parseTimestampMs(row[2]),
      };
    } catch {
      return null;
    }
  }
}

/**
 * Turn an SDK transport error into something a caller can branch on.
 *
 * The one case that must not be generic is the identity refusal: a pinned
 * credential naming another identity is a 403 the SDK flattens into
 * `AuthenticationError`, which is otherwise indistinguishable from a bad API
 * key. `missing_field` gets the same treatment for the opposite reason — it is
 * what an UNPINNED credential gets back in `credential` mode, and the fix is a
 * config change, not a retry.
 */
function translateMemoryError(
  err: unknown,
  operation: string,
  identity: string | undefined,
): unknown {
  const code = memoryErrorCode(err);
  if (code === "MEMORY_ACCESS_DENIED") {
    return new MemoryAccessDeniedError(
      `memory-synapcores: the gateway refused ${operation} for memory identity ` +
        `${identity === undefined ? "(credential pin)" : `'${identity}'`}: this credential is ` +
        "pinned to a different identity (MEMORY_ACCESS_DENIED). Either use a token pinned to " +
        'this identity, or set memory.identityFrom: "credential" so the gateway resolves the ' +
        "pin itself instead of the plugin naming one.",
    );
  }
  if (code === "missing_field" && identity === undefined) {
    return new MemoryCredentialUnpinnedError(unpinnedCredentialMessage(operation));
  }
  return err;
}

/**
 * Map a RECALL envelope onto the plugin's flat `MemorySearchResult[]`.
 *
 * Semantic facts come first, then episodes: the facts are the engine's
 * distilled claims and the episodes are the verbatim statements the legacy
 * backend used to return, so a caller that only reads the first N still gets
 * the more useful half. `relevance` -> `score`, `confidence` stays its own
 * field — the two are never merged.
 *
 * `current_state` is DELIBERATELY not folded in here. It is authoritative and
 * the whole point of the envelope's shape is that a model does not have to
 * guess which similar-looking string is the current one; callers that want it
 * read `data.current_state` (or `memory_current`) directly.
 */
function recallEnvelopeToResults(envelope: MemoryEnvelope): MemorySearchResult[] {
  const data = (envelope.data ?? {}) as Record<string, unknown>;
  const out: MemorySearchResult[] = [];
  const seen = new Set<string>();

  const push = (
    id: string,
    text: string,
    relevance: unknown,
    confidence: unknown,
    when: unknown,
  ): void => {
    if (!text || seen.has(id)) return;
    seen.add(id);
    out.push({
      entry: {
        id,
        text,
        vector: [],
        // The memory object stores no free-form metadata, so these have no
        // source on this backend — see MIGRATION.md §6.
        importance: 0,
        category: "other",
        createdAt: parseTimestampMs(when),
      },
      score: clamp01(toFiniteNumber(relevance)),
      ...(typeof confidence === "number" ? { confidence: clamp01(confidence) } : {}),
    });
  };

  for (const raw of Array.isArray(data.semantic) ? data.semantic : []) {
    const f = (raw ?? {}) as Record<string, unknown>;
    push(
      String(f.memory_id ?? ""),
      String(f.fact ?? ""),
      f.relevance,
      f.confidence,
      f.valid_from,
    );
  }
  for (const raw of Array.isArray(data.episodic) ? data.episodic : []) {
    const e = (raw ?? {}) as Record<string, unknown>;
    push(
      String(e.episode_id ?? ""),
      String(e.event ?? ""),
      e.relevance,
      e.confidence,
      e.occurred_at ?? e.recorded_at,
    );
  }
  return out.sort((a, b) => b.score - a.score);
}

/**
 * Which backend is live, and everything the plugin needs to know about it.
 */
export type MemoryBackendKind = "legacy" | "memory-object";

/**
 * Routes every memory operation to whichever backend this gateway supports.
 *
 * Detection runs ONCE, lazily, on the first memory operation (matching the
 * existing gateway-preflight semantics — nothing hits the network at
 * `register()` time), and the answer is memoized for the life of the plugin.
 */
class MemoryRouter {
  private resolution: Promise<MemoryBackendKind> | null = null;
  private resolvedKind: MemoryBackendKind | null = null;
  private warnedNoIsolation = false;

  constructor(
    private readonly legacy: MemoryDB,
    private readonly objectDb: MemoryObjectDB,
    private readonly cfg: MemoryObjectConfig,
    private readonly identities: IdentityResolver,
    private readonly logger: {
      info?: (msg: string) => void;
      warn?: (msg: string) => void;
      debug?: (msg: string) => void;
    },
  ) {}

  /** The resolved backend, or null before the first operation. */
  get kindIfKnown(): MemoryBackendKind | null {
    return this.resolvedKind;
  }

  get memoryName(): string {
    return this.objectDb.name;
  }

  async kind(): Promise<MemoryBackendKind> {
    if (!this.resolution) {
      this.resolution = this.detect().catch((err) => {
        // Reset so a transient failure (gateway restarting) can be retried,
        // exactly like the preflight promise in MemoryDB. An unpinned
        // credential is the exception: it is a configuration fact, not a
        // transient one, so the verdict sticks rather than re-probing the
        // gateway twice on every subsequent operation.
        if (!(err instanceof MemoryCredentialUnpinnedError)) {
          this.resolution = null;
        }
        throw err;
      });
    }
    return this.resolution;
  }

  private async detect(): Promise<MemoryBackendKind> {
    if (this.cfg.backend === "legacy") {
      return this.settle("legacy", "memory.backend = legacy (configured)");
    }

    const forced = this.cfg.backend === "memory-object";

    // A memory-object name is an IDENTIFIER on the engine side (it becomes part
    // of five physical table names), held to ^[A-Za-z_][A-Za-z0-9_]*$ and 64
    // characters. A longer `collection` is legacy-only.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(this.objectDb.name) || this.objectDb.name.length > 64) {
      const why =
        `memory object name '${this.objectDb.name}' is not a valid engine identifier ` +
        "(must match ^[A-Za-z_][A-Za-z0-9_]* and be at most 64 characters)";
      if (forced) {
        throw new MemoryFeatureUnavailableError(`memory-synapcores: ${why}. Set memory.name.`);
      }
      return this.settle("legacy", why);
    }

    // Preflight first: a down gateway or a rejected key must produce the
    // actionable message, not a raw transport error from the capability probe
    // (and definitely not a silent drop to the legacy backend).
    await this.legacy.ensureReady();

    let objects: Array<Record<string, unknown>>;
    try {
      objects = await this.objectDb.listMemories();
    } catch (err) {
      // An auth or connection failure is NOT a capability signal. Dropping to
      // the legacy backend here would hide a wrong API key behind a backend
      // that fails the same way one call later.
      const kind = classifyGatewayError(err);
      if (kind === "auth" || kind === "connection") {
        throw err;
      }
      if (forced) {
        throw new MemoryFeatureUnavailableError(
          "memory-synapcores: memory.backend = memory-object, but this gateway does not serve " +
            `/v1/memory (${String((err as { message?: string })?.message ?? err)}). ` +
            "The CREATE MEMORY object requires SynapCores engine v1.14.3+.",
        );
      }
      return this.settle(
        "legacy",
        "gateway does not serve /v1/memory (engine predates CREATE MEMORY / v1.14.3)",
      );
    }

    const existing = objects.find((o) => o?.memory_name === this.objectDb.name);
    if (!existing) {
      if (!this.cfg.autoCreate) {
        const why = `memory object '${this.objectDb.name}' does not exist and memory.autoCreate is off`;
        if (forced) throw new MemoryFeatureUnavailableError(`memory-synapcores: ${why}`);
        return this.settle("legacy", why);
      }
      try {
        await this.objectDb.createMemory(this.cfg.identityField, this.cfg.create);
        const withNote =
          Object.keys(this.cfg.create.options).length > 0 || this.cfg.create.agent
            ? ` (${this.cfg.create.agent ? `FOR AGENT ${this.cfg.create.agent}, ` : ""}WITH ${JSON.stringify(this.cfg.create.options)})`
            : "";
        this.logger.info?.(
          `memory-synapcores: created memory object '${this.objectDb.name}' IDENTITY ${this.cfg.identityField}${withNote}`,
        );
      } catch (err) {
        // Another OpenClaw instance won the race — that is success, not failure.
        if (memoryErrorCode(err) !== "MEMORY_ALREADY_EXISTS") {
          const why = `could not create memory object '${this.objectDb.name}': ${String(
            (err as { message?: string })?.message ?? err,
          )}`;
          if (forced) throw new MemoryFeatureUnavailableError(`memory-synapcores: ${why}`);
          return this.settle("legacy", why);
        }
      }
    }

    const identityField = String(existing?.identity_field ?? this.cfg.identityField);

    // In credential mode the plugin names no identity, which only works if the
    // gateway can resolve one from the token's `memory_identity` pin. Find out
    // NOW rather than on the first real memory operation: the answer is a
    // property of the credential, it never changes mid-session, and surfacing
    // it at startup puts it in the log and in `openclaw plugins doctor`
    // instead of in the middle of somebody's conversation.
    if (this.identities.source === "credential") {
      await this.probeCredentialPin();
    }

    return this.settle(
      "memory-object",
      `object '${this.objectDb.name}', IDENTITY ${identityField}, identity source: ${this.identities.source}`,
    );
  }

  /**
   * Ask the gateway whether this credential is pinned, without writing anything.
   *
   * `CURRENT` is the right probe: it is read-only, it goes through the same
   * `resolve_identity` enforcement point as every other verb, and an attribute
   * that was never recorded answers `not_found` rather than erroring — so a
   * pinned credential costs one cheap 200 and an unpinned one produces the
   * `missing_field` we are looking for. (Both outcomes verified against a live
   * gateway.)
   */
  private async probeCredentialPin(): Promise<void> {
    try {
      const envelope = await this.objectDb.current(undefined, CREDENTIAL_PIN_PROBE_ATTRIBUTE);
      const resolved = envelope.identity?.value;
      if (resolved) {
        this.logger.debug?.(
          `memory-synapcores: credential is pinned; the gateway resolves memory identity '${resolved}'`,
        );
      }
    } catch (err) {
      if (err instanceof MemoryCredentialUnpinnedError) {
        throw new MemoryCredentialUnpinnedError(
          unpinnedCredentialMessage("memory backend detection"),
        );
      }
      throw err;
    }
  }

  private settle(kind: MemoryBackendKind, detail: string): MemoryBackendKind {
    this.resolvedKind = kind;
    this.logger.info?.(
      kind === "memory-object"
        ? `memory-synapcores: memory backend = memory-object (${detail})`
        : `memory-synapcores: memory backend = legacy MEMORY_* namespace (${detail})`,
    );
    // Only warn when the operator ASKED for per-identity isolation. Since
    // `credential` became the default, an untouched config on a pre-v1.14.3
    // gateway would otherwise be told it cannot have something it never
    // requested — and a warning everybody sees is a warning nobody reads.
    if (
      kind === "legacy" &&
      this.cfg.identityFrom !== "static" &&
      (this.cfg.identityFromExplicit || this.cfg.requireIsolation)
    ) {
      const message =
        `memory-synapcores: memory.identityFrom = "${this.cfg.identityFrom}" asks for per-identity ` +
        "isolation, but the legacy MEMORY_* backend has no identity dimension — every identity " +
        `shares the namespace '${this.legacy.ns}'. Upgrade the engine to v1.14.3+ for real ` +
        "isolation, or set memory.identityFrom: \"static\" to say so explicitly.";
      if (this.cfg.requireIsolation) {
        throw new MemoryFeatureUnavailableError(
          `${message} (memory.requireIsolation is on, so this is fatal.)`,
        );
      }
      if (!this.warnedNoIsolation) {
        this.warnedNoIsolation = true;
        this.logger.warn?.(message);
      }
    }
    return kind;
  }

  // -- operations -----------------------------------------------------------

  async store(
    entry: Omit<MemoryEntry, "id" | "createdAt">,
    identity: string | undefined,
    sourceType: string,
  ): Promise<{ id: string; createdAt: number; changes?: Record<string, number>; status?: string }> {
    if ((await this.kind()) === "legacy") {
      const stored = await this.legacy.store(entry);
      return { id: stored.id, createdAt: stored.createdAt };
    }
    // No client-side dedup probe: deduplication, supersession and conflict
    // policy are the engine's job on this backend (SRS v1.2 — "the engine owns
    // the lifecycle"), and a second-guessing similarity check here would both
    // cost a round-trip and contradict the engine's supersession logic.
    const result = await this.objectDb.remember(identity, entry.text, {
      source_type: sourceType,
    });
    return {
      id: result.id,
      createdAt: Date.now(),
      changes: result.changes,
      status: result.status,
    };
  }

  async searchByQuery(
    query: string,
    limit: number,
    minScore: number,
    identity: string | undefined,
  ): Promise<MemorySearchResult[]> {
    if ((await this.kind()) === "legacy") {
      return this.legacy.searchByQuery(query, limit, minScore);
    }
    const envelope = await this.objectDb.recall(identity, query, {
      episodes: limit,
      facts: limit,
      prompt_ready: false,
    });
    return recallEnvelopeToResults(envelope)
      .filter((r) => r.score >= minScore)
      .slice(0, limit);
  }

  /**
   * Assemble prompt context.
   *
   * On the memory-object backend this returns the engine's own
   * `context.prompt_ready` projection rather than a hand-assembled join: the
   * engine already knows which claim is authoritative, keeps current state
   * ahead of a semantically-similar older episode, and preserves current state
   * when the token budget truncates. Re-deriving that client-side from a flat
   * result list is exactly what the object backend exists to stop.
   */
  async recallForPrompt(
    query: string,
    identity: string | undefined,
    limit: number,
  ): Promise<{
    promptReady: string | null;
    truncated: boolean;
    results: MemorySearchResult[];
    currentState: Record<string, unknown>;
  }> {
    if ((await this.kind()) === "legacy") {
      const results = await this.legacy.searchByQuery(query, limit, 0.3);
      return { promptReady: null, truncated: false, results, currentState: {} };
    }
    const envelope = await this.objectDb.recall(identity, query, {
      current: true,
      episodes: limit,
      facts: limit,
      prompt_ready: true,
      token_budget: this.cfg.tokenBudget,
    });
    const data = (envelope.data ?? {}) as Record<string, unknown>;
    const context = (data.context ?? {}) as Record<string, unknown>;
    const promptReady =
      typeof context.prompt_ready === "string" && context.prompt_ready.trim().length > 0
        ? context.prompt_ready.trim()
        : null;
    return {
      promptReady,
      truncated: context.truncated === true,
      results: recallEnvelopeToResults(envelope).slice(0, limit),
      currentState: (data.current_state ?? {}) as Record<string, unknown>,
    };
  }

  async deleteById(
    id: string,
    identity: string | undefined,
  ): Promise<{ deleted: boolean; affected?: Record<string, unknown> }> {
    if ((await this.kind()) === "legacy") {
      return { deleted: await this.legacy.delete(id) };
    }
    const envelope = await this.objectDb.forget(identity, { memoryId: id });
    const data = (envelope.data ?? {}) as Record<string, unknown>;
    return {
      deleted: envelope.status !== "not_found",
      affected: (data.affected ?? {}) as Record<string, unknown>,
    };
  }

  /**
   * Forget by free-text scope.
   *
   * Legacy has no such verb, so the plugin's search-then-confirm dance stays
   * there. The object backend has `FORGET … ABOUT`, which resolves the scope
   * semantically, propagates the removal to derived facts / embeddings /
   * relationships / current state / provenance, and reports the count per
   * category — so the caller gets a receipt instead of a guess.
   */
  async forgetAbout(
    scope: string,
    identity: string | undefined,
  ): Promise<{ supported: boolean; affected?: Record<string, unknown>; status?: string }> {
    if ((await this.kind()) === "legacy") {
      return { supported: false };
    }
    const envelope = await this.objectDb.forget(identity, { about: scope });
    const data = (envelope.data ?? {}) as Record<string, unknown>;
    return {
      supported: true,
      affected: (data.affected ?? {}) as Record<string, unknown>,
      status: envelope.status,
    };
  }

  async count(identity: string | undefined): Promise<number> {
    if ((await this.kind()) === "legacy") {
      return this.legacy.count();
    }
    return this.objectDb.count(identity);
  }

  async get(id: string, identity: string | undefined): Promise<MemoryEntry | null> {
    if ((await this.kind()) === "legacy") {
      return this.legacy.get(id);
    }
    return this.objectDb.get(id, identity);
  }

  /** `CURRENT` — deterministic state lookup. Throws on the legacy backend. */
  async current(attribute: string, identity: string | undefined): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("memory_current (CURRENT)");
    return this.objectDb.current(identity, attribute);
  }

  /** `TRACE` — why a value is believed. Throws on the legacy backend. */
  async trace(attribute: string, identity: string | undefined): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("memory_trace (TRACE)");
    return this.objectDb.trace(identity, attribute);
  }

  async consolidate(identity: string | undefined): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("consolidateMemory (CONSOLIDATE)");
    return this.objectDb.consolidate(identity);
  }

  /** `HISTORY` — temporal timeline for an attribute. Throws on the legacy backend. */
  async history(
    attribute: string | undefined,
    identity: string | undefined,
    limit?: number,
  ): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("memory_history (HISTORY)");
    return this.objectDb.history(identity, attribute, limit);
  }

  /** `SEARCH` — raw semantic search. Throws on the legacy backend. */
  async search(
    query: string,
    identity: string | undefined,
    options: Record<string, unknown> = {},
  ): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("SEARCH");
    return this.objectDb.search(identity, query, options);
  }

  /** `SUPERSEDE` — set an attribute's current value. Throws on the legacy backend. */
  async supersede(
    attribute: string,
    value: string,
    identity: string | undefined,
    options: Record<string, unknown> = {},
  ): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("SUPERSEDE");
    return this.objectDb.supersede(identity, attribute, value, options);
  }

  /** `RECONCILE` — re-resolve conflicts. Throws on the legacy backend. */
  async reconcile(attribute: string | undefined, identity: string | undefined): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("RECONCILE");
    return this.objectDb.reconcile(identity, attribute);
  }

  /** `EXPLAIN` — dry-run a recall. Throws on the legacy backend. */
  async explain(about: string, identity: string | undefined): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("EXPLAIN");
    return this.objectDb.explain(identity, about);
  }

  /** `DESCRIBE MEMORY` / `GET /:name/status` — definition + stats. Throws on legacy. */
  async describe(): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("DESCRIBE");
    return this.objectDb.describe();
  }

  async status(): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("status");
    return this.objectDb.status();
  }

  /** `ALTER MEMORY ... SET (...)`. Throws on the legacy backend. */
  async alter(options: Record<string, unknown>): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("ALTER MEMORY");
    return this.objectDb.alter(options);
  }

  /** `DROP MEMORY`. Throws on the legacy backend. */
  async drop(options: { cascade?: boolean; ifExists?: boolean } = {}): Promise<MemoryEnvelope> {
    await this.requireObjectBackend("DROP MEMORY");
    return this.objectDb.drop(options);
  }

  private async requireObjectBackend(operation: string): Promise<void> {
    if ((await this.kind()) !== "memory-object") {
      throw new MemoryFeatureUnavailableError(
        `memory-synapcores: ${operation} requires the CREATE MEMORY backend ` +
          "(SynapCores engine v1.14.3+). This gateway is on the legacy MEMORY_* path.",
      );
    }
  }
}

// ============================================================================
// Embeddings (engine-native)
// ============================================================================
//
// OpenAI has been fully removed. Client-side embeddings are now produced by
// the SynapCores engine via `client.embed()` (native `EMBED()`), the same
// embedding space the engine uses for the core memory ops
// (`MEMORY_STORE` / `MEMORY_RECALL`). Used by:
//   - `predictRelevance` / `trainRelevanceModel`: client-side cosine
//     between query and candidate text.
//   - `autoLinkSimilar` capture: writing an `embedding` property onto the
//     Memory graph node so `recallRelated`'s synthetic SIMILAR_TO edge
//     resolves at MATCH time.

class Embeddings {
  constructor(private client: SynapCores) {}

  async embed(text: string): Promise<number[]> {
    return (await this.client.embed(text)) as number[];
  }
}

// ============================================================================
// Math helpers (shared between linker + relevance scorer)
// ============================================================================

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function ageDays(createdAt: number, now: number = Date.now()): number {
  return Math.max(0, (now - createdAt) / (1000 * 60 * 60 * 24));
}

const CATEGORY_INDEX: Record<MemoryCategory, number> = {
  preference: 0,
  fact: 1,
  decision: 2,
  entity: 3,
  other: 4,
};

function categoryOneHot(category: MemoryCategory): number[] {
  const vec = [0, 0, 0, 0, 0];
  vec[CATEGORY_INDEX[category] ?? 4] = 1;
  return vec;
}

/**
 * Feature vector used by both the heuristic relevance scorer and the
 * AutoML training path. Keeping the layout in one place means
 * `predictRelevance` and `trainRelevanceModel` always see the same shape.
 */
function buildRelevanceFeatures(
  queryVector: number[],
  candidate: MemoryEntry,
  now: number = Date.now(),
): {
  cosine: number;
  ageDays: number;
  importance: number;
  category: MemoryCategory;
  vector: number[];
  asRecord: Record<string, number>;
} {
  const cosine = cosineSimilarity(queryVector, candidate.vector);
  const age = ageDays(candidate.createdAt, now);
  const oneHot = categoryOneHot(candidate.category);
  return {
    cosine,
    ageDays: age,
    importance: candidate.importance,
    category: candidate.category,
    vector: [cosine, age, candidate.importance, ...oneHot],
    asRecord: {
      cosine,
      age_days: age,
      importance: candidate.importance,
      category_preference: oneHot[0],
      category_fact: oneHot[1],
      category_decision: oneHot[2],
      category_entity: oneHot[3],
      category_other: oneHot[4],
    },
  };
}

// ============================================================================
// Rule-based capture filter
// ============================================================================

const MEMORY_TRIGGERS = [
  /zapamatuj si|pamatuj|remember/i,
  /preferuji|radši|nechci|prefer/i,
  /rozhodli jsme|budeme používat/i,
  /\+\d{10,}/,
  /[\w.-]+@[\w.-]+\.\w+/,
  /můj\s+\w+\s+je|je\s+můj/i,
  // "my favorite programming language is Rust" — allow any number of words
  // between "my" and "is/are", not just a single token.
  /\bmy\s+(?:\w+\s+){0,5}(?:is|are|was|were)\b|\bis\s+my\b/i,
  /i\s?'?\s?(?:like|prefer|hate|love|want|need|enjoy|dislike|avoid)/i,
  // constraints/restrictions: allergic to, can't eat, don't eat, no dairy, etc.
  /allerg(?:y|ic)|intoleran(?:ce|t)|can'?t (?:eat|have|drink)|don'?t (?:eat|drink)/i,
  // identity/context facts — deliberately specific, not a bare "i'm \w+"
  // (which would match almost every sentence and flood memory with junk)
  /my name is|call me|i work (?:at|for|as)|i live in|i'?m based in|i was born|i'?m (?:a|an)\s+\w+/i,
  // decisions/commitments
  /decided (?:to|that)|will use|we'?ll (?:use|go with|switch to)|going forward/i,
  // explicit remember requests
  /remember that|note that|for future reference|keep in mind|fyi\b/i,
  /always|never|important/i,
];

/** One extracted conversation turn: the text plus who said it. */
export type MessageTurn = { role: "user" | "assistant"; text: string };

/**
 * Extracts plain-text user/assistant turns from a session's `unknown[]`
 * message log, KEEPING the role.
 *
 * The role used to be discarded. It matters now: on the memory-object backend
 * it becomes the claim's `source_type`, which is the first dimension the
 * engine's conflict resolution considers — a user's statement should outrank
 * the assistant's paraphrase of it when the two disagree.
 */
function extractMessageTurns(messages: unknown[]): MessageTurn[] {
  const turns: MessageTurn[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const msgObj = msg as Record<string, unknown>;

    const role = msgObj.role;
    if (role !== "user" && role !== "assistant") {
      continue;
    }

    const content = msgObj.content;

    if (typeof content === "string") {
      turns.push({ role, text: content });
      continue;
    }

    if (Array.isArray(content)) {
      for (const block of content) {
        if (
          block &&
          typeof block === "object" &&
          "type" in block &&
          (block as Record<string, unknown>).type === "text" &&
          "text" in block &&
          typeof (block as Record<string, unknown>).text === "string"
        ) {
          turns.push({ role, text: (block as Record<string, unknown>).text as string });
        }
      }
    }
  }
  return turns;
}

export function shouldCapture(text: string): boolean {
  if (text.length < 10 || text.length > 500) {
    return false;
  }
  // Skip injected context from memory recall
  if (text.includes("<relevant-memories>")) {
    return false;
  }
  // Skip system-generated content
  if (text.startsWith("<") && text.includes("</")) {
    return false;
  }
  // Skip agent summary responses (contain markdown formatting)
  if (text.includes("**") && text.includes("\n-")) {
    return false;
  }
  // Skip emoji-heavy responses (likely agent output)
  const emojiCount = (text.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
  if (emojiCount > 3) {
    return false;
  }
  return MEMORY_TRIGGERS.some((r) => r.test(text));
}

// ============================================================================
// Memory capability: pre-compaction flush
//
// Mirrors OpenClaw's own bundled `memory-core` plugin, which does not rely on
// per-turn regex scanning at all — it registers a `flushPlanResolver` that the
// core runtime calls right before a session auto-compacts, spinning up a
// dedicated "flush" turn that asks the agent itself to write down whatever's
// worth keeping, in its own words, before older context gets discarded.
//
// memory-core's version tells the agent to append to a dated markdown file
// (it owns file-backed storage). We don't — SynapCores is the store — so our
// flush prompt tells the agent to call `memory_store` instead. `relativePath`
// is a required field of MemoryFlushPlan but is only ever used by memory-core
// to template its own file-path prompt text; we don't write files, so it's a
// label only, not a real path.
// ============================================================================

const MEMORY_FLUSH_SYSTEM_PROMPT = [
  "Pre-compaction memory flush turn.",
  "This session is about to auto-compact and older context will be discarded.",
  "Review the conversation and call the memory_store tool once per distinct " +
    "durable fact, preference, decision, or constraint worth keeping — in your " +
    "own words, not a verbatim transcript quote.",
  "Do not call memory_store for anything already covered by an existing memory; " +
    "use memory_recall first if you're unsure whether something is already stored.",
  "Do not answer the user during this turn.",
  `If there is nothing worth storing, reply with exactly ${SILENT_REPLY_TOKEN} and call no tools.`,
].join(" ");

const MEMORY_FLUSH_PROMPT = [
  "Pre-compaction memory flush.",
  "Store any durable facts, preferences, decisions, or constraints from this " +
    "conversation via memory_store, one call per fact.",
  `If nothing is worth storing, reply with exactly ${SILENT_REPLY_TOKEN}.`,
].join(" ");

function buildMemoryFlushPlan() {
  return {
    softThresholdTokens: 4000,
    forceFlushTranscriptBytes: 2 * 1024 * 1024,
    reserveTokensFloor: 20000,
    prompt: MEMORY_FLUSH_PROMPT,
    systemPrompt: MEMORY_FLUSH_SYSTEM_PROMPT,
    relativePath: "synapcores-memory-flush",
  };
}

function detectCategory(text: string): MemoryCategory {
  const lower = text.toLowerCase();
  if (/prefer|radši|like|love|hate|want/i.test(lower)) {
    return "preference";
  }
  if (/rozhodli|decided|will use|budeme/i.test(lower)) {
    return "decision";
  }
  if (/\+\d{10,}|@[\w.-]+\.\w+|is called|jmenuje se/i.test(lower)) {
    return "entity";
  }
  if (/is|are|has|have|je|má|jsou/i.test(lower)) {
    return "fact";
  }
  return "other";
}

// ============================================================================
// Cypher value escaping
// ============================================================================

/**
 * Escape a string literal for safe inlining into a Cypher query.
 *
 * Gateway v1.6.5.2+ rejects named-parameter bindings (`$param`) with HTTP
 * 400; the supported path is to inline literal values into the query
 * string. Memory IDs flow in from `MEMORY_STORE` (engine-generated, e.g.
 * `mem_1kv69sxfn_5ofzwK`) so they are normally safe, BUT we never trust
 * upstream input — every string that ends up inside `'...'` in a Cypher
 * fragment must go through this helper.
 *
 * Escapes single quotes (`'` -> `\'`) and backslashes (`\` -> `\\`).
 * Returns the inner content only; callers are responsible for the
 * surrounding quotes (so the helper is composable with template literals).
 */
export function escapeCypherString(value: string): string {
  return String(value).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

// ============================================================================
// Graph wiring — populate Memory nodes so SIMILAR_TO resolves at MATCH time
// ============================================================================

// Default cosine-similarity threshold for synthetic SIMILAR_TO edges.
// Used as the operand in `[:SIMILAR_TO > THRESHOLD]` Cypher fragments.
const SIMILAR_TO_THRESHOLD = 0.5;
// Max hops for recallRelated walks — caps runaway traversals.
const MAX_HOPS = 4;
// Default result cap for recallRelated.
const DEFAULT_RECALL_RELATED_LIMIT = 20;

/**
 * `linkSimilarMemories` — capture-time graph wiring.
 *
 * Inserts the Memory as a graph node carrying the engine-native embedding
 * under the property name `embedding` — the field the gateway's brute-force
 * vector index is wired against (see
 * `aidb_gateway::routes::graph::attach_default_vector_index`). Once a
 * Memory node exists, `recallRelated` can MATCH `[:SIMILAR_TO > T]`
 * against it and get neighbors back without any pre-stored edges.
 *
 * NOTE: the engine no longer exposes the vector it used internally for
 * `MEMORY_STORE`, so this helper re-embeds the text via `client.embed()`.
 * Because the node-property vector now comes from the engine's own
 * embedding space (previously it was a separate OpenAI space), it is
 * consistent with the vectors the engine produces for the core memory
 * ops — a consistency improvement over the prior OpenAI-based path.
 *
 * Failures here are non-fatal — the capture itself still succeeded in the
 * memory subsystem; we just log and move on so the rest of recall keeps
 * working.
 */
async function linkSimilarMemories(
  entry: MemoryEntry,
  client: SynapCores,
  embedVector: number[],
  logger: { warn?: (msg: string) => void } | undefined,
): Promise<number> {
  try {
    // KNOWN SDK GAP (@synapcores/sdk@<=0.5.0):
    //   `client.graph.nodes.create(label, props)` posts
    //   `{label: <single>, properties}` but the gateway's
    //   /v1/graph/nodes handler expects `{labels: <array>, properties}`
    //   (see aidb_gateway::routes::graph::CreateNodeRequest). The result
    //   is a node with `labels: []`, which never matches the `Memory`
    //   label filter in MATCH. We bypass the SDK helper for THIS one
    //   call and post the correct wire shape ourselves. Once the SDK
    //   fixes `GraphNodeApi.create` to send `labels`, this
    //   `_getHttpClient` call can be replaced with
    //   `client.graph.nodes.create(...)`.
    const http = (
      client as unknown as {
        _getHttpClient: () => { post: (p: string, b: unknown) => Promise<unknown> };
      }
    )._getHttpClient();
    await http.post("/graph/nodes", {
      labels: ["Memory"],
      properties: {
        id: entry.id,
        text: entry.text,
        embedding: embedVector,
        importance: entry.importance,
        category: entry.category,
        createdAt: entry.createdAt,
      },
    });
    return 1;
  } catch (err) {
    const msg = (err as { message?: string })?.message ?? String(err);
    logger?.warn?.(
      `memory-synapcores: failed to insert Memory graph node for ${entry.id}: ${msg} (capture still succeeded; recallRelated for this entry will return [])`,
    );
    return 0;
  }
}

// ============================================================================
// AutoML helpers — staged-collection training
// ============================================================================

const DEFAULT_RELEVANCE_MODEL = "openclaw_memory_relevance";
const MIN_TRAINING_SAMPLES = 10;
const TRAINING_TABLE_PREFIX = "openclaw_memory_relevance_training";

function relevanceModelName(workspace?: string): string {
  return workspace ? `${DEFAULT_RELEVANCE_MODEL}_${workspace}` : DEFAULT_RELEVANCE_MODEL;
}

function trainingTableName(workspace?: string): string {
  // SQL identifier — caller-controlled workspace must be alphanumeric.
  // Gateway rejects non-alphanumeric `collection` values anyway, so the
  // sanitisation here is belt-and-braces.
  const safe = workspace ? workspace.replace(/[^a-zA-Z0-9_]/g, "_") : "";
  return safe ? `${TRAINING_TABLE_PREFIX}_${safe}` : TRAINING_TABLE_PREFIX;
}

// ============================================================================
// SynapCores Extensions
// ============================================================================

/**
 * Build the runtime extension surface exposed on the plugin instance. These
 * methods are exported via `memoryPlugin.extensions` so callers can reach
 * them through the OpenClaw plugin registry.
 *
 * The four SynapCores-only methods use:
 *   - `client.executeQuery(...)` against MEMORY_RECALL for `recallFiltered`,
 *     wrapping the user's WHERE clause around the engine's recall result-set.
 *   - `client.graph.cypher(...)` for `recallRelated`, walking the synthetic
 *     `SIMILAR_TO` edges that resolve against `Memory.embedding`.
 *   - `client.automl.getModel(...).predict(...)` for `predictRelevance`, with
 *     a deterministic heuristic fallback when no model is trained yet.
 *   - `client.executeQuery(...)` + `client.automl.train(...)` for
 *     `trainRelevanceModel`, staging feedback rows in a SQL table the
 *     gateway can `SELECT * FROM` for training.
 */
export interface MemorySynapCoresExtensions {
  /**
   * Vector recall scoped by a SQL `WHERE` clause.
   *
   * @example
   *   await ext.recallFiltered({
   *     where: "category = 'preference' AND importance >= 0.7",
   *     semantic: "what kind of UI does the user prefer?",
   *     limit: 5,
   *   });
   */
  recallFiltered(options: RecallFilteredOptions): Promise<MemorySearchResult[]>;

  /**
   * Graph walk from a memory through `SIMILAR_TO` / `MENTIONS` / `RELATES_TO`
   * edges, returning the neighborhood.
   *
   * Wires against the gateway's synthetic-SIMILAR_TO Cypher syntax
   * (`[:SIMILAR_TO > T]`) which resolves against the `embedding` property
   * the plugin writes onto every captured Memory node. If a source memory
   * has not been promoted to a graph node yet (e.g. `autoLinkSimilar:
   * false`), this returns an empty array.
   *
   * @param memoryId - id of the source memory (engine-assigned, e.g. `mem_...`).
   * @param options - Hops, edge-kind filter, similarity threshold, limit.
   */
  recallRelated(memoryId: string, options?: RecallRelatedOptions): Promise<RelatedMemoryResult[]>;

  /**
   * Score candidate memories against a query.
   *
   * Two execution modes:
   *  - **Model mode**: if an AutoML model named
   *    `openclaw_memory_relevance[_<workspace>]` exists, the candidates'
   *    feature vectors are passed through `model.predict()`.
   *  - **Heuristic mode** (default fallback): `relevance = 0.6 * cosine +
   *    0.25 * exp(-age_days/14) + 0.15 * importance`.
   *
   * Train the model with `trainRelevanceModel(feedback)` to switch to model
   * mode automatically on the next call.
   *
   * @param query - The user query / agent prompt to score relevance against.
   * @param candidates - Memories to score (typically the top-K of a recall).
   * @returns Candidates with a `relevance` score in [0, 1], not re-sorted.
   */
  predictRelevance(
    query: string,
    candidates: MemoryEntry[],
  ): Promise<RelevanceScoredMemory[]>;

  /**
   * Train (or retrain) the AutoML model used by `predictRelevance`.
   *
   * Wires against the staged-collection workflow: feedback rows are
   * inserted into a SQL table the gateway's AutoML can read via
   * `SELECT * FROM {table}`, then `/v1/automl/train` is invoked with
   * `target: 'score'` to fit a regression model. Existing training data
   * in the staging table is preserved across calls so feedback accumulates
   * across sessions.
   *
   * @param feedback - Array of `{ memoryId, queryText, score }` samples.
   *                   Requires at least {@link MIN_TRAINING_SAMPLES} samples;
   *                   throws otherwise.
   * @returns The id and name of the trained model.
   */
  trainRelevanceModel(feedback: RelevanceFeedback[]): Promise<{ modelId: string; modelName: string }>;

  /**
   * Which backend this gateway resolved to — `"memory-object"` (engine
   * v1.14.3+ `CREATE MEMORY`) or `"legacy"` (the flat `MEMORY_*` namespace).
   * Runs capability detection on first call and memoizes the answer.
   */
  backendKind(): Promise<MemoryBackendKind>;

  /**
   * Force a consolidation cycle NOW (memory-object backend only).
   *
   * Optional control, NEVER a prerequisite: the engine schedules its own
   * consolidation and current state is already correct the moment a write
   * returns. This is deliberately NOT exposed as a model-facing tool — memory
   * correctness must not depend on an agent remembering to call it. Use it in
   * a test, before a report, or after a bulk import. Throws
   * {@link MemoryFeatureUnavailableError} on the legacy backend.
   */
  consolidateMemory(): Promise<MemoryEnvelope>;
}

async function ensureCandidateVector(
  candidate: MemoryEntry,
  embeddings: Embeddings,
): Promise<number[]> {
  if (Array.isArray(candidate.vector) && candidate.vector.length > 0) {
    return candidate.vector;
  }
  if (!candidate.text) return [];
  try {
    return await embeddings.embed(candidate.text);
  } catch {
    return [];
  }
}

function createExtensions(
  db: MemoryDB,
  router: MemoryRouter,
  identities: IdentityResolver,
  embeddings: Embeddings,
  client: SynapCores,
  _graphName: string,
  workspace?: string,
): MemorySynapCoresExtensions {
  const modelName = relevanceModelName(workspace);
  const stagingTable = trainingTableName(workspace);

  async function modelExists(name: string): Promise<boolean> {
    try {
      const models = await client.automl.listModels();
      return models.some((m) => m.name === name);
    } catch {
      return false;
    }
  }

  return {
    async backendKind(): Promise<MemoryBackendKind> {
      return router.kind();
    },

    async consolidateMemory(): Promise<MemoryEnvelope> {
      return router.consolidate(identities.forTool("consolidateMemory"));
    },

    async recallFiltered(options: RecallFilteredOptions): Promise<MemorySearchResult[]> {
      const limit = options.limit ?? 5;
      if ((await router.kind()) === "legacy") {
        return db.searchFiltered(options.semantic, options.where, limit);
      }

      // Memory-object backend: there is no metadata blob to filter on, so a
      // predicate naming `category` / `importance` / `createdAt` /
      // `metadata->>'…'` cannot be answered. Returning an empty result set
      // would look like "nothing matched"; throwing says what is actually
      // wrong and names both ways out.
      const { predicate, fields } = compileWherePredicate(options.where);
      const unsupported = fields.filter((f) => !MEMORY_OBJECT_WHERE_FIELDS.has(f));
      if (unsupported.length > 0) {
        throw new MemoryFeatureUnavailableError(
          `memory-synapcores.recallFiltered: the CREATE MEMORY backend stores no per-memory ` +
            `metadata, so it cannot filter on ${unsupported.join(", ")}. Filter on one of ` +
            `${Array.from(MEMORY_OBJECT_WHERE_FIELDS).join(", ")}, or set ` +
            'memory.backend: "legacy" to keep the metadata-filtering surface.',
        );
      }

      const identity = identities.forTool("recallFiltered");
      // Oversample so the client-side post-filter still has rows to return.
      const oversample = clampTopK(Math.max(limit * 5, 25));
      const rows = await router.searchByQuery(options.semantic, oversample, 0, identity);
      return rows
        .filter((r) =>
          predicate({
            entry: r.entry,
            score: r.score,
            confidence: r.confidence,
            metadata: null,
          }),
        )
        .slice(0, Number(limit));
    },

    async recallRelated(
      memoryId: string,
      options: RecallRelatedOptions = {},
    ): Promise<RelatedMemoryResult[]> {
      // The gateway derives `SIMILAR_TO` synthetically at MATCH time from
      // the graph backend's vector index on the `embedding` property.
      // `linkSimilarMemories` populates that index on every capture by
      // posting a `Memory` graph node carrying the embedding. With nodes
      // in place, this method composes a Cypher MATCH that walks
      // SIMILAR_TO (plus any non-synthetic edges the caller named via
      // `edgeKinds`) and returns the neighborhood.
      //
      // Wire constraints:
      //  - Multi-hop `[:SIMILAR_TO*1..N]` is rejected — SIMILAR_TO is
      //    single-hop only. We honour `hops` for non-synthetic edges
      //    and special-case hops=1 for SIMILAR_TO.
      //  - `$param` bindings are rejected — values are inlined and
      //    escaped via `escapeCypherString`.
      const hops = Math.min(Math.max(1, options.hops ?? 1), MAX_HOPS);
      const limit = options.limit ?? DEFAULT_RECALL_RELATED_LIMIT;
      const threshold = options.similarityThreshold ?? SIMILAR_TO_THRESHOLD;
      const edgeKinds = options.edgeKinds && options.edgeKinds.length > 0
        ? options.edgeKinds
        : ["SIMILAR_TO"];

      const id = escapeCypherString(memoryId);
      const out: RelatedMemoryResult[] = [];
      const seen = new Set<string>();

      for (const kind of edgeKinds) {
        const safeKind = String(kind).replace(/[^A-Z_]/g, "");
        if (!safeKind) continue;

        let cypher: string;
        if (safeKind === "SIMILAR_TO") {
          // Synthetic edge — single-hop only, threshold inlined.
          // Use undirected pattern so we surface neighbors regardless
          // of insertion order (the gateway computes similarity
          // symmetrically anyway).
          cypher = `MATCH (start:Memory {id: '${id}'})-[:SIMILAR_TO > ${threshold}]-(related:Memory) RETURN DISTINCT related LIMIT ${limit}`;
        } else {
          // Non-synthetic edge — variable-length supported.
          cypher = `MATCH (start:Memory {id: '${id}'})-[:${safeKind}*1..${hops}]-(related:Memory) RETURN DISTINCT related LIMIT ${limit}`;
        }

        let result: { columns?: string[]; rows?: unknown[][]; records?: Array<Record<string, unknown>> };
        try {
          // SDK 0.5.0 no longer exposes `client.graph.cypher` — the
          // canonical path is `POST /v1/graph/match` with `{sql: <cypher>}`.
          // We reach through the SDK's underlying axios instance via
          // `_getHttpClient()`, the same accessor `linkSimilarMemories`
          // uses for the `/graph/nodes` workaround.
          const http = (
            client as unknown as {
              _getHttpClient: () => {
                post: (p: string, b: unknown) => Promise<{ data: unknown }>;
              };
            }
          )._getHttpClient();
          const response = await http.post("/graph/match", { sql: cypher });
          const body = (response?.data ?? {}) as {
            columns?: string[];
            rows?: unknown[][];
            records?: Array<Record<string, unknown>>;
          };
          result = body;
        } catch (err) {
          const msg = (err as { message?: string })?.message ?? String(err);
          throw new Error(
            `memory-synapcores.recallRelated: graph query failed for edge kind '${safeKind}': ${msg}`,
          );
        }

        // Gateway returns `{ columns, rows, count }`. Older SDK shims
        // also emitted `records` (column-keyed objects); we honour both.
        const records = Array.isArray(result?.records) ? result!.records! : [];
        const cols = Array.isArray(result?.columns) ? result!.columns! : [];
        const rawRows = Array.isArray(result?.rows) ? result!.rows! : [];
        // Synthesize `records` from `(columns, rows)` if not provided.
        const synthRecords: Array<Record<string, unknown>> = records.length > 0
          ? records
          : rawRows
              .filter((r) => Array.isArray(r))
              .map((r) => {
                const obj: Record<string, unknown> = {};
                for (let i = 0; i < cols.length; i++) {
                  obj[cols[i]] = (r as unknown[])[i];
                }
                return obj;
              });
        const rows = rawRows;

        const harvest = (node: unknown): void => {
          if (!node || typeof node !== "object") return;
          const n = node as { id?: unknown; properties?: Record<string, unknown> };
          const props = (n.properties ?? {}) as Record<string, unknown>;
          const entryId = String(props.id ?? n.id ?? "");
          if (!entryId || entryId === memoryId) return;
          if (seen.has(entryId)) return;
          seen.add(entryId);
          out.push({
            entry: {
              id: entryId,
              text: typeof props.text === "string" ? (props.text as string) : "",
              vector: Array.isArray(props.embedding) ? (props.embedding as number[]) : [],
              importance:
                typeof props.importance === "number" ? (props.importance as number) : 0,
              category: (props.category as MemoryCategory) ?? "other",
              createdAt:
                typeof props.createdAt === "number" ? (props.createdAt as number) : 0,
            },
            hops: safeKind === "SIMILAR_TO" ? 1 : hops,
            via: [safeKind],
          });
        };

        for (const rec of synthRecords) {
          // records: { related: <node> }
          harvest((rec as Record<string, unknown>)?.related ?? Object.values(rec ?? {})[0]);
        }
        for (const row of rows) {
          if (!Array.isArray(row)) continue;
          harvest(row[0]);
        }

        if (out.length >= limit) break;
      }

      return out.slice(0, limit);
    },

    async predictRelevance(
      query: string,
      candidates: MemoryEntry[],
    ): Promise<RelevanceScoredMemory[]> {
      if (candidates.length === 0) return [];
      const queryVector = await embeddings.embed(query);
      const now = Date.now();
      // v0.4.0: MEMORY_RECALL doesn't return embeddings, so candidates
      // surfaced via the core recall path arrive with `vector: []`. We
      // lazily embed the candidate text when we need a cosine feature.
      const candidateVectors = await Promise.all(
        candidates.map((c) => ensureCandidateVector(c, embeddings)),
      );
      const features = candidates.map((c, i) =>
        buildRelevanceFeatures(queryVector, { ...c, vector: candidateVectors[i] }, now),
      );

      // Try model mode; on any failure (no model, transport error) fall
      // back to the heuristic so the caller never gets an empty result.
      if (await modelExists(modelName)) {
        try {
          const model = await client.automl.getModel(modelName);
          const inputs = features.map((f) => f.asRecord);
          const raw = await model.predict(inputs);
          const preds = Array.isArray(raw) ? raw : [raw];
          return candidates.map((entry, i) => ({
            entry,
            relevance: clamp01(extractPrediction(preds[i])),
          }));
        } catch {
          // fall through to heuristic
        }
      }

      // Heuristic mode (always available)
      return candidates.map((entry, i) => {
        const f = features[i];
        const recency = Math.exp(-f.ageDays / 14);
        const cosTerm = (f.cosine + 1) / 2; // map [-1, 1] -> [0, 1]
        const score = 0.6 * cosTerm + 0.25 * recency + 0.15 * entry.importance;
        return { entry, relevance: clamp01(score) };
      });
    },

    async trainRelevanceModel(
      feedback: RelevanceFeedback[],
    ): Promise<{ modelId: string; modelName: string }> {
      if (!Array.isArray(feedback) || feedback.length < MIN_TRAINING_SAMPLES) {
        throw new Error(
          `memory-synapcores.trainRelevanceModel: need at least ${MIN_TRAINING_SAMPLES} samples to train a relevance model (got ${
            Array.isArray(feedback) ? feedback.length : 0
          })`,
        );
      }

      // Stage rows in a SQL table the gateway's AutoML can `SELECT * FROM`.
      // The gateway requires the data to land in a real collection/table
      // before training. We:
      //   1. Hydrate each feedback row's memory into a full feature
      //      vector (cosine, age_days, importance, one-hot category) via
      //      the same `buildRelevanceFeatures` helper `predictRelevance`
      //      uses — so train and predict see identical schemas.
      //   2. CREATE TABLE IF NOT EXISTS the staging table.
      //   3. INSERT each row.
      //   4. Call client.automl.train({ collection, target: 'score', ... }).
      //
      // Feedback rows whose memories have been deleted are skipped (with
      // a soft warning to the caller via the throw message); training
      // proceeds with the remainder.

      // Hydrate feedback rows. Embed each query text once; pair with
      // the stored Memory's content (re-embedded via engine-native
      // `client.embed()`) to get the cosine feature.
      const hydrated: Array<{
        cosine: number;
        age_days: number;
        importance: number;
        category_preference: number;
        category_fact: number;
        category_decision: number;
        category_entity: number;
        category_other: number;
        score: number;
      }> = [];
      let missing = 0;
      // NOTE: on the memory-object backend the hydrated entries carry
      // importance 0 / category "other", because that backend stores no
      // per-memory metadata (MIGRATION.md §6). Six of the eight features stay
      // meaningful (cosine, age, and the one-hot that collapses to
      // category_other); training still works, but the model is thinner than
      // the feature list suggests.
      const identity = identities.forTool("trainRelevanceModel");
      for (const fb of feedback) {
        const mem = await router.get(fb.memoryId, identity).catch(() => null);
        if (!mem) {
          missing++;
          continue;
        }
        const memVector = await ensureCandidateVector(mem, embeddings);
        const queryVec = await embeddings.embed(fb.queryText);
        const feats = buildRelevanceFeatures(queryVec, { ...mem, vector: memVector });
        hydrated.push({
          cosine: feats.asRecord.cosine,
          age_days: feats.asRecord.age_days,
          importance: feats.asRecord.importance,
          category_preference: feats.asRecord.category_preference,
          category_fact: feats.asRecord.category_fact,
          category_decision: feats.asRecord.category_decision,
          category_entity: feats.asRecord.category_entity,
          category_other: feats.asRecord.category_other,
          score: clamp01(fb.score),
        });
      }
      if (hydrated.length < MIN_TRAINING_SAMPLES) {
        throw new Error(
          `memory-synapcores.trainRelevanceModel: after hydrating, only ${hydrated.length} feedback rows resolve to known memories (${missing} were missing) — need at least ${MIN_TRAINING_SAMPLES}`,
        );
      }

      // 2. CREATE TABLE IF NOT EXISTS. The gateway's SQL surface accepts
      //    standard CREATE TABLE. If the table already exists from a
      //    prior run we swallow the duplicate-table error and keep
      //    appending rows.
      const createSql = `CREATE TABLE ${stagingTable} (cosine FLOAT, age_days FLOAT, importance FLOAT, category_preference FLOAT, category_fact FLOAT, category_decision FLOAT, category_entity FLOAT, category_other FLOAT, score FLOAT)`;
      try {
        await client.executeQuery({ sql: createSql });
      } catch (err) {
        const msg = (err as { message?: string })?.message ?? String(err);
        if (!/already exists|duplicate/i.test(msg)) {
          // Some gateways return "Table … created successfully" as data
          // rather than throwing on re-create; only re-throw on truly
          // unexpected errors.
          throw new Error(
            `memory-synapcores.trainRelevanceModel: failed to provision staging table '${stagingTable}': ${msg}`,
          );
        }
      }

      // 3. INSERT each row. We send them individually to keep the SQL
      //    body small and stay friendly to the gateway's row limit.
      for (const row of hydrated) {
        const sql =
          `INSERT INTO ${stagingTable} (cosine, age_days, importance, category_preference, category_fact, category_decision, category_entity, category_other, score) VALUES (` +
          [
            row.cosine,
            row.age_days,
            row.importance,
            row.category_preference,
            row.category_fact,
            row.category_decision,
            row.category_entity,
            row.category_other,
            row.score,
          ]
            .map((n) => Number.isFinite(n) ? String(n) : "0")
            .join(", ") +
          ")";
        try {
          await client.executeQuery({ sql });
        } catch (err) {
          const msg = (err as { message?: string })?.message ?? String(err);
          throw new Error(
            `memory-synapcores.trainRelevanceModel: failed to insert training row into '${stagingTable}': ${msg}`,
          );
        }
      }

      // 4. Train via /v1/automl/train. The gateway issues
      //    `SELECT * FROM {collection}` internally and trains on the
      //    `target` column.
      const model = await client.automl.train({
        collection: stagingTable,
        target: "score",
        task: "regression",
        name: modelName,
        max_trials: 5,
        validation_split: 0.2,
      } as {
        collection: string;
        target: string;
        task: "regression";
        name: string;
        max_trials: number;
        validation_split: number;
      });

      return {
        modelId: model.id ?? modelName,
        modelName: model.name ?? modelName,
      };
    },
  };
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function extractPrediction(raw: unknown): number {
  if (typeof raw === "number") return raw;
  if (raw && typeof raw === "object") {
    const r = raw as Record<string, unknown>;
    if (typeof r.relevance === "number") return r.relevance as number;
    if (typeof r.prediction === "number") return r.prediction as number;
    if (typeof r.score === "number") return r.score as number;
    if (typeof r.value === "number") return r.value as number;
  }
  return 0;
}

// ============================================================================
// Plugin Definition
// ============================================================================

const memoryPlugin = {
  id: "memory-synapcores",
  name: "Memory (SynapCores)",
  description: "SynapCores-backed long-term memory with auto-recall/capture, SQL filtering, graph relations, and AutoML relevance",
  kind: "memory" as const,
  configSchema: memoryConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = memoryConfigSchema.parse(api.pluginConfig);
    const namespace = cfg.collection ?? DEFAULT_COLLECTION;
    // The new MemoryClient enforces namespace ^[A-Za-z_][A-Za-z0-9_]*$. Fail
    // loud and early if the OpenClaw config's `collection` value would be
    // rejected.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(namespace)) {
      throw new Error(
        `memory-synapcores: collection '${namespace}' is not a valid namespace ` +
          `(must match /^[A-Za-z_][A-Za-z0-9_]*$/). Update plugins.entries.memory-synapcores.config.collection.`,
      );
    }

    const client = new SynapCores({
      host: cfg.synapcores.host,
      port: cfg.synapcores.port,
      useHttps: cfg.synapcores.useHttps,
      apiKey: cfg.synapcores.apiKey,
    });

    // Pre-compaction flush is part of the automatic-capture lifecycle, so it
    // is only registered when the operator has opted into `autoCapture`. With
    // the privacy-safe default (autoCapture off) the plugin never prompts the
    // agent to store facts on its own — memory is written only when the user
    // explicitly calls memory_store / the `ltm` command.
    // Same official mechanism OpenClaw's bundled memory-core plugin uses — the
    // runtime calls the resolver right before the session auto-compacts.
    // Guarded: older hosts (pre-dating this API) would otherwise crash the
    // entire register() call over one optional capability.
    if (cfg.autoCapture && typeof api.registerMemoryCapability === "function") {
      api.registerMemoryCapability({
        flushPlanResolver: buildMemoryFlushPlan,
      });
    } else if (cfg.autoCapture) {
      api.logger.debug?.(
        "memory-synapcores: host does not support registerMemoryCapability; skipping pre-compaction flush",
      );
    }

    const db = new MemoryDB(client, namespace);
    // The memory OBJECT reuses the collection name by default so an operator
    // has one name to think about, not two. Its validity as an engine
    // identifier is checked during capability detection, not here — a name the
    // object backend cannot take is a reason to stay on the legacy path, not a
    // reason to refuse to load.
    const memoryObjectName = cfg.memory.name ?? namespace;
    const objectDb = new MemoryObjectDB(client, memoryObjectName);
    const identities = new IdentityResolver(cfg.memory);
    const router = new MemoryRouter(db, objectDb, cfg.memory, identities, api.logger);
    const embeddings = new Embeddings(client);
    const extensions = createExtensions(
      db,
      router,
      identities,
      embeddings,
      client,
      cfg.graph!,
      cfg.workspace,
    );
    const autoLinkSimilar = cfg.autoLinkSimilar !== false;

    api.logger.info(
      `memory-synapcores: plugin registered (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, namespace: ${namespace}, memory: ${memoryObjectName}, backend: ${cfg.memory.backend}, identity: ${cfg.memory.identityFrom}, autoLinkSimilar: ${autoLinkSimilar}, lazy init)`,
    );

    /**
     * Render a thrown memory error as a tool result.
     *
     * A MEMORY_ACCESS_DENIED refusal and a "this engine is too old" refusal are
     * both answers the model can act on, so they come back as ordinary tool
     * output with a stable `details.error` code rather than an exception that
     * the host renders as a stack trace. Anything else still throws.
     */
    const asToolError = (err: unknown) => {
      if (
        err instanceof MemoryAccessDeniedError ||
        err instanceof MemoryCredentialUnpinnedError ||
        err instanceof MemoryFeatureUnavailableError
      ) {
        return {
          content: [{ type: "text" as const, text: err.message }],
          details: { error: err.code, message: err.message },
        };
      }
      throw err;
    };

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall",
        description:
          "Search through long-term memories from past conversations. Call this proactively " +
          "at the start of a conversation or whenever the user references something that " +
          "sounds like it was established before (a preference, a name, a past decision, " +
          "a constraint like an allergy) — don't wait to be asked \"do you remember\".",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(Type.Number({ description: "Max results (default: 5)" })),
        }),
        async execute(_toolCallId, params) {
          const { query, limit = 5 } = params as { query: string; limit?: number };

          let identity: string | undefined;
          try {
            identity = identities.forTool("memory_recall");
          } catch (err) {
            return asToolError(err);
          }

          // On the memory-object backend this is a RECALL: the engine
          // assembles the context, keeps authoritative current state separate
          // from similar-looking older episodes, and hands back a
          // prompt-ready projection. Use it verbatim rather than re-deriving
          // one from a flat list.
          let recalled;
          try {
            recalled = await router.recallForPrompt(query, identity, limit);
          } catch (err) {
            return asToolError(err);
          }
          const { promptReady, truncated, results, currentState } = recalled;

          if (results.length === 0 && Object.keys(currentState).length === 0) {
            return {
              content: [{ type: "text", text: "No relevant memories found." }],
              details: { count: 0 },
            };
          }

          const text =
            promptReady ??
            results
              .map(
                (r, i) =>
                  `${i + 1}. [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}%)`,
              )
              .join("\n");

          // `score` is RELEVANCE and `confidence` is EVIDENCE STRENGTH. They
          // stay separate fields all the way out to the model — collapsing
          // them is the mistake the envelope's shape exists to prevent.
          const sanitizedResults = results.map((r) => ({
            id: r.entry.id,
            text: r.entry.text,
            category: r.entry.category,
            importance: r.entry.importance,
            score: r.score,
            ...(r.confidence === undefined ? {} : { confidence: r.confidence }),
          }));

          return {
            content: [
              {
                type: "text",
                text: promptReady
                  ? `Relevant memory:\n\n${text}`
                  : `Found ${results.length} memories:\n\n${text}`,
              },
            ],
            details: {
              count: results.length,
              memories: sanitizedResults,
              // Authoritative current state, reported separately from the
              // similarity-ranked list so the model never has to guess which
              // of several similar strings is the one that is true now.
              currentState,
              truncated,
            },
          };
        },
      },
      { name: "memory_recall" },
    );

    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store",
        description:
          "Save durable information to long-term memory, in your own words. Call this " +
          "proactively — don't wait to be told to remember something. Store it whenever the " +
          "user states: a personal preference or opinion (favorite/likes/dislikes/prefers), " +
          "an identifying fact (name, job, location, relationships), a constraint that affects " +
          "future answers (allergy, restriction, accessibility need), or a decision/commitment " +
          "made during the conversation. When in doubt about whether something is worth " +
          "remembering, store it — the cost of a missed memory is higher than an extra one.",
        parameters: Type.Object({
          text: Type.String({ description: "Information to remember" }),
          importance: Type.Optional(Type.Number({ description: "Importance 0-1 (default: 0.7)" })),
          category: Type.Optional(stringEnum(MEMORY_CATEGORIES)),
        }),
        async execute(_toolCallId, params) {
          const {
            text,
            importance = 0.7,
            category = "other",
          } = params as {
            text: string;
            importance?: number;
            category?: MemoryEntry["category"];
          };

          let identity: string | undefined;
          try {
            identity = identities.forTool("memory_store");
          } catch (err) {
            return asToolError(err);
          }

          // Duplicate suppression is BACKEND-DEPENDENT on purpose.
          //
          // Legacy `MEMORY_STORE` is an unconditional insert, so the plugin has
          // always had to probe for a near-identical memory first. The memory
          // object owns that policy itself — REMEMBER feeds consolidation,
          // which collapses duplicate facts, and conflict resolution, which
          // supersedes contradicted ones — so probing there costs a round-trip
          // AND second-guesses the engine's supersession logic.
          //
          // `router.kind()` is inside the try because backend detection is
          // where an unpinned credential is discovered — that has to reach the
          // model as a readable tool result, not as a raw rejection.
          let stored;
          try {
            if ((await router.kind()) === "legacy") {
              const existing = await db.searchByQuery(text, 1, 0.95);
              if (existing.length > 0) {
                return {
                  content: [
                    {
                      type: "text",
                      text: `Similar memory already exists: "${existing[0].entry.text}"`,
                    },
                  ],
                  details: {
                    action: "duplicate",
                    existingId: existing[0].entry.id,
                    existingText: existing[0].entry.text,
                  },
                };
              }
            }

            stored = await router.store(
              { text, vector: [], importance, category },
              identity,
              cfg.memory.sourceType,
            );
          } catch (err) {
            return asToolError(err);
          }
          const entry: MemoryEntry = {
            id: stored.id,
            text,
            vector: [],
            importance,
            category,
            createdAt: stored.createdAt,
          };

          // linkSimilarMemories inserts a Memory graph node carrying an
          // engine-native embedded representation (via `client.embed()`) so
          // recallRelated has something to MATCH against. Failures are
          // non-fatal (logged in-helper).
          if (autoLinkSimilar) {
            try {
              const embedVec = await embeddings.embed(text);
              await linkSimilarMemories(entry, client, embedVec, api.logger);
            } catch (err) {
              api.logger.warn(
                `memory-synapcores: graph-node embed failed for ${entry.id}: ${String(err)} (capture still succeeded)`,
              );
            }
          }

          return {
            content: [{ type: "text", text: `Stored: "${text.slice(0, 100)}..."` }],
            details: {
              action: "created",
              id: entry.id,
              // What the engine derived from this one statement (facts,
              // state changes, relationships, supersessions). Legacy stores
              // report nothing here — there is nothing to derive.
              ...(stored.changes ? { changes: stored.changes } : {}),
              ...(stored.status ? { status: stored.status } : {}),
            },
          };
        },
      },
      { name: "memory_store" },
    );

    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget",
        description: "Delete specific memories. GDPR-compliant.",
        parameters: Type.Object({
          query: Type.Optional(Type.String({ description: "Search to find memory" })),
          memoryId: Type.Optional(Type.String({ description: "Specific memory ID" })),
        }),
        async execute(_toolCallId, params) {
          const { query, memoryId } = params as { query?: string; memoryId?: string };

          let identity: string | undefined;
          try {
            identity = identities.forTool("memory_forget");
          } catch (err) {
            return asToolError(err);
          }

          if (memoryId) {
            let removed;
            try {
              removed = await router.deleteById(memoryId, identity);
            } catch (err) {
              return asToolError(err);
            }
            return {
              content: [{ type: "text", text: `Memory ${memoryId} forgotten.` }],
              details: {
                action: "deleted",
                id: memoryId,
                // FORGET propagates to derived facts, embeddings,
                // relationships, current-state values and provenance, and
                // reports the count per category. Legacy deletes one row.
                ...(removed.affected ? { affected: removed.affected } : {}),
              },
            };
          }

          if (query) {
            // The memory object has a real scope-based FORGET: the engine
            // resolves the scope semantically, propagates the removal, and
            // returns a receipt. The legacy path has no such verb, which is
            // why the search-then-confirm dance below still exists for it.
            let scoped;
            try {
              scoped = await router.forgetAbout(query, identity);
            } catch (err) {
              return asToolError(err);
            }
            if (scoped.supported) {
              const affected = (scoped.affected ?? {}) as Record<string, unknown>;
              const total = Object.values(affected).reduce<number>(
                (sum, v) => sum + (typeof v === "number" ? v : 0),
                0,
              );
              return {
                content: [
                  {
                    type: "text",
                    text:
                      total > 0
                        ? `Forgot everything about "${query}" (${total} record(s) across episodes, facts, state, relationships and provenance).`
                        : `Nothing matching "${query}" was stored.`,
                  },
                ],
                details: { action: "forgot_scope", scope: query, affected, status: scoped.status },
              };
            }

            const results = await db.searchByQuery(query, 5, 0.7);

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No matching memories found." }],
                details: { found: 0 },
              };
            }

            if (results.length === 1 && results[0].score > 0.9) {
              await db.delete(results[0].entry.id);
              return {
                content: [{ type: "text", text: `Forgotten: "${results[0].entry.text}"` }],
                details: { action: "deleted", id: results[0].entry.id },
              };
            }

            const list = results
              .map((r) => `- [${r.entry.id.slice(0, 8)}] ${r.entry.text.slice(0, 60)}...`)
              .join("\n");

            const sanitizedCandidates = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              score: r.score,
            }));

            return {
              content: [
                {
                  type: "text",
                  text: `Found ${results.length} candidates. Specify memoryId:\n${list}`,
                },
              ],
              details: { action: "candidates", candidates: sanitizedCandidates },
            };
          }

          return {
            content: [{ type: "text", text: "Provide query or memoryId." }],
            details: { error: "missing_param" },
          };
        },
      },
      { name: "memory_forget" },
    );

    // ------------------------------------------------------------------------
    // CREATE MEMORY-only tools (engine v1.14.3+)
    //
    // Registered unless the operator has pinned the plugin to the legacy
    // backend, because capability detection is async and registration is not.
    // On a legacy gateway they answer with a stable MEMORY_FEATURE_UNAVAILABLE
    // result instead of failing — the tool contract in openclaw.plugin.json
    // stays the same whatever the engine turns out to be.
    // ------------------------------------------------------------------------

    if (cfg.memory.backend !== "legacy") {
      api.registerTool(
        {
          name: "memory_current",
          label: "Memory Current Value",
          description:
            "Look up the CURRENT, authoritative value of one remembered attribute (e.g. " +
            "subscription_plan, cloud_provider, dietary_restriction). Use this — not " +
            "memory_recall — whenever you need an exact fact rather than related context: it " +
            "is a deterministic state lookup, never a similarity search, so it cannot return " +
            "an outdated value that merely looks similar. It reports one of: the accepted " +
            "value with its authority and since-when; nothing recorded; or a conflict with " +
            "both competing claims, which the engine deliberately will not resolve for you.",
          parameters: Type.Object({
            attribute: Type.String({
              description: "Attribute name, e.g. subscription_plan (identifier characters only)",
            }),
          }),
          async execute(_toolCallId, params) {
            const { attribute } = params as { attribute: string };
            try {
              const identity = identities.forTool("memory_current");
              const envelope = await router.current(attribute, identity);
              const data = (envelope.data ?? {}) as Record<string, unknown>;

              if (envelope.status === "not_found") {
                return {
                  content: [{ type: "text", text: `Nothing is recorded for '${attribute}'.` }],
                  details: { status: "not_found", attribute, value: null },
                };
              }
              if (envelope.status === "conflict") {
                // The engine does NOT pick between competing authoritative
                // claims, and neither does the plugin. Both are handed to the
                // model so it can ask rather than assert.
                const claims = Array.isArray(data.competing_claims) ? data.competing_claims : [];
                const rendered = claims
                  .map((c) => {
                    const claim = (c ?? {}) as Record<string, unknown>;
                    return `- ${String(claim.value)} (${String(claim.authority ?? "unknown authority")}, confidence ${String(claim.confidence ?? "?")})`;
                  })
                  .join("\n");
                return {
                  content: [
                    {
                      type: "text",
                      text:
                        `'${attribute}' has competing claims that could not be separated by ` +
                        `authority, time or confidence:\n${rendered}\n` +
                        "Ask the user which is correct rather than picking one.",
                    },
                  ],
                  details: { status: "conflict", attribute, competingClaims: claims },
                };
              }

              return {
                content: [
                  {
                    type: "text",
                    text:
                      `${attribute} = ${String(data.value)} ` +
                      `(authority: ${String(data.authority ?? "unknown")}, since ${String(data.valid_from ?? "unknown")})`,
                  },
                ],
                details: {
                  status: envelope.status ?? "ok",
                  attribute,
                  value: data.value ?? null,
                  // Evidence strength, NOT a retrieval score.
                  confidence: data.confidence ?? null,
                  authority: data.authority ?? null,
                  validFrom: data.valid_from ?? null,
                  sourceMemoryId: data.source_memory_id ?? null,
                },
              };
            } catch (err) {
              return asToolError(err);
            }
          },
        },
        { name: "memory_current" },
      );

      api.registerTool(
        {
          name: "memory_trace",
          label: "Memory Trace",
          description:
            "Explain WHY a remembered value is believed: the source statement verbatim, when it " +
            "was recorded versus when the event happened, how it was extracted, the confidence, " +
            "and what it superseded. Use it when a fact is disputed, when a decision is " +
            "high-stakes, or to check yourself before asserting something from memory.",
          parameters: Type.Object({
            attribute: Type.String({ description: "Attribute name to explain" }),
          }),
          async execute(_toolCallId, params) {
            const { attribute } = params as { attribute: string };
            try {
              const identity = identities.forTool("memory_trace");
              const envelope = await router.trace(attribute, identity);
              const data = (envelope.data ?? {}) as Record<string, unknown>;
              const sources = Array.isArray(data.derived_from) ? data.derived_from : [];

              if (envelope.status === "not_found" || sources.length === 0) {
                return {
                  content: [{ type: "text", text: `No provenance recorded for '${attribute}'.` }],
                  details: { status: envelope.status ?? "not_found", attribute },
                };
              }

              const rendered = sources
                .map((s) => {
                  const src = (s ?? {}) as Record<string, unknown>;
                  return (
                    `- "${String(src.content ?? "")}" ` +
                    `(${String(src.source_type ?? "unknown source")}, recorded ${String(src.recorded_at ?? "?")}` +
                    `, event ${String(src.event_time ?? "?")}, via ${String(src.extraction_method ?? "?")})`
                  );
                })
                .join("\n");

              return {
                content: [
                  {
                    type: "text",
                    text:
                      `${attribute} = ${String(data.value)} (confidence ${String(data.confidence ?? "?")}) ` +
                      `is believed because:\n${rendered}`,
                  },
                ],
                details: {
                  status: envelope.status ?? "ok",
                  attribute,
                  value: data.value ?? null,
                  confidence: data.confidence ?? null,
                  derivedFrom: sources,
                  supersedes: data.supersedes ?? null,
                  derivation: data.derivation ?? null,
                },
              };
            } catch (err) {
              return asToolError(err);
            }
          },
        },
        { name: "memory_trace" },
      );

      api.registerTool(
        {
          name: "memory_history",
          label: "Memory History",
          description:
            "Show the TIMELINE of how a remembered attribute's value changed over time — the " +
            "ordered list of what it was, when each value held, and what superseded what. Use " +
            "this when the user asks what they used to think/prefer/decide, or when you need to " +
            "reason about a change ('you switched from X to Y last month'), rather than just the " +
            "current value (memory_current) or related context (memory_recall). Omit the " +
            "attribute to get the most recent changes across all attributes.",
          parameters: Type.Object({
            attribute: Type.Optional(
              Type.String({
                description:
                  "Attribute whose history to show, e.g. subscription_plan. Omit for a cross-attribute recent timeline.",
              }),
            ),
            limit: Type.Optional(
              Type.Integer({ minimum: 1, description: "Max timeline entries to return." }),
            ),
          }),
          async execute(_toolCallId, params) {
            const { attribute, limit } = params as { attribute?: string; limit?: number };
            try {
              const identity = identities.forTool("memory_history");
              const envelope = await router.history(attribute, identity, limit);
              const data = (envelope.data ?? {}) as Record<string, unknown>;
              const entries = Array.isArray(data.history)
                ? data.history
                : Array.isArray(data.entries)
                  ? data.entries
                  : [];

              if (envelope.status === "not_found" || entries.length === 0) {
                return {
                  content: [
                    {
                      type: "text",
                      text: attribute
                        ? `No history recorded for '${attribute}'.`
                        : "No history recorded yet.",
                    },
                  ],
                  details: { status: envelope.status ?? "not_found", attribute: attribute ?? null },
                };
              }

              const rendered = entries
                .map((e) => {
                  const row = (e ?? {}) as Record<string, unknown>;
                  const attr = row.attribute ? `${String(row.attribute)} = ` : "";
                  return (
                    `- ${attr}${String(row.value ?? "")} ` +
                    `(${String(row.valid_from ?? row.recorded_at ?? "?")}` +
                    `${row.valid_to ? ` → ${String(row.valid_to)}` : ""}` +
                    `${row.authority ? `, ${String(row.authority)}` : ""})`
                  );
                })
                .join("\n");

              return {
                content: [
                  {
                    type: "text",
                    text:
                      (attribute ? `History of '${attribute}':` : "Recent memory history:") +
                      `\n${rendered}`,
                  },
                ],
                details: {
                  status: envelope.status ?? "ok",
                  attribute: attribute ?? null,
                  history: entries,
                },
              };
            } catch (err) {
              return asToolError(err);
            }
          },
        },
        { name: "memory_history" },
      );
    }

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const memory = program.command("ltm").description("SynapCores memory plugin commands");

        memory
          .command("list")
          .description("List memories")
          .action(async () => {
            const count = await router.count(identities.forTool("ltm list"));
            console.log(`Total memories: ${count}`);
          });

        memory
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .action(async (query, opts) => {
            const results = await router.searchByQuery(
              query,
              parseInt(opts.limit),
              0.3,
              identities.forTool("ltm search"),
            );
            const output = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              importance: r.entry.importance,
              score: r.score,
              ...(r.confidence === undefined ? {} : { confidence: r.confidence }),
            }));
            console.log(JSON.stringify(output, null, 2));
          });

        memory
          .command("stats")
          .description("Show memory statistics")
          .action(async () => {
            const backend = await router.kind();
            console.log(`Backend: ${backend}`);
            if (backend === "memory-object") {
              console.log(`Memory object: ${router.memoryName}`);
              console.log(`Identity source: ${identities.source}`);
              // Prefer the engine's own status/describe over a client-side
              // COUNT, which is wrong under a credential-pinned identity.
              try {
                const envelope = await router.status();
                console.log(JSON.stringify(envelope.data ?? {}, null, 2));
                return;
              } catch (err) {
                console.log(`(status unavailable: ${(err as { message?: string })?.message ?? err})`);
              }
            } else {
              console.log(`Namespace: ${namespace}`);
            }
            const count = await router.count(identities.forTool("ltm stats"));
            console.log(`Total memories: ${count}`);
          });

        memory
          .command("describe")
          .description("Describe the memory object's definition and options (memory-object backend)")
          .action(async () => {
            const envelope = await router.describe();
            console.log(JSON.stringify(envelope.data ?? {}, null, 2));
          });

        memory
          .command("history")
          .description("Show how an attribute's value changed over time (memory-object backend)")
          .argument("[attribute]", "Attribute to show history for; omit for a recent cross-attribute timeline")
          .option("--limit <n>", "Max timeline entries")
          .action(async (attribute, opts) => {
            const limit = opts.limit ? parseInt(opts.limit, 10) : undefined;
            const envelope = await router.history(
              attribute,
              identities.forTool("ltm history"),
              limit,
            );
            console.log(JSON.stringify(envelope.data ?? {}, null, 2));
          });

        memory
          .command("consolidate")
          .description("Run memory consolidation now (memory-object backend)")
          .action(async () => {
            const envelope = await router.consolidate(identities.forTool("ltm consolidate"));
            console.log(JSON.stringify(envelope.data ?? {}, null, 2));
          });

        memory
          .command("supersede")
          .description("Explicitly set an attribute's current value (memory-object backend)")
          .argument("<attribute>", "Attribute name")
          .argument("<value>", "New current value")
          .action(async (attribute, value) => {
            const envelope = await router.supersede(
              attribute,
              value,
              identities.forTool("ltm supersede"),
            );
            console.log(JSON.stringify(envelope.data ?? {}, null, 2));
          });

        memory
          .command("reconcile")
          .description("Re-run conflict resolution, optionally for one attribute (memory-object backend)")
          .argument("[attribute]", "Attribute to reconcile; omit for all")
          .action(async (attribute) => {
            const envelope = await router.reconcile(
              attribute,
              identities.forTool("ltm reconcile"),
            );
            console.log(JSON.stringify(envelope.data ?? {}, null, 2));
          });

        memory
          .command("drop")
          .description("Drop the memory object (memory-object backend)")
          .option("--cascade", "Also drop all stored memories")
          .action(async (opts) => {
            const envelope = await router.drop({ cascade: opts.cascade === true, ifExists: true });
            console.log(JSON.stringify(envelope.data ?? { dropped: router.memoryName }, null, 2));
          });
      },
      { commands: ["ltm"] },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Auto-recall: inject relevant memories into the prompt build.
    // Uses the current `before_prompt_build` hook (the phase OpenClaw's own
    // Active Memory plugin uses for recall injection). The event still carries
    // `prompt`, and the result type (PluginHookBeforePromptBuildResult) still
    // exposes `prependContext` — so per-turn recall injection is preserved.
    if (cfg.autoRecall) {
      // `ctx` is OpenClaw's PluginHookAgentContext — the ONLY place the host
      // hands a plugin a caller identity (senderId / accountId / sessionKey).
      // `memory.identityFrom: "host"` reads it here; tools cannot, which is
      // why they refuse rather than guess (see IdentityResolver).
      api.on("before_prompt_build", async (event, ctx) => {
        if (!event.prompt || event.prompt.length < 5) {
          return;
        }

        const identity = identities.forHook(ctx, "auto-recall");
        if (identity === null) {
          // `host` mode with nothing to derive an identity from. Skipping is
          // the safe answer: the alternative is injecting some other
          // identity's memories into this user's prompt.
          api.logger.debug?.(
            "memory-synapcores: auto-recall skipped — no memory identity in the hook context",
          );
          return;
        }

        try {
          const { promptReady, truncated, results } = await router.recallForPrompt(
            event.prompt,
            identity,
            3,
          );

          // The engine's own projection when we have one: it already ranks
          // authoritative current state above a semantically similar older
          // episode, and preserves current state when the budget truncates.
          const memoryContext =
            promptReady ??
            (results.length === 0
              ? null
              : results.map((r) => `- [${r.entry.category}] ${r.entry.text}`).join("\n"));

          if (!memoryContext) {
            return;
          }

          api.logger.info?.(
            `memory-synapcores: injecting memory context (${results.length} record(s)${truncated ? ", truncated to budget" : ""})`,
          );

          return {
            prependContext: `<relevant-memories>\nThe following memories may be relevant to this conversation:\n${memoryContext}\n</relevant-memories>`,
          };
        } catch (err) {
          if (
            err instanceof MemoryAccessDeniedError ||
            err instanceof MemoryCredentialUnpinnedError
          ) {
            // An identity refusal, and an unpinned credential, are both
            // configuration problems rather than transient recall failures —
            // say so instead of burying them in "recall failed".
            api.logger.warn(`memory-synapcores: auto-recall refused: ${err.message}`);
            return;
          }
          api.logger.warn(`memory-synapcores: recall failed: ${String(err)}`);
        }
      });
    }

    // Auto-capture: analyze and store important information after agent ends
    if (cfg.autoCapture) {
      api.on("agent_end", async (event, ctx) => {
        if (!event.success || !event.messages || event.messages.length === 0) {
          return;
        }

        const identity = identities.forHook(ctx, "auto-capture");
        if (identity === null) {
          api.logger.debug?.(
            "memory-synapcores: auto-capture skipped — no memory identity in the hook context",
          );
          return;
        }

        try {
          const turns = extractMessageTurns(event.messages);

          // Filter for capturable content
          const toCapture = turns.filter((turn) => turn.text && shouldCapture(turn.text));
          if (toCapture.length === 0) {
            return;
          }

          const backend = await router.kind();

          // Store each capturable piece (limit to 3 per conversation)
          let stored = 0;
          for (const turn of toCapture.slice(0, 3)) {
            const text = turn.text;
            const category = detectCategory(text);

            // Legacy only: `MEMORY_STORE` is an unconditional insert, so the
            // client has to suppress near-duplicates. The memory object owns
            // dedup itself (see the memory_store tool).
            if (backend === "legacy") {
              const existing = await db.searchByQuery(text, 1, 0.95);
              if (existing.length > 0) {
                continue;
              }
            }

            // Claim authority follows the turn's ROLE. A sentence the user
            // said is a user_statement; a sentence the assistant produced is
            // the agent's own derivation, and the engine's conflict
            // resolution ranks it lower — which is exactly right when the two
            // disagree.
            const stored_ = await router.store(
              { text, vector: [], importance: 0.7, category },
              identity,
              turn.role === "user" ? cfg.memory.sourceType : "agent_derivation",
            );
            const entry: MemoryEntry = {
              id: stored_.id,
              text,
              vector: [],
              importance: 0.7,
              category,
              createdAt: stored_.createdAt,
            };
            if (autoLinkSimilar) {
              try {
                const embedVec = await embeddings.embed(text);
                await linkSimilarMemories(entry, client, embedVec, api.logger);
              } catch (err) {
                api.logger.warn(
                  `memory-synapcores: graph-node embed failed for ${entry.id}: ${String(err)}`,
                );
              }
            }
            stored++;
          }

          if (stored > 0) {
            api.logger.info(`memory-synapcores: auto-captured ${stored} memories`);
          }
        } catch (err) {
          if (
            err instanceof MemoryAccessDeniedError ||
            err instanceof MemoryCredentialUnpinnedError
          ) {
            api.logger.warn(`memory-synapcores: auto-capture refused: ${err.message}`);
            return;
          }
          api.logger.warn(`memory-synapcores: capture failed: ${String(err)}`);
        }
      });
    }

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: "memory-synapcores",
      start: async () => {
        // Best-effort startup probe: surface a DB-down / bad-key problem
        // immediately in the logs (and `openclaw plugins doctor`) without
        // bricking the host — memory ops still lazily retry on first use.
        try {
          await preflightGateway(client);
          // Resolve the backend here too, so `openclaw plugins doctor` and the
          // startup log answer "which memory subsystem am I actually on?"
          // without waiting for the first tool call. Non-fatal: memory ops
          // still detect lazily and retry.
          const backend = await router.kind().catch((err) => {
            api.logger.warn(
              `memory-synapcores: memory backend detection deferred: ${String((err as Error)?.message ?? err)}`,
            );
            return null;
          });
          api.logger.info(
            `memory-synapcores: initialized (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, ` +
              `backend: ${backend ?? "pending"}, ` +
              `${backend === "memory-object" ? `memory: ${router.memoryName}, identity: ${identities.source}` : `namespace: ${namespace}`}, ` +
              "embeddings: engine-native)",
          );
        } catch (err) {
          api.logger.warn(`memory-synapcores: ${String((err as Error)?.message ?? err)}`);
          api.logger.warn(
            "memory-synapcores: continuing with lazy init — memory operations will retry on first use.",
          );
        }
      },
      stop: () => {
        api.logger.info("memory-synapcores: stopped");
      },
    });

    // ========================================================================
    // SynapCores-only extensions
    // ========================================================================

    // Expose the extension surface on the plugin instance so callers can
    // reach `recallFiltered` / `recallRelated` / `predictRelevance` /
    // `trainRelevanceModel` via the OpenClaw plugin registry
    // (e.g. `plugin.extensions.recallFiltered`).
    //
    // NOTE: attach to `pluginEntry` (the definePluginEntry result that is the
    // default export), NOT the inner `memoryPlugin` — definePluginEntry returns
    // a fresh normalized object, so attaching to the inner one would hide the
    // extensions from every consumer of the loaded plugin.
    (pluginEntry as unknown as { extensions: MemorySynapCoresExtensions }).extensions = extensions;
  },
};

// Annotate with `ReturnType<typeof definePluginEntry>` (the DefinedPluginEntry
// shape) so the emitted declaration references only the public, importable
// `definePluginEntry` symbol instead of an internal openclaw module path.
// Without this, tsc emits TS2742 because the inferred type points at
// `openclaw/dist/types-*.js`, which is not a portable/nameable reference.
const pluginEntry: ReturnType<typeof definePluginEntry> =
  definePluginEntry(memoryPlugin);

export default pluginEntry;

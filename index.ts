/**
 * OpenClaw Memory (SynapCores) Plugin
 *
 * Long-term memory with vector search for AI conversations.
 * Uses SynapCores AIDB for storage via the engine-side
 * `MEMORY_STORE` / `MEMORY_RECALL` / `MEMORY_FORGET` primitives and
 * OpenAI for the relevance-extension features that still need a
 * client-side embedding (cosine feature in `predictRelevance`,
 * graph-node embedding when `autoLinkSimilar` is on).
 *
 * Provides seamless auto-recall and auto-capture via lifecycle hooks,
 * plus four SynapCores-only extensions (SQL-filtered recall, graph-relation
 * walks, AutoML relevance scoring, and a model-training helper) — see
 * `recallFiltered`, `recallRelated`, `predictRelevance`, and
 * `trainRelevanceModel`.
 *
 * v0.4.0 migrates the core memory ops to `@synapcores/sdk@^0.5.0`'s
 * `client.memory` surface. Requires SynapCores gateway v1.8.5-ce+.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { Type } from "typebox";
import OpenAI from "openai";
import { stringEnum } from "openclaw/plugin-sdk/core";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { SynapCores } from "@synapcores/sdk";
import type { MemoryRecord } from "@synapcores/sdk";
import {
  MEMORY_CATEGORIES,
  type MemoryCategory,
  memoryConfigSchema,
  vectorDimsForModel,
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
  score: number;
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
   */
  private async ensureReady(): Promise<void> {
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
   * SQL-filtered recall. v0.4.0 forwards the WHERE clause through to a
   * `SELECT … FROM MEMORY_RECALL(?, ?, ?) WHERE <where>` query so the
   * filter runs engine-side over the recall result-set.
   *
   * Legacy shorthand: callers wrote v0.3.x filters like
   * `category = 'preference' AND importance >= 0.7`. v0.4.0 stores those
   * fields inside the JSON `metadata` column, so the bare-identifier form
   * would no longer resolve. We rewrite the four well-known shorthands
   * (`category`, `importance`, `createdAt`, `text`) to their JSON-extract
   * form on the way in so most existing callers keep working unchanged.
   * Callers that already use `metadata->>'…'` get a no-op.
   */
  async searchFiltered(
    semantic: string,
    where: string,
    limit = 5,
  ): Promise<MemorySearchResult[]> {
    await this.ensureReady();
    // Oversample so the post-filter still has enough rows to return.
    const oversample = clampTopK(Math.max(limit * 5, 25));
    const rewritten = rewriteLegacyWhere(where);
    const sql =
      "SELECT id, content, similarity, metadata, created_at " +
      `FROM MEMORY_RECALL(?, ?, ?) WHERE ${rewritten} LIMIT ${Number(limit)}`;
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
    return mapRecallRows(result);
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
          "FROM MEMORY_RECALL(?, ?, ?) WHERE id = ? LIMIT 1",
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
  const out: MemorySearchResult[] = [];
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
 * Rewrite a handful of legacy column-name shorthands into the JSON-extract
 * form expected against `MEMORY_RECALL`'s output schema. This is best-
 * effort and intentionally limited — callers writing complex SQL should
 * use the engine-native syntax directly.
 *
 * Rewrites:
 *   category    -> metadata->>'category'
 *   importance  -> CAST(metadata->>'importance' AS REAL)
 *   createdAt   -> CAST(metadata->>'createdAt' AS INTEGER)
 *   text        -> content
 *
 * Identifiers preceded by a dot (e.g. `m.category`) or wrapped in quotes
 * are left untouched.
 */
function rewriteLegacyWhere(where: string): string {
  const rules: Array<[RegExp, string]> = [
    [/(?<![.'"\w])category(?![\w])/g, "metadata->>'category'"],
    [/(?<![.'"\w])importance(?![\w])/g, "CAST(metadata->>'importance' AS REAL)"],
    [/(?<![.'"\w])createdAt(?![\w])/g, "CAST(metadata->>'createdAt' AS INTEGER)"],
    [/(?<![.'"\w])text(?![\w])/g, "content"],
  ];
  let out = where;
  for (const [re, repl] of rules) {
    out = out.replace(re, repl);
  }
  return out;
}

// ============================================================================
// OpenAI Embeddings
// ============================================================================
//
// v0.4.0 NOTE: the engine now owns embedding for the core memory ops
// (`MEMORY_STORE` / `MEMORY_RECALL`). OpenAI is still used by:
//   - `predictRelevance` / `trainRelevanceModel`: client-side cosine
//     between query and candidate text.
//   - `autoLinkSimilar` capture: writing an `embedding` property onto the
//     Memory graph node so `recallRelated`'s synthetic SIMILAR_TO edge
//     resolves at MATCH time.
// Callers that never use the relevance extensions and disable
// `autoLinkSimilar` will not hit the OpenAI client.

class Embeddings {
  private client: OpenAI;

  constructor(
    apiKey: string,
    private model: string,
  ) {
    this.client = new OpenAI({ apiKey });
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
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
  /my\s+\w+\s+is|is\s+my/i,
  /i (like|prefer|hate|love|want|need)/i,
  /always|never|important/i,
];

function shouldCapture(text: string): boolean {
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
 * Inserts the Memory as a graph node carrying the OpenAI embedding under
 * the property name `embedding` — the field the gateway's brute-force
 * vector index is wired against (see
 * `aidb_gateway::routes::graph::attach_default_vector_index`). Once a
 * Memory node exists, `recallRelated` can MATCH `[:SIMILAR_TO > T]`
 * against it and get neighbors back without any pre-stored edges.
 *
 * v0.4.0 NOTE: the engine no longer exposes the vector it used internally
 * for `MEMORY_STORE`, so this helper still embeds the text via the
 * OpenAI client. The two embeddings (engine + OpenAI) need not match;
 * each is independent and serves a different recall path.
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
    async recallFiltered(options: RecallFilteredOptions): Promise<MemorySearchResult[]> {
      const limit = options.limit ?? 5;
      return db.searchFiltered(options.semantic, options.where, limit);
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
      // the stored Memory's content (re-embedded via OpenAI) to get the
      // cosine feature.
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
      for (const fb of feedback) {
        const mem = await db.get(fb.memoryId).catch(() => null);
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
    // Validate `vectorDimsForModel` runs without throwing; the dimension
    // value itself isn't used in v0.4.0 because the engine owns embedding
    // for the core memory ops.
    vectorDimsForModel(cfg.embedding.model ?? "text-embedding-3-small");
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
    const db = new MemoryDB(client, namespace);
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model!);
    const extensions = createExtensions(
      db,
      embeddings,
      client,
      cfg.graph!,
      cfg.workspace,
    );
    const autoLinkSimilar = cfg.autoLinkSimilar !== false;

    api.logger.info(
      `memory-synapcores: plugin registered (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, namespace: ${namespace}, autoLinkSimilar: ${autoLinkSimilar}, lazy init)`,
    );

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall",
        description:
          "Search through long-term memories. Use when you need context about user preferences, past decisions, or previously discussed topics.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(Type.Number({ description: "Max results (default: 5)" })),
        }),
        async execute(_toolCallId, params) {
          const { query, limit = 5 } = params as { query: string; limit?: number };

          const results = await db.searchByQuery(query, limit, 0.1);

          if (results.length === 0) {
            return {
              content: [{ type: "text", text: "No relevant memories found." }],
              details: { count: 0 },
            };
          }

          const text = results
            .map(
              (r, i) =>
                `${i + 1}. [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}%)`,
            )
            .join("\n");

          const sanitizedResults = results.map((r) => ({
            id: r.entry.id,
            text: r.entry.text,
            category: r.entry.category,
            importance: r.entry.importance,
            score: r.score,
          }));

          return {
            content: [{ type: "text", text: `Found ${results.length} memories:\n\n${text}` }],
            details: { count: results.length, memories: sanitizedResults },
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
          "Save important information in long-term memory. Use for preferences, facts, decisions.",
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

          // Check for duplicates via the engine's recall path. The engine
          // re-embeds `text` internally so we don't have to.
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

          const entry = await db.store({
            text,
            vector: [],
            importance,
            category,
          });

          // linkSimilarMemories inserts a Memory graph node carrying an
          // OpenAI-embedded representation so recallRelated has something
          // to MATCH against. Failures are non-fatal (logged in-helper).
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
            details: { action: "created", id: entry.id },
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

          if (memoryId) {
            await db.delete(memoryId);
            return {
              content: [{ type: "text", text: `Memory ${memoryId} forgotten.` }],
              details: { action: "deleted", id: memoryId },
            };
          }

          if (query) {
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
            const count = await db.count();
            console.log(`Total memories: ${count}`);
          });

        memory
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .action(async (query, opts) => {
            const results = await db.searchByQuery(query, parseInt(opts.limit), 0.3);
            const output = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              importance: r.entry.importance,
              score: r.score,
            }));
            console.log(JSON.stringify(output, null, 2));
          });

        memory
          .command("stats")
          .description("Show memory statistics")
          .action(async () => {
            const count = await db.count();
            console.log(`Total memories: ${count}`);
          });
      },
      { commands: ["ltm"] },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Auto-recall: inject relevant memories before agent starts
    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event) => {
        if (!event.prompt || event.prompt.length < 5) {
          return;
        }

        try {
          const results = await db.searchByQuery(event.prompt, 3, 0.3);

          if (results.length === 0) {
            return;
          }

          const memoryContext = results
            .map((r) => `- [${r.entry.category}] ${r.entry.text}`)
            .join("\n");

          api.logger.info?.(`memory-synapcores: injecting ${results.length} memories into context`);

          return {
            prependContext: `<relevant-memories>\nThe following memories may be relevant to this conversation:\n${memoryContext}\n</relevant-memories>`,
          };
        } catch (err) {
          api.logger.warn(`memory-synapcores: recall failed: ${String(err)}`);
        }
      });
    }

    // Auto-capture: analyze and store important information after agent ends
    if (cfg.autoCapture) {
      api.on("agent_end", async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) {
          return;
        }

        try {
          // Extract text content from messages (handling unknown[] type)
          const texts: string[] = [];
          for (const msg of event.messages) {
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
              texts.push(content);
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
                  texts.push((block as Record<string, unknown>).text as string);
                }
              }
            }
          }

          // Filter for capturable content
          const toCapture = texts.filter((text) => text && shouldCapture(text));
          if (toCapture.length === 0) {
            return;
          }

          // Store each capturable piece (limit to 3 per conversation)
          let stored = 0;
          for (const text of toCapture.slice(0, 3)) {
            const category = detectCategory(text);

            // Check for duplicates (high similarity threshold)
            const existing = await db.searchByQuery(text, 1, 0.95);
            if (existing.length > 0) {
              continue;
            }

            const entry = await db.store({
              text,
              vector: [],
              importance: 0.7,
              category,
            });
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
          api.logger.info(
            `memory-synapcores: initialized (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, namespace: ${namespace}, model: ${cfg.embedding.model})`,
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

const pluginEntry = definePluginEntry(memoryPlugin);

export default pluginEntry;

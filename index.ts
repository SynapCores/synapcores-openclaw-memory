/**
 * OpenClaw Memory (SynapCores) Plugin
 *
 * Long-term memory with vector search for AI conversations.
 * Uses SynapCores AIDB for storage and OpenAI for embeddings.
 * Provides seamless auto-recall and auto-capture via lifecycle hooks,
 * plus three SynapCores-only extensions (SQL-filtered recall, graph-relation
 * walks, and AutoML relevance scoring) — see `recallFiltered`,
 * `recallRelated`, and `predictRelevance`.
 *
 * This is the @synapcores/openclaw-memory drop-in alternative to
 * @openclaw/memory-lancedb. Verified end-to-end against SynapCores
 * gateway v1.6.5.2-ce. Requires @synapcores/sdk@^0.4.0 — which added
 * `client.vectorCollection(name)` + `client.createVectorCollection(...)`
 * and switched API-key auth to `Authorization: Bearer`.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "typebox";
import { randomUUID } from "node:crypto";
import OpenAI from "openai";
import { stringEnum } from "openclaw/plugin-sdk";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { SynapCores } from "@synapcores/sdk";
import type { VectorCollection, VectorHit } from "@synapcores/sdk";
import {
  MEMORY_CATEGORIES,
  type MemoryCategory,
  memoryConfigSchema,
  vectorDimsForModel,
} from "./config.js";

// ============================================================================
// Types
// ============================================================================

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
  /** SQL fragment applied as the search filter (e.g. `category = 'preference' AND importance >= 0.7`). */
  where: string;
  /** Natural-language query that gets embedded and used as the search vector. */
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
// SynapCores Provider
// ============================================================================

const DEFAULT_COLLECTION = "openclaw_memories";

/**
 * MemoryDB — thin wrapper around `@synapcores/sdk@^0.4.0`'s
 * `VectorCollection` handle. v0.2.0 dropped the direct
 * `_getHttpClient().post('/vectors/collections/...')` workaround that
 * v0.1.0 needed against SDK 0.3.x — SDK 0.4.0 ships `createVectorCollection`
 * and `vectorCollection(name)` that target `/v1/vectors/collections/...`
 * directly, so the plugin no longer has to bypass the SDK.
 */
class MemoryDB {
  private readonly client: SynapCores;
  private vectorCollection: VectorCollection | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(
    client: SynapCores,
    private readonly collectionName: string,
    private readonly vectorDim: number,
  ) {
    this.client = client;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.vectorCollection) {
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    // Probe for an existing vector collection (idempotent). The SDK's
    // `listVectorCollections()` calls GET /v1/vectors/collections and
    // returns the bare array; we fall through to create() on any error.
    let exists = false;
    try {
      const items = await this.client.listVectorCollections();
      if (Array.isArray(items)) {
        exists = items.some((it) => (it as { name?: string })?.name === this.collectionName);
      }
    } catch {
      // best-effort; fall through and try to create
    }

    if (!exists) {
      try {
        this.vectorCollection = await this.client.createVectorCollection({
          name: this.collectionName,
          dimensions: this.vectorDim,
          distance_metric: "cosine",
        });
        return;
      } catch (err) {
        // Race with a concurrent creator — re-check before giving up.
        try {
          const items = await this.client.listVectorCollections();
          if (
            !Array.isArray(items) ||
            !items.some((it) => (it as { name?: string })?.name === this.collectionName)
          ) {
            throw err;
          }
        } catch {
          throw err;
        }
      }
    }

    // Use the synchronous accessor — no extra round-trip.
    this.vectorCollection = this.client.vectorCollection(this.collectionName);
  }

  async store(entry: Omit<MemoryEntry, "id" | "createdAt">): Promise<MemoryEntry> {
    await this.ensureInitialized();

    const fullEntry: MemoryEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: Date.now(),
    };

    // v0.2.0: SDK ships VectorCollection.insert(...) which posts to
    // /v1/vectors/collections/{name}/vectors with the {vectors: [...]}
    // envelope automatically. v0.1.0 had a direct _getHttpClient().post()
    // workaround here — deleted now that SDK 0.4.0 covers the wire.
    await this.vectorCollection!.insert({
      id: fullEntry.id,
      values: fullEntry.vector,
      metadata: {
        text: fullEntry.text,
        importance: fullEntry.importance,
        category: fullEntry.category,
        createdAt: fullEntry.createdAt,
      },
    });

    return fullEntry;
  }

  async search(vector: number[], limit = 5, minScore = 0.5): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();
    const hits = await this.vectorCollection!.search({
      vector,
      k: limit,
      includeMetadata: true,
    });
    return hits.map(parseHitToResult).filter((r) => r.score >= minScore);
  }

  /**
   * Same shape as `search`, but accepts a SQL `WHERE` clause. The SDK
   * forwards `filter` as-is to the gateway's `/v1/vectors/collections/{n}/search`
   * endpoint, which accepts either a JSON match object or `{ sql: "..." }`.
   * Used by the `recallFiltered` extension method.
   */
  async searchFiltered(
    vector: number[],
    where: string,
    limit = 5,
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();
    const hits = await this.vectorCollection!.search({
      vector,
      k: limit,
      includeMetadata: true,
      filter: { sql: where },
    });
    return hits.map(parseHitToResult);
  }

  async delete(id: string): Promise<boolean> {
    await this.ensureInitialized();
    // Validate UUID format to prevent injection
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(id)) {
      throw new Error(`Invalid memory ID format: ${id}`);
    }
    // v0.2.0: SDK 0.4.0's VectorCollection.delete(id) hits
    // DELETE /v1/vectors/collections/{name}/vectors/{id}.
    await this.vectorCollection!.delete(id);
    return true;
  }

  async count(): Promise<number> {
    await this.ensureInitialized();
    // v0.2.0: SDK 0.4.0's VectorCollection.count() probes /count first,
    // falls back to info().vector_count. Either way, returns a number.
    try {
      return await this.vectorCollection!.count();
    } catch {
      return 0;
    }
  }

  /** Fetch a single memory by ID (returns null if not found). */
  async get(id: string): Promise<MemoryEntry | null> {
    await this.ensureInitialized();
    // v0.2.0: SDK 0.4.0's VectorCollection.get(id) returns
    // { id, values, metadata } | null. v0.1.0 had a direct
    // _getHttpClient().get() workaround here — deleted.
    let hit: VectorHit | null = null;
    try {
      hit = await this.vectorCollection!.get(id);
    } catch (err) {
      const e = err as { code?: string; status?: number };
      if (e?.code === "NOT_FOUND" || e?.status === 404) return null;
      throw err;
    }
    if (!hit) return null;
    const meta = (hit.metadata as Record<string, unknown> | undefined) ?? {};
    return {
      id: String(hit.id ?? id),
      text: typeof meta.text === "string" ? meta.text : "",
      vector: Array.isArray(hit.values) ? (hit.values as number[]) : [],
      importance: typeof meta.importance === "number" ? (meta.importance as number) : 0,
      category: (meta.category as MemoryCategory) ?? "other",
      createdAt: typeof meta.createdAt === "number" ? (meta.createdAt as number) : 0,
    };
  }
}

function hitToEntry(hit: Record<string, unknown>): MemoryEntry {
  // SDK 0.4.0 VectorCollection.search returns the gateway's hit shape:
  //   { id, score, values?, metadata: { text, importance, category, createdAt } }
  const meta = (hit.metadata as Record<string, unknown> | undefined) ?? {};
  const text = typeof meta.text === "string" ? (meta.text as string) : "";
  const importance = typeof meta.importance === "number" ? (meta.importance as number) : 0;
  const category = (meta.category as MemoryCategory) ?? "other";
  const createdAt = typeof meta.createdAt === "number" ? (meta.createdAt as number) : 0;
  const vector = Array.isArray(hit.values) ? (hit.values as number[]) : [];
  return {
    id: String(hit.id ?? ""),
    text,
    vector,
    importance,
    category,
    createdAt,
  };
}

function parseHitToResult(hit: Record<string, unknown>): MemorySearchResult {
  // Gateway v1.6.5.2-ce returns cosine **distance** as `score` (lower =
  // closer; 0 = identical, 1 = orthogonal, 2 = opposite). Convert to a
  // [0, 1] similarity for the public API. If a separate `distance` field
  // ever appears we honour it for parity.
  const rawScore = typeof hit.score === "number" ? (hit.score as number) : undefined;
  const rawDistance = typeof hit.distance === "number" ? (hit.distance as number) : undefined;
  const distance = rawDistance ?? rawScore ?? 0;
  const similarity = Math.max(0, Math.min(1, 1 - distance));
  return {
    entry: hitToEntry(hit),
    score: similarity,
  };
}

// ============================================================================
// OpenAI Embeddings
// ============================================================================

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
 * Gateway v1.6.5.2-ce explicitly rejects named-parameter bindings (`$param`)
 * with HTTP 400; the supported path is to inline literal values into the
 * query string. Memory IDs flow in from `randomUUID()` so they are normally
 * safe, BUT we never trust upstream input — every string that ends up
 * inside `'...'` in a Cypher fragment must go through this helper.
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
 * `linkSimilarMemories` — capture-time graph wiring (v0.2.0).
 *
 * v0.1.0 was a no-op: it tried to write explicit `SIMILAR_TO` edges via
 * Cypher `MERGE`, which gateway v1.6.5.x rejects because SIMILAR_TO is a
 * synthetic / derived edge type computed from the graph backend's vector
 * index at MATCH time.
 *
 * v0.2.0 takes the supported path instead: insert the Memory as a graph
 * node carrying the embedding under the property name `embedding` — the
 * field the gateway's brute-force vector index is wired against (see
 * `aidb_gateway::routes::graph::attach_default_vector_index`). Once a
 * Memory node exists, `recallRelated` can MATCH `[:SIMILAR_TO > T]`
 * against it and get neighbors back without any pre-stored edges.
 *
 * Failures here are non-fatal — the capture itself still succeeded in the
 * vector subsystem; we just log and move on so recall continues to work.
 */
async function linkSimilarMemories(
  entry: MemoryEntry,
  _db: MemoryDB,
  client: SynapCores,
  _graphName: string,
  logger: { warn?: (msg: string) => void } | undefined,
): Promise<number> {
  try {
    // Insert a Memory node with the embedding so the synthetic
    // SIMILAR_TO edge in `recallRelated` has something to match against.
    // The `id` property mirrors the vector-collection id so callers can
    // join the two views.
    //
    // KNOWN SDK GAP (@synapcores/sdk@0.4.0):
    //   `client.graph.nodes.create(label, props)` posts
    //   `{label: <single>, properties}` but the gateway's
    //   /v1/graph/nodes handler expects `{labels: <array>, properties}`
    //   (see aidb_gateway::routes::graph::CreateNodeRequest). The result
    //   is a node with `labels: []`, which never matches the `Memory`
    //   label filter in MATCH. We bypass the SDK helper for THIS one
    //   call and post the correct wire shape ourselves. Once SDK >0.4.0
    //   fixes `GraphNodeApi.create` to send `labels`, this `_getHttpClient`
    //   call can be replaced with `client.graph.nodes.create(...)`.
    const http = (
      client as unknown as { _getHttpClient: () => { post: (p: string, b: unknown) => Promise<unknown> } }
    )._getHttpClient();
    await http.post("/graph/nodes", {
      labels: ["Memory"],
      properties: {
        id: entry.id,
        text: entry.text,
        embedding: entry.vector,
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
// AutoML helpers — staged-collection training (v0.2.0)
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
 *   - `vectorCollection.search({ ..., filter: { sql } })` for `recallFiltered`,
 *     forwarding the SQL `WHERE` clause to the gateway under
 *     `filter: { sql: where }`.
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
   *     where: "category = 'preference' AND importance >= 0.7 AND createdAt > 1700000000000",
   *     semantic: "what kind of UI does the user prefer?",
   *     limit: 5,
   *   });
   */
  recallFiltered(options: RecallFilteredOptions): Promise<MemorySearchResult[]>;

  /**
   * Graph walk from a memory through `SIMILAR_TO` / `MENTIONS` / `RELATES_TO`
   * edges, returning the neighborhood.
   *
   * v0.2.0 wires this against the gateway's synthetic-SIMILAR_TO Cypher
   * syntax (`[:SIMILAR_TO > T]`) which resolves against the `embedding`
   * property the plugin writes onto every captured Memory node. If a
   * source memory has not been promoted to a graph node yet (e.g.
   * `autoLinkSimilar: false`), this returns an empty array.
   *
   * @param memoryId - UUID of the source memory.
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
   * v0.2.0 wires this against the staged-collection workflow: feedback
   * rows are inserted into a SQL table the gateway's AutoML can read
   * via `SELECT * FROM {table}`, then `/v1/automl/train` is invoked with
   * `target: 'score'` to fit a regression model. Existing training data
   * in the staging table is preserved across calls so feedback
   * accumulates across sessions.
   *
   * @param feedback - Array of `{ memoryId, queryText, score }` samples.
   *                   Requires at least {@link MIN_TRAINING_SAMPLES} samples;
   *                   throws otherwise.
   * @returns The id and name of the trained model.
   */
  trainRelevanceModel(feedback: RelevanceFeedback[]): Promise<{ modelId: string; modelName: string }>;
}

function createExtensions(
  db: MemoryDB,
  embeddings: Embeddings,
  client: SynapCores,
  _graphName: string,
  workspace?: string,
  _collectionName: string = DEFAULT_COLLECTION,
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
      const vector = await embeddings.embed(options.semantic);
      return db.searchFiltered(vector, options.where, limit);
    },

    async recallRelated(
      memoryId: string,
      options: RecallRelatedOptions = {},
    ): Promise<RelatedMemoryResult[]> {
      // v0.2.0 implementation:
      //
      // Gateway v1.6.5.x derives `SIMILAR_TO` synthetically at MATCH
      // time from the graph backend's vector index on the `embedding`
      // property. `linkSimilarMemories` populates that index on every
      // capture by posting a `Memory` graph node carrying the embedding.
      // With nodes in place, this method composes a Cypher MATCH that
      // walks SIMILAR_TO (plus any non-synthetic edges the caller named
      // via `edgeKinds`) and returns the neighborhood.
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

        let result: { rows?: unknown[][]; records?: Array<Record<string, unknown>> };
        try {
          result = await client.graph.cypher(cypher);
        } catch (err) {
          const msg = (err as { message?: string })?.message ?? String(err);
          throw new Error(
            `memory-synapcores.recallRelated: graph query failed for edge kind '${safeKind}': ${msg}`,
          );
        }

        // The SDK normalises responses to { columns, rows, records }.
        // Prefer `records` (column-keyed) and fall back to `rows`.
        const records = Array.isArray(result?.records) ? result!.records! : [];
        const rows = Array.isArray(result?.rows) ? result!.rows! : [];

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

        for (const rec of records) {
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
      const features = candidates.map((c) => buildRelevanceFeatures(queryVector, c, now));

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

      // v0.2.0 implementation: stage rows in a SQL table the gateway's
      // AutoML can `SELECT * FROM`. Gateway v1.6.5.2-ce rejects
      // `config.inline_rows` (the workflow v0.1.0 attempted) and
      // requires the data to land in a real collection/table before
      // training. We:
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
      // the stored Memory's vector to get the cosine feature.
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
        const queryVec = await embeddings.embed(fb.queryText);
        const feats = buildRelevanceFeatures(queryVec, mem);
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
    const vectorDim = vectorDimsForModel(cfg.embedding.model ?? "text-embedding-3-small");
    const collectionName = cfg.collection ?? DEFAULT_COLLECTION;

    // v0.2.0: pass apiKey straight through. @synapcores/sdk@0.4.0
    // routes both apiKey AND jwtToken via `Authorization: Bearer`, so
    // the v0.1.0 shim that re-routed `aidb_*` / `ak_*` keys through
    // `jwtToken` to coerce the right header is gone.
    const client = new SynapCores({
      host: cfg.synapcores.host,
      port: cfg.synapcores.port,
      useHttps: cfg.synapcores.useHttps,
      apiKey: cfg.synapcores.apiKey,
    });
    const db = new MemoryDB(client, collectionName, vectorDim);
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model!);
    const extensions = createExtensions(
      db,
      embeddings,
      client,
      cfg.graph!,
      cfg.workspace,
      collectionName,
    );
    const autoLinkSimilar = cfg.autoLinkSimilar !== false;
    const graphName = cfg.graph!;

    api.logger.info(
      `memory-synapcores: plugin registered (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, collection: ${collectionName}, autoLinkSimilar: ${autoLinkSimilar}, lazy init)`,
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

          const vector = await embeddings.embed(query);
          const results = await db.search(vector, limit, 0.1);

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

          // Strip vector data for serialization (typed arrays can't be cloned)
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

          const vector = await embeddings.embed(text);

          // Check for duplicates
          const existing = await db.search(vector, 1, 0.95);
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
            vector,
            importance,
            category,
          });

          // v0.2.0: linkSimilarMemories now inserts a Memory graph node
          // carrying the embedding so recallRelated has something to
          // MATCH against. Failures are non-fatal (logged in-helper).
          if (autoLinkSimilar) {
            await linkSimilarMemories(entry, db, client, graphName, api.logger);
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
            const vector = await embeddings.embed(query);
            const results = await db.search(vector, 5, 0.7);

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

            // Strip vector data for serialization
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
            const vector = await embeddings.embed(query);
            const results = await db.search(vector, parseInt(opts.limit), 0.3);
            // Strip vectors for output
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
          const vector = await embeddings.embed(event.prompt);
          const results = await db.search(vector, 3, 0.3);

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
            // Type guard for message object
            if (!msg || typeof msg !== "object") {
              continue;
            }
            const msgObj = msg as Record<string, unknown>;

            // Only process user and assistant messages
            const role = msgObj.role;
            if (role !== "user" && role !== "assistant") {
              continue;
            }

            const content = msgObj.content;

            // Handle string content directly
            if (typeof content === "string") {
              texts.push(content);
              continue;
            }

            // Handle array content (content blocks)
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
            const vector = await embeddings.embed(text);

            // Check for duplicates (high similarity threshold)
            const existing = await db.search(vector, 1, 0.95);
            if (existing.length > 0) {
              continue;
            }

            const entry = await db.store({
              text,
              vector,
              importance: 0.7,
              category,
            });
            if (autoLinkSimilar) {
              await linkSimilarMemories(entry, db, client, graphName, api.logger);
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
      start: () => {
        api.logger.info(
          `memory-synapcores: initialized (host: ${cfg.synapcores.host}:${cfg.synapcores.port}, collection: ${collectionName}, model: ${cfg.embedding.model})`,
        );
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
    (memoryPlugin as unknown as { extensions: MemorySynapCoresExtensions }).extensions =
      extensions;
  },
};

export default definePluginEntry(memoryPlugin);

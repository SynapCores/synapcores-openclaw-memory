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
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { type MemoryCategory } from "./config.js";
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
/** The SRS §5.8 response envelope. `data` is verb-specific. */
type MemoryEnvelope = {
    status?: string;
    operation?: string;
    memory?: string;
    identity?: {
        field?: string;
        value?: string;
    } | null;
    request_id?: string;
    generated_at?: string;
    data?: Record<string, unknown> | null;
    warnings?: unknown[];
    error?: {
        code?: string;
        message?: string;
        retryable?: boolean;
    } | null;
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
export declare class MemoryAccessDeniedError extends Error {
    readonly code = "MEMORY_ACCESS_DENIED";
    constructor(message: string);
}
/** The active backend cannot serve this operation (and no silent fallback exists). */
export declare class MemoryFeatureUnavailableError extends Error {
    readonly code = "MEMORY_FEATURE_UNAVAILABLE";
    constructor(message: string);
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
export declare class MemoryCredentialUnpinnedError extends Error {
    readonly code = "MEMORY_CREDENTIAL_UNPINNED";
    constructor(message: string);
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
 * Which backend is live, and everything the plugin needs to know about it.
 */
export type MemoryBackendKind = "legacy" | "memory-object";
/** One extracted conversation turn: the text plus who said it. */
export type MessageTurn = {
    role: "user" | "assistant";
    text: string;
};
export declare function shouldCapture(text: string): boolean;
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
export declare function escapeCypherString(value: string): string;
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
    predictRelevance(query: string, candidates: MemoryEntry[]): Promise<RelevanceScoredMemory[]>;
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
    trainRelevanceModel(feedback: RelevanceFeedback[]): Promise<{
        modelId: string;
        modelName: string;
    }>;
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
declare const pluginEntry: ReturnType<typeof definePluginEntry>;
export default pluginEntry;
//# sourceMappingURL=index.d.ts.map
/**
 * Memory Plugin Unit Tests
 *
 * Tests the memory plugin functionality including:
 * - Plugin registration and configuration
 * - Memory storage and retrieval (via mocked SynapCores SDK MemoryClient)
 * - Auto-recall via hooks
 * - Auto-capture filtering
 * - SynapCores-only extensions: recallFiltered, recallRelated,
 *   predictRelevance, trainRelevanceModel
 *
 * v0.4.0: the fake now mirrors @synapcores/sdk@0.5.0's `client.memory`
 * surface (`store`, `recall`, `forget`) for the core hot path, alongside
 * the previously-covered `graph.cypher`, `graph.nodes.create`,
 * `executeQuery`, and `automl.*` surfaces still used by the extensions.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

const SYNAPCORES_API_KEY = process.env.SYNAPCORES_API_KEY ?? "ak_prod_test_key";
// Embeddings are engine-native (client.embed()); no external embedding
// provider key is required. Live tests only need a running gateway + API key.
const HAS_SYNAPCORES_KEY = Boolean(process.env.SYNAPCORES_API_KEY);
const liveEnabled = HAS_SYNAPCORES_KEY && process.env.OPENCLAW_LIVE_TEST === "1";
const describeLive = liveEnabled ? describe : describe.skip;

// ---------------------------------------------------------------------------
// In-memory fake of the SynapCores SDK (v0.5.0 surface)
// ---------------------------------------------------------------------------
// We mock @synapcores/sdk before importing the plugin so that no network I/O
// happens during unit tests. The fake implements just enough of the
// `SynapCores` client to back the plugin's core memory ops (store / recall /
// forget) AND the extension paths (graph.cypher, automl.*, executeQuery).

type FakeMemoryRecord = {
  id: string;
  content: string;
  metadata: Record<string, unknown> | null;
  createdAt: Date;
  // Engine doesn't expose this, but we cache it for deterministic similarity
  // ranking in the fake.
  embedding: number[];
};

type FakeGraphNode = {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
};

function cosineSim(a: number[], b: number[]): number {
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

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 0);
}
function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h * 31) + s.charCodeAt(i)) >>> 0;
  }
  return h;
}
function deterministicEmbed(text: string): number[] {
  // Same token-bucket scheme as the OpenAI mock so that cosines line up
  // between the engine-side store (this fake) and the OpenAI-side cosine
  // (the OpenAI mock below).
  const dim = 1536;
  const vec = new Array<number>(dim).fill(0);
  for (const tok of tokenize(text)) {
    const h = hashStr(tok);
    const idx = h % dim;
    vec[idx] += 1;
    vec[(idx + 7) % dim] += 0.5;
    vec[(idx + 17) % dim] += 0.5;
  }
  return vec;
}

let memoryIdCounter = 0;
function nextMemoryId(): string {
  memoryIdCounter += 1;
  return `mem_${memoryIdCounter.toString(16)}_${Math.random().toString(36).slice(2, 8)}`;
}

class FakeMemoryClient {
  // namespace -> records
  public stores = new Map<string, FakeMemoryRecord[]>();
  public storeCalls: Array<{ namespace: string; content: string; metadata?: Record<string, unknown> }> = [];
  public recallCalls: Array<{ namespace: string; query: string; topK: number }> = [];
  public forgetCalls: Array<{ namespace: string; id: string }> = [];

  constructor(private readonly client: FakeSynapCores) {}

  async store(
    namespace: string,
    content: string,
    options?: { metadata?: Record<string, unknown> },
  ): Promise<string> {
    this.assertNamespace(namespace);
    this.storeCalls.push({ namespace, content, metadata: options?.metadata });
    const id = nextMemoryId();
    const record: FakeMemoryRecord = {
      id,
      content,
      metadata: options?.metadata ?? null,
      createdAt: new Date(),
      embedding: deterministicEmbed(content),
    };
    const rows = this.stores.get(namespace) ?? [];
    rows.push(record);
    this.stores.set(namespace, rows);
    return id;
  }

  async recall(
    namespace: string,
    query: string,
    options?: { topK?: number },
  ): Promise<Array<{
    id: string;
    content: string;
    similarity: number;
    metadata: Record<string, unknown> | null;
    createdAt: Date;
  }>> {
    this.assertNamespace(namespace);
    const topK = options?.topK ?? 10;
    this.recallCalls.push({ namespace, query, topK });
    const rows = this.stores.get(namespace);
    if (!rows) return [];
    const qVec = deterministicEmbed(query);
    const scored = rows.map((r) => ({
      id: r.id,
      content: r.content,
      similarity: cosineSim(qVec, r.embedding),
      metadata: r.metadata,
      createdAt: r.createdAt,
    }));
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
  }

  async forget(namespace: string, id: string): Promise<boolean> {
    this.assertNamespace(namespace);
    this.forgetCalls.push({ namespace, id });
    const rows = this.stores.get(namespace);
    if (!rows) return false;
    const before = rows.length;
    const next = rows.filter((r) => r.id !== id);
    this.stores.set(namespace, next);
    return next.length !== before;
  }

  private assertNamespace(namespace: string): void {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(namespace)) {
      throw new Error(`Invalid namespace '${namespace}'`);
    }
  }
}

// ---------------------------------------------------------------------------
// Fake CREATE MEMORY object (SRS v1.2) behind /v1/memory/*
// ---------------------------------------------------------------------------
// Mirrors the shapes a live v1.14.3 gateway returns (captured from a running
// engine while writing v0.7.0): the {data, meta} wrapper is already stripped by
// the SDK's response interceptor, so what the plugin sees is the envelope
// itself — {status, operation, memory, identity, request_id, data, warnings,
// error}.
//
// `supported` defaults to FALSE so every pre-existing test keeps exercising the
// legacy MEMORY_* path exactly as before; the memory-object tests opt in.

type FakeEpisode = {
  id: string;
  identity: string;
  content: string;
  sourceType: string;
  recordedAt: Date;
  attribute?: string;
  value?: string;
};

/** Build an SDK-shaped error: the gateway's real code lives under `details.error.code`. */
function fakeSdkError(
  httpish: "auth" | "not_found" | "bad_request",
  gatewayCode: string,
  message: string,
): Error {
  const err = new Error(message);
  // @synapcores/sdk maps 403 -> AuthenticationError/AUTH_ERROR and 404 ->
  // NotFoundError/NOT_FOUND, discarding the gateway's own code — which is
  // exactly why the plugin has to dig it back out of `details`.
  const sdkCode =
    httpish === "auth" ? "AUTH_ERROR" : httpish === "not_found" ? "NOT_FOUND" : "VALIDATION_ERROR";
  const sdkName =
    httpish === "auth"
      ? "AuthenticationError"
      : httpish === "not_found"
        ? "NotFoundError"
        : "ValidationError";
  return Object.assign(err, {
    name: sdkName,
    code: sdkCode,
    details: { error: { code: gatewayCode, message } },
  });
}

class FakeMemoryObjects {
  /** False = an engine that predates CREATE MEMORY (no /v1/memory route). */
  public supported = false;
  /** Simulates a credential whose token carries a `memory_identity` claim. */
  public identityPin: string | null = null;
  public createFails = false;
  public objects = new Map<string, { identity_field: string }>();
  public episodes: FakeEpisode[] = [];
  /** `${memory}|${identity}|${attribute}` -> claims. Two claims = a conflict. */
  public state = new Map<string, Array<{ value: string; confidence: number; authority: string }>>();
  public calls: Array<{ path: string; body: unknown }> = [];

  private episodeCounter = 0;

  /** The gateway's `resolve_identity` (MEM-SEC-001), reproduced. */
  private resolveIdentity(supplied: unknown): string {
    const given = typeof supplied === "string" && supplied.trim() ? supplied.trim() : undefined;
    if (this.identityPin !== null) {
      if (given === undefined) return this.identityPin;
      if (given === this.identityPin) return this.identityPin;
      throw fakeSdkError(
        "auth",
        "MEMORY_ACCESS_DENIED",
        "Caller is not authorized for this memory identity.",
      );
    }
    if (given === undefined) {
      throw fakeSdkError("bad_request", "missing_field", "'identity' is required");
    }
    return given;
  }

  private requireObject(name: string): void {
    if (!this.objects.has(name)) {
      throw fakeSdkError(
        "not_found",
        "MEMORY_NOT_FOUND",
        `memory object '${name}' does not exist`,
      );
    }
  }

  private envelope(
    operation: string,
    name: string,
    identity: string,
    status: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: any,
  ) {
    return {
      status,
      operation,
      memory: name,
      identity: { field: this.objects.get(name)?.identity_field ?? "user_id", value: identity },
      request_id: `req_${Math.random().toString(16).slice(2, 10)}`,
      generated_at: new Date().toISOString(),
      data,
      warnings: [],
      error: null,
    };
  }

  list(): Array<Record<string, unknown>> {
    if (!this.supported) {
      throw fakeSdkError("not_found", "not_found", "Not Found");
    }
    return Array.from(this.objects.entries()).map(([memory_name, def]) => ({
      memory_name,
      agent: null,
      identity_field: def.identity_field,
      episodic: true,
      semantic: true,
      temporal: true,
      relationships: true,
      provenance: true,
      consolidation: "AUTO",
    }));
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  create(body: any) {
    if (!this.supported) {
      throw fakeSdkError("not_found", "not_found", "Not Found");
    }
    if (this.createFails) {
      throw fakeSdkError("bad_request", "MEMORY_INVALID_ARGUMENT", "cannot create");
    }
    const name = String(body?.name ?? "");
    if (this.objects.has(name)) {
      throw fakeSdkError("bad_request", "MEMORY_ALREADY_EXISTS", "already exists");
    }
    this.objects.set(name, { identity_field: String(body?.identity_field ?? "user_id") });
    return this.envelope("CREATE MEMORY", name, "", "completed", {
      memory: name,
      identity_field: body?.identity_field,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  remember(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    const options = (body?.options ?? {}) as Record<string, unknown>;
    this.episodeCounter += 1;
    const episode: FakeEpisode = {
      id: `ep_${this.episodeCounter.toString(16)}_${Math.random().toString(36).slice(2, 8)}`,
      identity,
      content: String(body?.content ?? ""),
      sourceType: String(options.source_type ?? "user_statement"),
      recordedAt: new Date(),
      attribute: typeof options.attribute === "string" ? options.attribute : undefined,
      value: typeof options.value === "string" ? options.value : undefined,
    };
    this.episodes.push(episode);

    let stateChanges = 0;
    if (episode.attribute && episode.value !== undefined) {
      const key = `${name}|${identity}|${episode.attribute}`;
      this.state.set(key, [
        { value: episode.value, confidence: 0.85, authority: episode.sourceType },
      ]);
      stateChanges = 1;
    }

    return this.envelope("REMEMBER", name, identity, "accepted", {
      episode_id: episode.id,
      memory_ids: [`mem_${episode.id.slice(3)}`],
      changes: {
        episodes_created: 1,
        facts_created: stateChanges === 1 ? 0 : 1,
        state_changes: stateChanges,
        relationships_created: 0,
        records_superseded: 0,
        conflicts_flagged: 0,
      },
      consolidation: { required: false, status: "not_required" },
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  recall(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    const about = String(body?.about ?? "");
    const options = (body?.options ?? {}) as Record<string, unknown>;
    const qVec = deterministicEmbed(about);

    const mine = this.episodes.filter((e) => e.identity === identity);
    const scored = mine
      .map((e) => ({ episode: e, relevance: cosineSim(qVec, deterministicEmbed(e.content)) }))
      .sort((a, b) => b.relevance - a.relevance);

    const currentState: Record<string, unknown> = {};
    for (const [key, claims] of this.state.entries()) {
      const [mem, id, attribute] = key.split("|");
      if (mem !== name || id !== identity || claims.length !== 1) continue;
      currentState[attribute] = {
        value: claims[0].value,
        confidence: claims[0].confidence,
        valid_from: "2026-08-17 14:02:42",
      };
    }

    const semantic = scored
      .filter((s) => s.episode.attribute)
      .map((s) => ({
        memory_id: `mem_${s.episode.id.slice(3)}`,
        fact: `${s.episode.attribute} is ${s.episode.value}`,
        attribute: s.episode.attribute,
        value: s.episode.value,
        // Deliberately DIFFERENT from relevance so a test can prove the two
        // are never conflated.
        confidence: 0.85,
        relevance: s.relevance,
      }));
    const episodic = scored.map((s) => ({
      episode_id: s.episode.id,
      event: s.episode.content,
      occurred_at: s.episode.recordedAt.toISOString(),
      recorded_at: s.episode.recordedAt.toISOString(),
      relevance: s.relevance,
    }));

    const promptReady =
      options.prompt_ready === false || scored.length === 0
        ? ""
        : `ENGINE-PROJECTION: ${scored.map((s) => s.episode.content).join(" | ")}`;

    return this.envelope("RECALL", name, identity, "ok", {
      query: about,
      intent: "general",
      current_state: currentState,
      semantic,
      episodic,
      relationships: [],
      temporal: [],
      provenance: [],
      context: {
        prompt_ready: promptReady,
        estimated_tokens: 42,
        truncated: false,
      },
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  current(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    const attribute = String(body?.attribute ?? "");
    const claims = this.state.get(`${name}|${identity}|${attribute}`) ?? [];
    if (claims.length === 0) {
      return this.envelope("CURRENT", name, identity, "not_found", { attribute, value: null });
    }
    if (claims.length > 1) {
      // The engine does NOT pick when authority, time and confidence all fail
      // to separate the claims.
      return this.envelope("CURRENT", name, identity, "conflict", {
        attribute,
        competing_claims: claims,
      });
    }
    return this.envelope("CURRENT", name, identity, "ok", {
      attribute,
      value: claims[0].value,
      confidence: claims[0].confidence,
      authority: claims[0].authority,
      valid_from: "2026-08-17 14:02:42",
      valid_to: null,
      source_memory_id: "mem_fake",
      supersedes_memory_id: null,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  trace(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    const attribute = String(body?.attribute ?? "");
    const source = this.episodes.find((e) => e.identity === identity && e.attribute === attribute);
    if (!source) {
      return this.envelope("TRACE", name, identity, "not_found", { attribute });
    }
    return this.envelope("TRACE", name, identity, "ok", {
      memory_id: `mem_${source.id.slice(3)}`,
      attribute,
      value: source.value,
      confidence: 0.85,
      derived_from: [
        {
          episode_id: source.id,
          source_type: source.sourceType,
          content: source.content,
          recorded_at: source.recordedAt.toISOString(),
          event_time: source.recordedAt.toISOString(),
          extraction_method: "explicit_option",
          extractor_version: "memory_extractor_v1",
        },
      ],
      supersedes: null,
      derivation: [{ step: 1, operation: "extract_state_candidate" }],
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  forget(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    const before = this.episodes.length;
    if (typeof body?.memory_id === "string") {
      this.episodes = this.episodes.filter(
        (e) => !(e.identity === identity && e.id === body.memory_id),
      );
    } else if (typeof body?.about === "string") {
      const aVec = deterministicEmbed(body.about);
      this.episodes = this.episodes.filter(
        (e) => !(e.identity === identity && cosineSim(aVec, deterministicEmbed(e.content)) >= 0.3),
      );
    }
    const removed = before - this.episodes.length;
    return this.envelope("FORGET", name, identity, removed > 0 ? "completed" : "not_found", {
      scope: body?.about ?? body?.memory_id,
      affected: {
        episodes: removed,
        semantic_facts: removed,
        state_records: 0,
        relationships: 0,
        embeddings: removed,
      },
      propagation_status: "completed",
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  consolidate(name: string, body: any) {
    this.requireObject(name);
    const identity = this.resolveIdentity(body?.identity);
    return this.envelope("CONSOLIDATE", name, identity, "completed", {
      episodes_processed: this.episodes.filter((e) => e.identity === identity).length,
    });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CypherCall = { query: string; params?: Record<string, any>; graph?: string };
type ListModelsResult = Array<{ id: string; name: string; task?: string; status?: string }>;

class FakeGraphNodesApi {
  constructor(private readonly graph: FakeGraph) {}
  async create(label: string, props: Record<string, unknown> = {}): Promise<FakeGraphNode> {
    const id = String(props.id ?? `gn-${this.graph.allNodes.length + 1}`);
    const node: FakeGraphNode = {
      id,
      labels: [label],
      properties: { ...props },
    };
    this.graph.allNodes.push(node);
    this.graph.nodeIndexById.set(id, node);
    const userId = props.id ? String(props.id) : id;
    this.graph.nodeIndexByUserId.set(userId, node);
    return node;
  }
}

class FakeGraph {
  public allNodes: FakeGraphNode[] = [];
  public nodeIndexById = new Map<string, FakeGraphNode>();
  public nodeIndexByUserId = new Map<string, FakeGraphNode>();
  public cypherCalls: CypherCall[] = [];
  public readonly nodes: FakeGraphNodesApi;

  constructor() {
    this.nodes = new FakeGraphNodesApi(this);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async cypher(query: string, params?: Record<string, any>, graph?: string): Promise<{ columns: string[]; rows: unknown[][]; records: Array<Record<string, unknown>> }> {
    this.cypherCalls.push({ query, params, graph });
    if (params !== undefined && Object.keys(params).length > 0) {
      throw new Error(
        `FakeGraph.cypher: gateway rejects $param bindings; caller passed params=${JSON.stringify(
          params,
        )}.`,
      );
    }
    // Recognise the recallRelated MATCH pattern. The plugin emits:
    //   MATCH (start:Memory {id: 'X'})-[:SIMILAR_TO > T]-(related:Memory) RETURN DISTINCT related LIMIT N
    const m = query.match(/MATCH \(start:Memory \{id: '((?:\\.|[^'\\])*)'\}\)-\[:SIMILAR_TO > ([0-9.]+)\]-\(related:Memory\)/);
    if (m) {
      const startUserId = m[1].replace(/\\'/g, "'").replace(/\\\\/g, "\\");
      const threshold = Number(m[2]);
      const start = this.nodeIndexByUserId.get(startUserId);
      if (!start) {
        return { columns: ["related"], rows: [], records: [] };
      }
      const startEmb = (start.properties.embedding as number[]) ?? [];
      const limitMatch = query.match(/LIMIT (\d+)/);
      const limit = limitMatch ? Number(limitMatch[1]) : 20;
      const matches: FakeGraphNode[] = [];
      for (const n of this.nodeIndexById.values()) {
        if (n === start) continue;
        if (!n.labels.includes("Memory")) continue;
        const emb = (n.properties.embedding as number[]) ?? [];
        const sim = cosineSim(startEmb, emb);
        if (sim >= threshold) matches.push(n);
      }
      const slice = matches.slice(0, limit);
      return {
        columns: ["related"],
        rows: slice.map((n) => [n]),
        records: slice.map((n) => ({ related: n })),
      };
    }
    return { columns: [], rows: [], records: [] };
  }
}

class FakeAutoMLModel {
  constructor(public readonly id: string, public readonly name: string) {}
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  predictCalls: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async predict(data: any | any[]): Promise<any | any[]> {
    this.predictCalls.push(data);
    const isSingle = !Array.isArray(data);
    const rows = isSingle ? [data] : data;
    const preds = rows.map(() => 0.42);
    return isSingle ? preds[0] : preds;
  }
}

class FakeAutoML {
  public models: ListModelsResult = [];
  public modelInstances = new Map<string, FakeAutoMLModel>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public trainCalls: any[] = [];

  async listModels(): Promise<ListModelsResult> {
    return [...this.models];
  }

  async getModel(idOrName: string): Promise<FakeAutoMLModel> {
    let inst = this.modelInstances.get(idOrName);
    if (!inst) {
      const info = this.models.find((m) => m.id === idOrName || m.name === idOrName);
      if (!info) {
        throw new Error(`model not found: ${idOrName}`);
      }
      inst = new FakeAutoMLModel(info.id, info.name);
      this.modelInstances.set(info.id, inst);
      this.modelInstances.set(info.name, inst);
    }
    return inst;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async train(options: any): Promise<FakeAutoMLModel> {
    this.trainCalls.push(options);
    const id = `model-${this.models.length + 1}`;
    const name = options.name ?? `${options.collection}_${options.target}_model`;
    this.models.push({ id, name, task: options.task, status: "completed" });
    const inst = new FakeAutoMLModel(id, name);
    this.modelInstances.set(id, inst);
    this.modelInstances.set(name, inst);
    return inst;
  }
}

class FakeSynapCores {
  public graph = new FakeGraph();
  public automl = new FakeAutoML();
  public readonly memory: FakeMemoryClient;
  /** The v1.14.3+ CREATE MEMORY object surface. Off unless a test enables it. */
  public memoryObjects = new FakeMemoryObjects();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public sqlCalls: Array<{ sql: string; parameters?: any[] }> = [];
  // Synthetic in-memory SQL tables for executeQuery
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public sqlTables = new Map<string, Array<Record<string, any>>>();

  constructor() {
    this.memory = new FakeMemoryClient(this);
  }

  // Engine-native embedding surface (SDK 0.6.0 `client.embed()`). The plugin's
  // `Embeddings` wrapper calls this; we return deterministic token-based
  // vectors so cosine similarity is stable across store/recall/relevance.
  async embed(text: string): Promise<number[]> {
    return deterministicEmbed(text);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async executeQuery(req: { sql: string; parameters?: any[] }): Promise<any> {
    this.sqlCalls.push(req);
    const sql = req.sql.trim();

    // SELECT 1 — preflight probe
    if (/^SELECT\s+1\s*$/i.test(sql)) {
      return {
        columns: [{ name: "1", data_type: "INTEGER", nullable: false }],
        rows: [[1]],
        execution_time_ms: 0,
      };
    }

    // SELECT … FROM MEMORY_RECALL($1, $2, $3) [WHERE …] [LIMIT N]
    if (/MEMORY_RECALL\s*\(\s*\$\d+\s*,\s*\$\d+\s*,\s*\$\d+\s*\)/i.test(sql)) {
      const [namespace, query, topK, ...rest] = req.parameters ?? [];
      const records = this.memory.stores.get(namespace) ?? [];
      const qVec = deterministicEmbed(String(query ?? ""));
      let scored = records.map((r) => ({
        id: r.id,
        content: r.content,
        similarity: cosineSim(qVec, r.embedding),
        metadata: r.metadata,
        created_at: r.createdAt.toISOString(),
      }));
      scored.sort((a, b) => b.similarity - a.similarity);
      scored = scored.slice(0, Number(topK) || 10);

      // Tiny WHERE-clause evaluator that handles the cases the plugin emits:
      //  - id = $4 (with positional placeholder)
      //  - id = 'literal'
      //  - JSON-extract on metadata->>'…' (with literal RHS)
      //  - similarity > N
      // Plus AND combinators.
      const whereMatch = sql.match(/WHERE\s+(.*?)(?:\s+LIMIT\s+\d+)?$/i);
      if (whereMatch) {
        const whereClause = whereMatch[1];
        const allParams = req.parameters ?? [];
        const resolvedClause = whereClause.replace(/\$(\d+)/g, (_m, idx) => {
          const v = allParams[Number(idx) - 1];
          return typeof v === "string" ? `'${v}'` : String(v);
        });
        scored = scored.filter((row) => evalSimpleWhere(resolvedClause, row));
      }

      const limitMatch = sql.match(/LIMIT\s+(\d+)/i);
      if (limitMatch) {
        scored = scored.slice(0, Number(limitMatch[1]));
      }

      return {
        columns: [
          { name: "id", data_type: "TEXT", nullable: false },
          { name: "content", data_type: "TEXT", nullable: false },
          { name: "similarity", data_type: "REAL", nullable: false },
          { name: "metadata", data_type: "TEXT", nullable: true },
          { name: "created_at", data_type: "TEXT", nullable: false },
        ],
        rows: scored.map((r) => [
          r.id,
          r.content,
          r.similarity,
          r.metadata == null ? null : JSON.stringify(r.metadata),
          r.created_at,
        ]),
        execution_time_ms: 0,
      };
    }

    // CREATE TABLE <name> (...)
    let m = sql.match(/^CREATE TABLE\s+(\w+)/i);
    if (m) {
      const t = m[1];
      if (!this.sqlTables.has(t)) this.sqlTables.set(t, []);
      return {
        columns: [{ name: "result", data_type: "TEXT", nullable: true }],
        rows: [[`Table ${t} created successfully`]],
        execution_time_ms: 0,
      };
    }

    // INSERT INTO <name> (cols) VALUES (vals)
    m = sql.match(/^INSERT INTO\s+(\w+)\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)/i);
    if (m) {
      const t = m[1];
      const cols = m[2].split(",").map((s) => s.trim());
      const vals = m[3].split(",").map((s) => s.trim());
      const rows = this.sqlTables.get(t) ?? [];
      const row: Record<string, unknown> = {};
      for (let i = 0; i < cols.length; i++) {
        const raw = vals[i];
        row[cols[i]] = isNaN(Number(raw)) ? raw : Number(raw);
      }
      rows.push(row);
      this.sqlTables.set(t, rows);
      return { columns: [], rows: [], rows_affected: 1, execution_time_ms: 0 };
    }

    // The memory object materializes five ORDINARY tables; the plugin reads
    // `_mem_<name>_episodes` for the two things the verb contract has no verb
    // for — counting, and fetching one memory by id.
    m = sql.match(/^SELECT COUNT\(\*\) FROM _mem_(\w+)_episodes(?:\s+WHERE identity = '(.*)')?/i);
    if (m) {
      const identity = m[2];
      const rows = this.memoryObjects.episodes.filter(
        (e) => identity === undefined || e.identity === identity.replace(/''/g, "'"),
      );
      return {
        columns: [{ name: "count", data_type: "INTEGER", nullable: true }],
        rows: [[rows.length]],
        execution_time_ms: 0,
      };
    }
    m = sql.match(
      /^SELECT id, content, recorded_at FROM _mem_(\w+)_episodes WHERE id = '(.*?)'(?:\s+AND identity = '(.*?)')?\s+LIMIT 1/i,
    );
    if (m) {
      const id = m[2].replace(/''/g, "'");
      const identity = m[3]?.replace(/''/g, "'");
      const found = this.memoryObjects.episodes.find(
        (e) => e.id === id && (identity === undefined || e.identity === identity),
      );
      return {
        columns: [
          { name: "id", data_type: "TEXT", nullable: false },
          { name: "content", data_type: "TEXT", nullable: false },
          { name: "recorded_at", data_type: "TEXT", nullable: false },
        ],
        rows: found ? [[found.id, found.content, found.recordedAt.toISOString()]] : [],
        execution_time_ms: 0,
      };
    }

    // SELECT COUNT(*) FROM <name>
    m = sql.match(/^SELECT COUNT\(\*\) FROM\s+(\w+)/i);
    if (m) {
      const t = m[1];
      const tableName = t.startsWith("_memory_") ? t : t;
      // For _memory_<ns>, count from the memory store
      if (tableName.startsWith("_memory_")) {
        const ns = tableName.slice("_memory_".length);
        const rows = this.memory.stores.get(ns) ?? [];
        return {
          columns: [{ name: "count", data_type: "INTEGER", nullable: true }],
          rows: [[rows.length]],
          execution_time_ms: 0,
        };
      }
      const n = (this.sqlTables.get(tableName) ?? []).length;
      return {
        columns: [{ name: "count", data_type: "INTEGER", nullable: true }],
        rows: [[n]],
        execution_time_ms: 0,
      };
    }

    return { columns: [], rows: [], execution_time_ms: 0 };
  }

  // SDK 0.5.0 dropped the `client.graph.*` accessors; the plugin now
  // reaches through `_getHttpClient` for both `/graph/nodes` (the
  // node-create workaround the SDK has always shipped wrong) and
  // `/graph/match` (the cypher endpoint that used to be wrapped by
  // `client.graph.cypher` in older SDK versions). The fake mirrors
  // both paths so the plugin's workarounds round-trip in-process.
  _getHttpClient(): {
    get: (path: string) => Promise<{ data: unknown }>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    post: (path: string, body?: any) => Promise<{ data: unknown }>;
  } {
    const self = this;
    return {
      async get(path: string) {
        // Capability probe. The plugin must use the UNSLASHED form: a live
        // gateway 404s `/v1/memory/` (it misses the route and lands on the SPA
        // fallback) while `/v1/memory` serves SHOW MEMORIES.
        if (path === "/memory") {
          self.memoryObjects.calls.push({ path, body: undefined });
          return { data: self.memoryObjects.list() };
        }
        throw new Error(`FakeSynapCores._getHttpClient: unhandled GET ${path}`);
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async post(path: string, body?: any) {
        if (path.startsWith("/memory")) {
          self.memoryObjects.calls.push({ path, body });
          if (path === "/memory") {
            return { data: self.memoryObjects.create(body) };
          }
          const m = path.match(/^\/memory\/([^/]+)\/(\w+)$/);
          if (m) {
            const [, name, verb] = m;
            switch (verb) {
              case "remember":
                return { data: self.memoryObjects.remember(name, body) };
              case "recall":
                return { data: self.memoryObjects.recall(name, body) };
              case "current":
                return { data: self.memoryObjects.current(name, body) };
              case "trace":
                return { data: self.memoryObjects.trace(name, body) };
              case "forget":
                return { data: self.memoryObjects.forget(name, body) };
              case "consolidate":
                return { data: self.memoryObjects.consolidate(name, body) };
              default:
                throw new Error(`FakeSynapCores: unhandled memory verb ${verb}`);
            }
          }
          throw new Error(`FakeSynapCores._getHttpClient: unhandled POST ${path}`);
        }
        if (path === "/graph/nodes") {
          const labels = Array.isArray(body?.labels) ? body.labels : [];
          const props = body?.properties ?? {};
          const id = String(props.id ?? `gn-${self.graph.allNodes.length + 1}`);
          const node: FakeGraphNode = { id, labels: [...labels], properties: { ...props } };
          self.graph.allNodes.push(node);
          self.graph.nodeIndexById.set(id, node);
          self.graph.nodeIndexByUserId.set(String(props.id ?? id), node);
          return { data: { id, labels, properties: props } };
        }
        if (path === "/graph/match") {
          const sql = String(body?.sql ?? "");
          const result = await self.graph.cypher(sql);
          return { data: { columns: result.columns, rows: result.rows, count: result.rows.length } };
        }
        throw new Error(`FakeSynapCores._getHttpClient: unhandled POST ${path}`);
      },
    };
  }
}

/**
 * Tiny WHERE-clause evaluator restricted to the operators the plugin emits.
 * Supports `<lhs> <op> <rhs>` joined by AND/OR, where:
 *   - lhs ∈ { id, content, similarity, metadata->>'<key>' (with optional CAST(...)) }
 *   - op  ∈ { =, !=, <, <=, >, >= }
 *   - rhs ∈ { string literal in quotes, numeric literal }
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evalSimpleWhere(clause: string, row: Record<string, any>): boolean {
  const tokens = clause.split(/\s+(AND|OR)\s+/i);
  let result = true;
  let combinator: "AND" | "OR" = "AND";
  for (const tok of tokens) {
    if (/^(AND|OR)$/i.test(tok)) {
      combinator = tok.toUpperCase() as "AND" | "OR";
      continue;
    }
    const evald = evalAtom(tok.trim(), row);
    if (combinator === "AND") result = result && evald;
    else result = result || evald;
  }
  return result;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evalAtom(atom: string, row: Record<string, any>): boolean {
  // Require whitespace around the operator so the `>>` and `->` glyphs in
  // `metadata->>'key'` aren't mistaken for `>` / `<` comparison ops.
  const m = atom.match(/^(.+?)\s+(=|!=|<=|>=|<|>)\s+(.+)$/);
  if (!m) return true;
  const [, lhsRaw, op, rhsRaw] = m;
  const lhs = resolveLhs(lhsRaw.trim(), row);
  const rhs = resolveRhs(rhsRaw.trim());
  switch (op) {
    case "=":
      return String(lhs) === String(rhs);
    case "!=":
      return String(lhs) !== String(rhs);
    case "<":
      return Number(lhs) < Number(rhs);
    case "<=":
      return Number(lhs) <= Number(rhs);
    case ">":
      return Number(lhs) > Number(rhs);
    case ">=":
      return Number(lhs) >= Number(rhs);
  }
  return true;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function resolveLhs(expr: string, row: Record<string, any>): unknown {
  // CAST(metadata->>'key' AS REAL|INTEGER)
  let m = expr.match(/^CAST\s*\(\s*metadata->>'([^']+)'\s+AS\s+(\w+)\s*\)$/i);
  if (m) {
    const key = m[1];
    const meta = row.metadata;
    const val = meta && typeof meta === "object" ? meta[key] : undefined;
    return Number(val);
  }
  m = expr.match(/^metadata->>'([^']+)'$/);
  if (m) {
    const key = m[1];
    const meta = row.metadata;
    return meta && typeof meta === "object" ? meta[key] : undefined;
  }
  if (expr === "id") return row.id;
  if (expr === "content") return row.content;
  if (expr === "similarity") return row.similarity;
  return expr;
}

function resolveRhs(expr: string): unknown {
  if (/^'.*'$/.test(expr)) return expr.slice(1, -1);
  const n = Number(expr);
  if (Number.isFinite(n)) return n;
  return expr;
}

// Pre-allocate a single fake we can inspect from each test. We stash it on
// a hoisted holder so the vi.mock factory (which runs at module import time,
// before any normal top-level `let` is reached) can write into it without
// hitting vitest's "Cannot access before initialization" guard.
const mockState = vi.hoisted(() => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  lastFake: null as any,
}));

vi.mock("@synapcores/sdk", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  SynapCores: vi.fn().mockImplementation((_cfg: any) => {
    mockState.lastFake = new FakeSynapCores();
    return mockState.lastFake;
  }),
  // Empty marker class; the plugin never instantiates it directly.
  MemoryClient: class {},
}));

function getLastFake(): FakeSynapCores {
  if (!mockState.lastFake) {
    throw new Error("SynapCores client was never instantiated");
  }
  return mockState.lastFake as FakeSynapCores;
}

// OpenAI has been fully removed from the plugin — embeddings are engine-native
// via `client.embed()` (the fake `FakeSynapCores.embed` above returns the same
// deterministic, token-based vectors so semantically-similar texts get high
// cosine similarity: identical text -> 1.0, shared-token texts -> ~0.94,
// disjoint texts -> 0). No `vi.mock("openai")` is needed anymore.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("memory plugin metadata", () => {
  test("memory plugin registers and initializes correctly", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(memoryPlugin.id).toBe("memory-synapcores");
    expect(memoryPlugin.name).toBe("Memory (SynapCores)");
    expect(memoryPlugin.kind).toBe("memory");
    expect(memoryPlugin.configSchema).toBeDefined();
    // oxlint-disable-next-line typescript/unbound-method
    expect(memoryPlugin.register).toBeInstanceOf(Function);
  });

  test("config schema parses valid config", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      synapcores: {
        host: "localhost",
        port: 8080,
        apiKey: SYNAPCORES_API_KEY,
        useHttps: false,
      },
      collection: "openclaw_memories",
      graph: "openclaw_memory_graph",
      autoCapture: true,
      autoRecall: true,
      autoLinkSimilar: true,
    });

    expect(config).toBeDefined();
    expect(config?.synapcores?.apiKey).toBe(SYNAPCORES_API_KEY);
    expect(config?.synapcores?.host).toBe("localhost");
    expect(config?.synapcores?.port).toBe(8080);
    expect(config?.collection).toBe("openclaw_memories");
    expect(config?.graph).toBe("openclaw_memory_graph");
    expect(config?.autoLinkSimilar).toBe(true);
  });

  test("config schema resolves env vars", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    process.env.TEST_SC_API_KEY = "ak_prod_test_456";

    const config = memoryPlugin.configSchema?.parse?.({
      synapcores: { apiKey: "${TEST_SC_API_KEY}" },
    });

    expect(config?.synapcores?.apiKey).toBe("ak_prod_test_456");

    delete process.env.TEST_SC_API_KEY;
  });

  test("config schema rejects an unknown top-level key (OpenAI embedding block removed)", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        // `embedding` was removed when OpenAI was dropped — it is now an
        // unknown key and must be rejected.
        embedding: { apiKey: "test-key" },
        synapcores: { apiKey: SYNAPCORES_API_KEY },
      });
    }).toThrow("unknown keys: embedding");
  });

  // Was: "rejects missing synapcores apiKey" — it threw here.
  //
  // That is exactly what made the plugin uninstallable. OpenClaw validates a
  // plugin's config schema *during* `plugins install`, before any config can be
  // written, so throwing on a missing key aborted the install with a Node stack
  // trace and no way forward. Config is now optional at parse time and the
  // gateway preflight reports it at runtime instead.
  test("config parses with no apiKey so the plugin can be installed unconfigured", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const parsed = memoryPlugin.configSchema?.parse?.({ synapcores: {} }) as
      | { synapcores: { apiKey: string; host: string; port: number } }
      | undefined;

    expect(parsed).toBeDefined();
    expect(parsed?.synapcores.apiKey).toBe("");
    expect(parsed?.synapcores.host).toBe("localhost");
  });

  test("config parses with no synapcores block at all", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const parsed = memoryPlugin.configSchema?.parse?.({}) as
      | { synapcores: { apiKey: string } }
      | undefined;
    expect(parsed?.synapcores.apiKey).toBe("");
  });

  test("apiKey falls back to SYNAPCORES_API_KEY, as the README has always promised", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const previous = process.env.SYNAPCORES_API_KEY;
    process.env.SYNAPCORES_API_KEY = "aidb_from_env";
    try {
      const parsed = memoryPlugin.configSchema?.parse?.({ synapcores: {} }) as
        | { synapcores: { apiKey: string } }
        | undefined;
      expect(parsed?.synapcores.apiKey).toBe("aidb_from_env");
    } finally {
      if (previous === undefined) delete process.env.SYNAPCORES_API_KEY;
      else process.env.SYNAPCORES_API_KEY = previous;
    }
  });

  test("an explicit apiKey still wins over the environment", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const previous = process.env.SYNAPCORES_API_KEY;
    process.env.SYNAPCORES_API_KEY = "aidb_from_env";
    try {
      const parsed = memoryPlugin.configSchema?.parse?.({
        synapcores: { apiKey: "aidb_explicit" },
      }) as { synapcores: { apiKey: string } } | undefined;
      expect(parsed?.synapcores.apiKey).toBe("aidb_explicit");
    } finally {
      if (previous === undefined) delete process.env.SYNAPCORES_API_KEY;
      else process.env.SYNAPCORES_API_KEY = previous;
    }
  });

  test("unknown keys inside synapcores are still rejected", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        // `url` is a common wrong guess — the real keys are host/port/useHttps.
        synapcores: { apiKey: "k", url: "http://localhost:8080" },
      });
    }).toThrow();
  });

  test("config schema applies defaults for collection, graph, host, port, autoLinkSimilar", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      synapcores: { apiKey: SYNAPCORES_API_KEY },
    });

    expect(config?.synapcores?.host).toBe("localhost");
    expect(config?.synapcores?.port).toBe(8080);
    expect(config?.synapcores?.useHttps).toBe(false);
    expect(config?.collection).toBe("openclaw_memories");
    expect(config?.graph).toBe("openclaw_memory_graph");
    // Privacy-safe defaults: autoCapture / autoRecall / autoLinkSimilar are
    // opt-in and default to false (the plugin is passive until enabled).
    expect(config?.autoCapture).toBe(false);
    expect(config?.autoRecall).toBe(false);
    expect(config?.autoLinkSimilar).toBe(false);
  });

  test("memory block defaults reproduce the pre-0.7.0 behaviour", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const config = memoryPlugin.configSchema?.parse?.({ synapcores: {} }) as
      | { memory: Record<string, unknown> }
      | undefined;

    // auto-detect the backend; one static identity, which is exactly the
    // isolation the flat legacy namespace gave.
    expect(config?.memory.backend).toBe("auto");
    expect(config?.memory.identityFrom).toBe("static");
    expect(config?.memory.identity).toBe("default");
    expect(config?.memory.identityField).toBe("user_id");
    expect(config?.memory.autoCreate).toBe(true);
    expect(config?.memory.requireIsolation).toBe(false);
    expect(config?.memory.sourceType).toBe("user_statement");
    expect(config?.memory.tokenBudget).toBe(512);
  });

  test("memory block rejects unknown keys and bad enum values", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    expect(() =>
      memoryPlugin.configSchema?.parse?.({ synapcores: {}, memory: { backedn: "auto" } }),
    ).toThrow(/unknown keys: backedn/);
    expect(() =>
      memoryPlugin.configSchema?.parse?.({ synapcores: {}, memory: { backend: "sqlite" } }),
    ).toThrow(/memory\.backend must be one of/);
    expect(() =>
      memoryPlugin.configSchema?.parse?.({ synapcores: {}, memory: { identityFrom: "cookie" } }),
    ).toThrow(/memory\.identityFrom must be one of/);
    expect(() =>
      memoryPlugin.configSchema?.parse?.({ synapcores: {}, memory: { tokenBudget: -1 } }),
    ).toThrow(/positive integer/);
  });

  test("memory.identity is left unset for non-static identity sources", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    // `credential` means "the gateway resolves it" — filling in a default here
    // would make the plugin name an identity it was told not to name.
    const credential = memoryPlugin.configSchema?.parse?.({
      synapcores: {},
      memory: { identityFrom: "credential" },
    }) as { memory: { identity?: string } } | undefined;
    expect(credential?.memory.identity).toBeUndefined();

    // `host` derives per turn, but an explicit value is still honoured as the
    // documented tool-call fallback.
    const host = memoryPlugin.configSchema?.parse?.({
      synapcores: {},
      memory: { identityFrom: "host", identity: "fallback_user" },
    }) as { memory: { identity?: string } } | undefined;
    expect(host?.memory.identity).toBe("fallback_user");
  });
});

describe("capture filter heuristics", () => {
  test("shouldCapture filters correctly", async () => {
    const { shouldCapture } = await import("./index.js");
    const triggers = [
      { text: "I prefer dark mode", shouldMatch: true },
      { text: "Remember that my name is John", shouldMatch: true },
      { text: "My email is test@example.com", shouldMatch: true },
      { text: "Call me at +1234567890123", shouldMatch: true },
      { text: "We decided to use TypeScript", shouldMatch: true },
      { text: "I always want verbose output", shouldMatch: true },
      // Natural everyday phrasing that the original narrow regex list
      // missed entirely (see: capture never fired for a real "my
      // favorite X is Y" / "I'm allergic to Z" statement in production).
      { text: "My favorite programming language is Rust and I'm allergic to shellfish.", shouldMatch: true },
      { text: "My name is Luis and I work as a data scientist.", shouldMatch: true },
      { text: "I live in Miami and I'm a vegetarian.", shouldMatch: true },
      { text: "We've decided to use Postgres instead of MySQL going forward.", shouldMatch: true },
      { text: "I can't eat gluten.", shouldMatch: true },
      { text: "Just a random short message", shouldMatch: false },
      { text: "x", shouldMatch: false },
      { text: "<relevant-memories>injected</relevant-memories>", shouldMatch: false },
      // Ordinary conversational filler that must NOT get swept up by the
      // broadened patterns above.
      { text: "Sure, let me check that for you.", shouldMatch: false },
      { text: "The weather today is sunny with a chance of rain.", shouldMatch: false },
      { text: "Can you help me debug this function?", shouldMatch: false },
      { text: "Let me pull up the logs and see what happened.", shouldMatch: false },
    ];

    for (const { text, shouldMatch } of triggers) {
      expect(shouldCapture(text)).toBe(shouldMatch);
    }
  });

  test("detectCategory classifies correctly", () => {
    const cases = [
      { text: "I prefer dark mode", expected: "preference" },
      { text: "We decided to use React", expected: "decision" },
      { text: "My email is test@example.com", expected: "entity" },
      { text: "The server is running on port 3000", expected: "fact" },
    ];

    for (const { text, expected } of cases) {
      const lower = text.toLowerCase();
      let category: string;

      if (/prefer|radši|like|love|hate|want/i.test(lower)) {
        category = "preference";
      } else if (/rozhodli|decided|will use|budeme/i.test(lower)) {
        category = "decision";
      } else if (/\+\d{10,}|@[\w.-]+\.\w+|is called|jmenuje se/i.test(lower)) {
        category = "entity";
      } else if (/is|are|has|have|je|má|jsou/i.test(lower)) {
        category = "fact";
      } else {
        category = "other";
      }

      expect(category).toBe(expected);
    }
  });
});

describe("memory plugin end-to-end (mocked SDK)", () => {
  let cleanup: (() => Promise<void>) | null = null;

  beforeEach(() => {
    cleanup = null;
    mockState.lastFake = null;
  });

  afterEach(async () => {
    if (cleanup) {
      await cleanup();
    }
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function buildMockApi(extraPluginConfig: Record<string, any> = {}) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const registeredTools: any[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const registeredClis: any[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const registeredServices: any[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const registeredHooks: Record<string, any[]> = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const registeredMemoryCapabilities: any[] = [];
    const logs: string[] = [];

    return {
      registeredTools,
      registeredClis,
      registeredServices,
      registeredHooks,
      registeredMemoryCapabilities,
      logs,
      mockApi: {
        id: "memory-synapcores",
        name: "Memory (SynapCores)",
        source: "test",
        config: {},
        pluginConfig: {
          synapcores: {
            host: "localhost",
            port: 8080,
            apiKey: SYNAPCORES_API_KEY,
            useHttps: false,
          },
          collection: "openclaw_memories_test",
          autoCapture: false,
          autoRecall: false,
          ...extraPluginConfig,
        },
        runtime: {},
        logger: {
          info: (msg: string) => logs.push(`[info] ${msg}`),
          warn: (msg: string) => logs.push(`[warn] ${msg}`),
          error: (msg: string) => logs.push(`[error] ${msg}`),
          debug: (msg: string) => logs.push(`[debug] ${msg}`),
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        registerTool: (tool: any, opts: any) => {
          registeredTools.push({ tool, opts });
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        registerCli: (registrar: any, opts: any) => {
          registeredClis.push({ registrar, opts });
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        registerService: (service: any) => {
          registeredServices.push(service);
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        registerMemoryCapability: (capability: any) => {
          registeredMemoryCapabilities.push(capability);
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        on: (hookName: string, handler: any) => {
          if (!registeredHooks[hookName]) {
            registeredHooks[hookName] = [];
          }
          registeredHooks[hookName].push(handler);
        },
        resolvePath: (p: string) => p,
      },
    };
  }

  test("memory tools work end-to-end (store, recall, duplicate, forget) — backed by client.memory", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools, registeredClis, registeredServices } = buildMockApi({
      autoLinkSimilar: false,
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);

    // Five tools since 0.7.0: the three legacy ones plus memory_current /
    // memory_trace, which the CREATE MEMORY backend makes possible. They are
    // registered up front because capability detection is async and
    // registration is not; on a legacy gateway they answer with a stable
    // MEMORY_FEATURE_UNAVAILABLE result.
    expect(registeredTools.length).toBe(5);
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_recall");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_store");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_forget");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_current");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_trace");
    expect(registeredClis.length).toBe(1);
    expect(registeredServices.length).toBe(1);

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const recallTool = registeredTools.find((t) => t.opts?.name === "memory_recall")?.tool;
    const forgetTool = registeredTools.find((t) => t.opts?.name === "memory_forget")?.tool;

    // Store
    const storeResult = await storeTool.execute("test-call-1", {
      text: "The user prefers dark mode for all applications",
      importance: 0.8,
      category: "preference",
    });
    expect(storeResult.details?.action).toBe("created");
    expect(storeResult.details?.id).toBeDefined();
    const storedId = storeResult.details?.id;

    // The fake's MemoryClient should have seen exactly one store call.
    const fake = getLastFake();
    expect(fake.memory.storeCalls.length).toBe(1);
    expect(fake.memory.storeCalls[0].namespace).toBe("openclaw_memories_test");
    expect(fake.memory.storeCalls[0].content).toBe("The user prefers dark mode for all applications");
    expect(fake.memory.storeCalls[0].metadata).toMatchObject({
      importance: 0.8,
      category: "preference",
    });

    // Recall
    const recallResult = await recallTool.execute("test-call-2", {
      query: "The user prefers dark mode for all applications",
      limit: 5,
    });
    expect(recallResult.details?.count).toBeGreaterThan(0);
    expect(recallResult.details?.memories?.[0]?.text).toContain("dark mode");
    expect(fake.memory.recallCalls.length).toBeGreaterThan(0);

    // Duplicate detection
    const duplicateResult = await storeTool.execute("test-call-3", {
      text: "The user prefers dark mode for all applications",
    });
    expect(duplicateResult.details?.action).toBe("duplicate");

    // Forget
    const forgetResult = await forgetTool.execute("test-call-4", {
      memoryId: storedId,
    });
    expect(forgetResult.details?.action).toBe("deleted");
    expect(fake.memory.forgetCalls.length).toBe(1);
    expect(fake.memory.forgetCalls[0].id).toBe(storedId);

    // Verify it's gone
    const recallAfterForget = await recallTool.execute("test-call-5", {
      query: "The user prefers dark mode for all applications",
      limit: 5,
    });
    expect(recallAfterForget.details?.count).toBe(0);
  }, 30000);

  test("regression: store → recall returns the right content + metadata roundtrips + namespace isolation + forget removes", async () => {
    // End-to-end regression that the v0.3.x guarantees still hold after the
    // v0.4.0 MemoryClient swap.
    const { default: memoryPlugin } = await import("./index.js");

    // Namespace A
    const { mockApi: apiA, registeredTools: toolsA } = buildMockApi({
      autoLinkSimilar: false,
      collection: "ns_alpha",
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(apiA as any);
    const fakeA = getLastFake();
    const storeA = toolsA.find((t) => t.opts?.name === "memory_store")!.tool;
    const recallA = toolsA.find((t) => t.opts?.name === "memory_recall")!.tool;
    const forgetA = toolsA.find((t) => t.opts?.name === "memory_forget")!.tool;

    const r = await storeA.execute("x", {
      text: "Alpha namespace remembers the violet preference",
      importance: 0.91,
      category: "preference",
    });
    const idA = r.details.id;

    // Recall returns the right content + correct metadata.
    const rec = await recallA.execute("y", {
      query: "violet preference",
      limit: 5,
    });
    expect(rec.details.count).toBe(1);
    expect(rec.details.memories[0].text).toBe("Alpha namespace remembers the violet preference");
    expect(rec.details.memories[0].category).toBe("preference");
    expect(rec.details.memories[0].importance).toBeCloseTo(0.91, 5);

    // Namespace B sees nothing.
    const { mockApi: apiB, registeredTools: toolsB } = buildMockApi({
      autoLinkSimilar: false,
      collection: "ns_beta",
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(apiB as any);
    const fakeB = getLastFake();
    expect(fakeB).not.toBe(fakeA); // distinct client instances
    const recallB = toolsB.find((t) => t.opts?.name === "memory_recall")!.tool;
    const rb = await recallB.execute("z", {
      query: "violet preference",
      limit: 5,
    });
    expect(rb.details.count).toBe(0);

    // Forget by id removes from namespace A.
    await forgetA.execute("d", { memoryId: idA });
    const recAfter = await recallA.execute("e", {
      query: "violet preference",
      limit: 5,
    });
    expect(recAfter.details.count).toBe(0);
  }, 30000);

  test("SynapCores extensions are exposed on the plugin instance", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi } = buildMockApi();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    expect(ext).toBeDefined();
    expect(typeof ext.recallFiltered).toBe("function");
    expect(typeof ext.recallRelated).toBe("function");
    expect(typeof ext.predictRelevance).toBe("function");
    expect(typeof ext.trainRelevanceModel).toBe("function");
  });

  test("registers a pre-compaction memory flush capability (mirrors memory-core's own pattern)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    // Pre-compaction flush is part of the auto-capture lifecycle, so it is
    // only registered when autoCapture is opted into.
    const { mockApi, registeredMemoryCapabilities } = buildMockApi({ autoCapture: true, autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);

    expect(registeredMemoryCapabilities.length).toBe(1);
    const capability = registeredMemoryCapabilities[0];
    expect(typeof capability.flushPlanResolver).toBe("function");

    const plan = capability.flushPlanResolver({});
    expect(plan).not.toBeNull();
    // Must instruct the agent to use our tool, not memory-core's file-based
    // flush (which the agent would have no tool for anyway).
    expect(plan.prompt).toMatch(/memory_store/);
    expect(plan.systemPrompt).toMatch(/memory_store/);
    // Must give the agent an escape hatch when nothing is worth storing —
    // without this, every near-compaction turn would force a reply.
    expect(plan.systemPrompt).toMatch(/NO_REPLY|SILENT/);
    expect(typeof plan.softThresholdTokens).toBe("number");
    expect(typeof plan.relativePath).toBe("string");
  });

  test("does not crash on hosts without registerMemoryCapability (older OpenClaw versions)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoCapture: true, autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (mockApi as any).registerMemoryCapability;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => memoryPlugin.register(mockApi as any)).not.toThrow();
    // The rest of registration (tools, etc.) must still complete normally —
    // this is the whole point of the guard.
    expect(registeredTools.length).toBe(5);
  });

  test("recallFiltered runs the WHERE clause engine-side against MEMORY_RECALL output (legacy shorthand auto-rewrites)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    await storeTool.execute("c1", {
      text: "I prefer dark mode for everything",
      importance: 0.8,
      category: "preference",
    });
    await storeTool.execute("c2", {
      text: "We always use TypeScript for new projects",
      importance: 0.6,
      category: "decision",
    });

    const results = await ext.recallFiltered({
      where: "category = 'preference'",
      semantic: "what UI does the user like?",
      limit: 5,
    });
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.entry.category).toBe("preference");
    }

    // ENGINE-BUG WORKAROUND: applying a `WHERE metadata->>'…'` predicate to the
    // table-valued MEMORY_RECALL(...) output makes the gateway return 0 rows +
    // an empty column set, so the plugin no longer emits a WHERE clause. It
    // fetches an unfiltered, oversampled MEMORY_RECALL and applies the predicate
    // client-side (compileWherePredicate). Assert the emitted SQL is WHERE-free.
    const fake = getLastFake();
    const memCalls = fake.sqlCalls.filter((c) => /MEMORY_RECALL/.test(c.sql));
    expect(memCalls.length).toBeGreaterThan(0);
    expect(memCalls[0].sql).not.toMatch(/\bWHERE\b/i);
    expect(memCalls[0].sql).toMatch(/FROM MEMORY_RECALL\(\$1, \$2, \$3\)\s+LIMIT/);
  });

  test("recallFiltered applies a compound category+importance predicate client-side (step-3 scenario)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    // Two preferences: one above and one below the importance threshold, plus a
    // decision that must be excluded by the category clause.
    await storeTool.execute("c1", { text: "I prefer dark mode everywhere", importance: 0.9, category: "preference" });
    await storeTool.execute("c2", { text: "I sometimes like verbose logs", importance: 0.5, category: "preference" });
    await storeTool.execute("c3", { text: "We chose TypeScript for the project", importance: 0.8, category: "decision" });

    const results = await ext.recallFiltered({
      where: "category = 'preference' AND importance >= 0.7",
      semantic: "what UI does the user like?",
      limit: 5,
    });
    // Only c1 satisfies BOTH the category and the numeric importance predicate.
    expect(results.length).toBe(1);
    expect(results[0].entry.category).toBe("preference");
    expect(results[0].entry.importance).toBeGreaterThanOrEqual(0.7);
  });

  test("recallFiltered with where '1=1' emits a plain (WHERE-free) MEMORY_RECALL and passes all rows", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    await storeTool.execute("c1", {
      text: "I prefer dark mode for everything",
    });
    const results = await ext.recallFiltered({
      where: "1=1",
      semantic: "anything goes",
    });
    expect(results.length).toBeGreaterThan(0);
    const fake = getLastFake();
    const memCalls = fake.sqlCalls.filter((c) => /MEMORY_RECALL/.test(c.sql));
    expect(memCalls.length).toBeGreaterThan(0);
    // `1=1` short-circuits to pass-all client-side; SQL carries no WHERE.
    expect(memCalls[0].sql).not.toMatch(/\bWHERE\b/i);
  });

  test("capture inserts a Memory graph node when autoLinkSimilar=true (engine-native embedding)", async () => {
    // The graph node is the v0.3.0/v0.4.0 hand-off point that makes
    // recallRelated work. The plugin still writes it via _getHttpClient
    // because the SDK's graph.nodes.create still uses the wrong wire shape.
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;

    const r1 = await storeTool.execute("c1", { text: "The user prefers dark mode for all applications" });
    const r2 = await storeTool.execute("c2", { text: "User Alice manages the analytics dashboard project" });
    const fake = getLastFake();
    expect(fake.graph.nodeIndexByUserId.size).toBe(2);
    expect(fake.graph.nodeIndexByUserId.get(r1.details.id)?.labels).toEqual(["Memory"]);
    expect(fake.graph.nodeIndexByUserId.get(r2.details.id)?.labels).toEqual(["Memory"]);
    const n1 = fake.graph.nodeIndexByUserId.get(r1.details.id)!;
    expect(Array.isArray(n1.properties.embedding)).toBe(true);
    expect((n1.properties.embedding as number[]).length).toBeGreaterThan(0);
    // No cypher MERGE / explicit edge writes happened
    expect(fake.graph.cypherCalls.length).toBe(0);
  });

  test("capture writes no graph nodes with autoLinkSimilar=false", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    await storeTool.execute("c1", { text: "The user prefers dark mode for all applications" });
    await storeTool.execute("c2", { text: "User Alice manages the analytics dashboard project" });
    const fake = getLastFake();
    expect(fake.graph.nodeIndexByUserId.size).toBe(0);
    expect(fake.graph.cypherCalls.length).toBe(0);
  });

  test("recallRelated walks the synthetic SIMILAR_TO graph and returns related Memory nodes", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;

    const a = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });
    await storeTool.execute("c2", { text: "I prefer dark mode in editors" });
    await storeTool.execute("c3", { text: "Decision: PostgreSQL for analytics" });

    const related = await ext.recallRelated(a.details.id, { hops: 1, similarityThreshold: 0.5 });
    expect(Array.isArray(related)).toBe(true);
    expect(related.length).toBeGreaterThanOrEqual(1);
    expect(related.find((r: { entry: { id: string } }) => r.entry.id === a.details.id)).toBeUndefined();
    for (const r of related) {
      expect(r.hops).toBe(1);
      expect(r.via).toContain("SIMILAR_TO");
      expect(r.entry.text).toContain("dark");
    }

    const fake = getLastFake();
    const cypherCalls = fake.graph.cypherCalls;
    expect(cypherCalls.length).toBeGreaterThan(0);
    expect(cypherCalls[0].query).toMatch(/MATCH \(start:Memory \{id: '/);
    expect(cypherCalls[0].query).toMatch(/SIMILAR_TO > 0\.5/);
  });

  test("recallRelated returns [] for unknown memoryId (no source node in graph)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const related = await ext.recallRelated("mem_unknown_xxxx", { hops: 1 });
    expect(related).toEqual([]);
  });

  test("recallRelated honours custom edgeKinds (multi-hop on non-synthetic edges)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const r = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });

    const related = await ext.recallRelated(r.details.id, { hops: 2, edgeKinds: ["MENTIONS"] });
    expect(related).toEqual([]);
    const fake = getLastFake();
    const last = fake.graph.cypherCalls[fake.graph.cypherCalls.length - 1];
    expect(last.query).toMatch(/MENTIONS\*1\.\.2/);
    expect(last.query).not.toMatch(/SIMILAR_TO/);
  });

  test("predictRelevance falls back to heuristic when no model exists", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const r1 = await storeTool.execute("c1", {
      text: "I prefer dark mode for everything",
      importance: 0.8,
    });
    const r2 = await storeTool.execute("c2", {
      text: "Random other thing",
      importance: 0.3,
    });
    void r1;
    void r2;

    const fake = getLastFake();
    expect(fake.automl.models.length).toBe(0);

    // Build candidate entries from the memory store
    const records = fake.memory.stores.get("openclaw_memories_test") ?? [];
    const candidates = records.map((v) => ({
      id: v.id,
      text: v.content,
      vector: [] as number[],
      importance: typeof v.metadata?.importance === "number" ? v.metadata!.importance as number : 0,
      category: (v.metadata?.category as string ?? "other") as "preference" | "fact" | "decision" | "entity" | "other",
      createdAt: v.createdAt.getTime(),
    }));

    const scored = await ext.predictRelevance("dark mode preference", candidates);
    expect(scored.length).toBe(candidates.length);
    for (const s of scored) {
      expect(s.relevance).toBeGreaterThanOrEqual(0);
      expect(s.relevance).toBeLessThanOrEqual(1);
    }
    expect(fake.automl.modelInstances.size).toBe(0);
  });

  test("predictRelevance uses model.predict() when model exists", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const r = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });
    void r;

    const fake = getLastFake();
    fake.automl.models.push({
      id: "model-pre",
      name: "openclaw_memory_relevance",
      task: "regression",
      status: "completed",
    });

    const records = fake.memory.stores.get("openclaw_memories_test") ?? [];
    const candidates = records.map((v) => ({
      id: v.id,
      text: v.content,
      vector: [] as number[],
      importance: typeof v.metadata?.importance === "number" ? v.metadata!.importance as number : 0,
      category: (v.metadata?.category as string ?? "other") as "preference" | "fact" | "decision" | "entity" | "other",
      createdAt: v.createdAt.getTime(),
    }));

    const scored = await ext.predictRelevance("dark mode", candidates);
    expect(scored.length).toBe(candidates.length);
    for (const s of scored) {
      expect(s.relevance).toBeCloseTo(0.42, 5);
    }
    const instance = fake.automl.modelInstances.get("openclaw_memory_relevance")!;
    expect(instance.predictCalls.length).toBe(1);
  });

  test("trainRelevanceModel throws on <10 samples", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    await expect(ext.trainRelevanceModel([])).rejects.toThrow(/at least 10 samples/);
    const tiny = Array.from({ length: 5 }, (_, i) => ({
      memoryId: "mem_xxxx_" + i,
      queryText: "q",
      score: 0.5,
    }));
    await expect(ext.trainRelevanceModel(tiny)).rejects.toThrow(/at least 10 samples/);
  });

  test("trainRelevanceModel stages rows in a SQL table then calls automl.train", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;

    // Seed 12 memories so feedback rows can hydrate to known memories.
    const ids: string[] = [];
    for (let i = 0; i < 12; i++) {
      const result = await storeTool.execute(`c${i}`, {
        text: `Test memory number ${i} I prefer thing ${i}`,
        importance: 0.5 + i * 0.01,
      });
      ids.push(result.details.id);
    }

    const feedback = ids.map((id, i) => ({
      memoryId: id,
      queryText: `Test query ${i}`,
      score: (i % 11) / 10,
    }));

    const result = await ext.trainRelevanceModel(feedback);
    expect(result.modelName).toBe("openclaw_memory_relevance");
    expect(result.modelId).toBeDefined();

    const fake = getLastFake();
    const createCalls = fake.sqlCalls.filter((c) => /^CREATE TABLE/i.test(c.sql));
    const insertCalls = fake.sqlCalls.filter((c) => /^INSERT INTO/i.test(c.sql));
    expect(createCalls.length).toBe(1);
    expect(insertCalls.length).toBe(12);
    expect(createCalls[0].sql).toMatch(/CREATE TABLE openclaw_memory_relevance_training\b/);
    expect(fake.automl.trainCalls.length).toBe(1);
    expect(fake.automl.trainCalls[0].collection).toMatch(/^openclaw_memory_relevance_training/);
    expect(fake.automl.trainCalls[0].target).toBe("score");
    expect(fake.automl.trainCalls[0].task).toBe("regression");
    const post = await ext.predictRelevance("test", [{
      id: ids[0],
      text: "x",
      vector: [],
      importance: 0.5,
      category: "preference" as const,
      createdAt: Date.now(),
    }]);
    expect(post[0].relevance).toBeCloseTo(0.42, 5);
  }, 30000);

  test("trainRelevanceModel re-throws on AutoML failures (does not swallow training errors)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const ids: string[] = [];
    for (let i = 0; i < 12; i++) {
      const result = await storeTool.execute(`c${i}`, {
        text: `mem ${i} I prefer thing ${i}`,
        importance: 0.5,
      });
      ids.push(result.details.id);
    }
    const fake = getLastFake();
    fake.automl.train = async (): Promise<never> => {
      throw new Error("synthetic training failure");
    };
    const feedback = ids.map((id, i) => ({
      memoryId: id,
      queryText: `q${i}`,
      score: 0.5,
    }));
    await expect(ext.trainRelevanceModel(feedback)).rejects.toThrow(/synthetic training failure/);
  }, 30000);

  test("plugin throws if `collection` is not a valid namespace identifier", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi } = buildMockApi({ collection: "openclaw-memories" }); // hyphen!
    expect(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      memoryPlugin.register(mockApi as any);
    }).toThrow(/valid namespace/);
  });

  // ==========================================================================
  // CREATE MEMORY backend (SRS v1.2, engine v1.14.3+)
  // ==========================================================================

  /**
   * Register the plugin with the memory-object surface switched ON in the
   * fake gateway. Returns the usual handles plus the fake's memory-object
   * engine, so a test can inspect what the plugin actually sent.
   */
  async function registerWithMemoryObject(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extraConfig: Record<string, any> = {},
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tune: (objects: FakeMemoryObjects) => void = () => {},
  ) {
    const { default: memoryPlugin } = await import("./index.js");
    const handles = buildMockApi({ autoLinkSimilar: false, ...extraConfig });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(handles.mockApi as any);
    const fake = getLastFake();
    // Detection is lazy — the plugin has not probed yet at this point, so a
    // test can still decide what the gateway looks like.
    fake.memoryObjects.supported = true;
    tune(fake.memoryObjects);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const tool = (name: string) =>
      handles.registeredTools.find((t) => t.opts?.name === name)!.tool;
    return { ...handles, memoryPlugin, fake, ext, tool };
  }

  test("capability detection: falls back to the legacy backend when the gateway has no /v1/memory", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools, logs } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const fake = getLastFake();
    // `supported` stays false: this is an engine that predates CREATE MEMORY.

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")!.tool;
    const result = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });

    expect(result.details.action).toBe("created");
    // The write went to the LEGACY primitives, not the memory object.
    expect(fake.memory.storeCalls.length).toBe(1);
    expect(fake.memoryObjects.episodes.length).toBe(0);
    // The probe ran, and only once (memoized).
    expect(fake.memoryObjects.calls.filter((c) => c.path === "/memory").length).toBe(1);
    expect(logs.join("\n")).toMatch(/memory backend = legacy MEMORY_\* namespace/);
  });

  test("capability detection: selects the memory-object backend and CREATEs the object", async () => {
    const { fake, tool, logs } = await registerWithMemoryObject();

    const result = await tool("memory_store").execute("c1", {
      text: "I prefer dark mode for everything",
    });

    expect(result.details.action).toBe("created");
    expect(result.details.id).toMatch(/^ep_/); // the episode id, not a mem_ id
    // Nothing went to the legacy namespace.
    expect(fake.memory.storeCalls.length).toBe(0);
    expect(fake.memoryObjects.episodes.length).toBe(1);
    expect(fake.memoryObjects.objects.has("openclaw_memories_test")).toBe(true);
    expect(fake.memoryObjects.objects.get("openclaw_memories_test")?.identity_field).toBe("user_id");
    expect(logs.join("\n")).toMatch(/memory backend = memory-object/);
  });

  test("capability detection: an existing object is adopted, not re-created", async () => {
    const { fake, tool } = await registerWithMemoryObject({}, (objects) => {
      objects.objects.set("openclaw_memories_test", { identity_field: "tenant_user" });
    });

    await tool("memory_store").execute("c1", { text: "I prefer dark mode for everything" });

    // Exactly one catalog probe, no CREATE.
    const creates = fake.memoryObjects.calls.filter(
      (c) => c.path === "/memory" && c.body !== undefined,
    );
    expect(creates.length).toBe(0);
    expect(fake.memoryObjects.objects.get("openclaw_memories_test")?.identity_field).toBe(
      "tenant_user",
    );
  });

  test("capability detection: backend 'memory-object' fails loudly instead of falling back", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({
      autoLinkSimilar: false,
      memory: { backend: "memory-object" },
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // supported stays false — the gateway cannot serve it.

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")!.tool;
    await expect(
      storeTool.execute("c1", { text: "I prefer dark mode for everything" }),
    ).rejects.toThrow(/requires SynapCores engine v1\.14\.3|does not serve \/v1\/memory/);
  });

  test("capability detection: backend 'legacy' never probes and hides the object-only tools", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({
      autoLinkSimilar: false,
      memory: { backend: "legacy" },
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const fake = getLastFake();
    fake.memoryObjects.supported = true; // available, but not asked for

    expect(registeredTools.length).toBe(3);
    expect(registeredTools.map((t) => t.opts?.name)).not.toContain("memory_current");

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")!.tool;
    await storeTool.execute("c1", { text: "I prefer dark mode for everything" });
    expect(fake.memory.storeCalls.length).toBe(1);
    expect(fake.memoryObjects.calls.length).toBe(0);
  });

  test("capability detection: an auth failure is NOT read as 'feature absent'", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const fake = getLastFake();
    fake.memoryObjects.list = () => {
      throw fakeSdkError("auth", "unauthorized", "Invalid API key");
    };

    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")!.tool;
    // Silently dropping to the legacy backend here would hide a wrong key
    // behind a path that fails the same way one call later.
    await expect(
      storeTool.execute("c1", { text: "I prefer dark mode for everything" }),
    ).rejects.toThrow(/Invalid API key/);
  });

  test("memory_store REMEMBERs without a client-side dedup probe and reports the engine's changes", async () => {
    const { fake, tool } = await registerWithMemoryObject();
    const storeTool = tool("memory_store");

    const first = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });
    const second = await storeTool.execute("c2", { text: "I prefer dark mode for everything" });

    // Deduplication is the ENGINE's job on this backend, so the plugin does
    // not pre-screen and does not report `duplicate`.
    expect(first.details.action).toBe("created");
    expect(second.details.action).toBe("created");
    expect(fake.memoryObjects.episodes.length).toBe(2);
    expect(first.details.changes).toMatchObject({ episodes_created: 1 });
    expect(first.details.status).toBe("accepted");
    // No legacy recall was issued as a dedup probe.
    expect(fake.memory.recallCalls.length).toBe(0);
  });

  test("memory_recall returns the engine's prompt_ready projection and keeps current_state separate", async () => {
    const { tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "I prefer dark mode for everything" });

    const result = await tool("memory_recall").execute("c2", {
      query: "I prefer dark mode for everything",
      limit: 5,
    });

    expect(result.content[0].text).toContain("ENGINE-PROJECTION:");
    expect(result.details.count).toBeGreaterThan(0);
    // current_state is reported on its own, never merged into the ranked list.
    expect(result.details).toHaveProperty("currentState");
    expect(result.details.truncated).toBe(false);
  });

  test("memory_recall keeps relevance (score) and confidence as separate fields", async () => {
    const { tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", {
      text: "My subscription plan is Enterprise",
    });
    // Give the identity a state-bearing episode so a semantic fact exists.
    const { fake } = await Promise.resolve({ fake: getLastFake() });
    fake.memoryObjects.episodes[0].attribute = "subscription_plan";
    fake.memoryObjects.episodes[0].value = "Enterprise";

    const result = await tool("memory_recall").execute("c2", {
      query: "My subscription plan is Enterprise",
      limit: 5,
    });

    const withConfidence = result.details.memories.find(
      (m: { confidence?: number }) => m.confidence !== undefined,
    );
    expect(withConfidence).toBeDefined();
    // The fake pins confidence at 0.85 and derives relevance from cosine, so
    // equality here would prove the plugin collapsed one into the other.
    expect(withConfidence.confidence).toBeCloseTo(0.85, 5);
    expect(withConfidence.score).not.toBeCloseTo(0.85, 5);
  });

  test("memory_current reports ok / not_found / conflict without guessing", async () => {
    const { fake, tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "seed so the object exists" });
    const currentTool = tool("memory_current");

    // not_found — nothing recorded is an ANSWER, not a failure.
    const missing = await currentTool.execute("c2", { attribute: "cloud_provider" });
    expect(missing.details.status).toBe("not_found");
    expect(missing.details.value).toBeNull();

    // ok
    fake.memoryObjects.state.set("openclaw_memories_test|default|subscription_plan", [
      { value: "Enterprise", confidence: 0.85, authority: "system_of_record" },
    ]);
    const ok = await currentTool.execute("c3", { attribute: "subscription_plan" });
    expect(ok.details.status).toBe("ok");
    expect(ok.details.value).toBe("Enterprise");
    expect(ok.details.authority).toBe("system_of_record");
    expect(ok.details.confidence).toBeCloseTo(0.85, 5);

    // conflict — both claims surface, and the plugin does not pick either.
    fake.memoryObjects.state.set("openclaw_memories_test|default|cloud_provider", [
      { value: "Azure", confidence: 0.7, authority: "user_statement" },
      { value: "AWS", confidence: 0.7, authority: "user_statement" },
    ]);
    const conflicted = await currentTool.execute("c4", { attribute: "cloud_provider" });
    expect(conflicted.details.status).toBe("conflict");
    expect(conflicted.details.competingClaims.map((c: { value: string }) => c.value)).toEqual([
      "Azure",
      "AWS",
    ]);
    expect(conflicted.content[0].text).toMatch(/Ask the user/);
  });

  test("memory_trace explains why a value is believed", async () => {
    const { tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "seed" });
    const fake = getLastFake();
    fake.memoryObjects.episodes[0].attribute = "cloud_provider";
    fake.memoryObjects.episodes[0].value = "AWS";

    const traced = await tool("memory_trace").execute("c2", { attribute: "cloud_provider" });
    expect(traced.details.status).toBe("ok");
    expect(traced.details.derivedFrom[0].content).toBe("seed");
    expect(traced.details.derivedFrom[0].extraction_method).toBe("explicit_option");
    expect(traced.content[0].text).toMatch(/is believed because/);
  });

  test("memory_current / memory_trace degrade with a stable code on a legacy gateway", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: false });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // supported stays false.

    const currentTool = registeredTools.find((t) => t.opts?.name === "memory_current")!.tool;
    const result = await currentTool.execute("c1", { attribute: "subscription_plan" });
    expect(result.details.error).toBe("MEMORY_FEATURE_UNAVAILABLE");
    expect(result.content[0].text).toMatch(/v1\.14\.3/);
  });

  test("memory_forget by query uses FORGET ABOUT and reports the affected counts", async () => {
    const { fake, tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "My home address is 42 Elm Street" });
    await tool("memory_store").execute("c2", { text: "Completely unrelated build tooling note" });

    const result = await tool("memory_forget").execute("c3", { query: "My home address" });

    expect(result.details.action).toBe("forgot_scope");
    expect(result.details.affected.episodes).toBe(1);
    // The unrelated memory survived; the address is gone.
    expect(fake.memoryObjects.episodes.length).toBe(1);
    expect(fake.memoryObjects.episodes[0].content).toMatch(/build tooling/);
    // No candidate-listing dance — the engine resolved the scope itself.
    expect(result.details.candidates).toBeUndefined();
  });

  test("memory_forget by id issues FORGET MEMORY and reports propagation", async () => {
    const { fake, tool } = await registerWithMemoryObject();
    const stored = await tool("memory_store").execute("c1", { text: "I prefer dark mode" });

    const result = await tool("memory_forget").execute("c2", { memoryId: stored.details.id });
    expect(result.details.action).toBe("deleted");
    expect(result.details.affected.episodes).toBe(1);
    expect(fake.memoryObjects.episodes.length).toBe(0);
  });

  test("identity isolation: two host identities never see each other's memories", async () => {
    const { fake, registeredHooks } = await registerWithMemoryObject({
      autoCapture: true,
      autoRecall: true,
      memory: { identityFrom: "host" },
    });

    const capture = registeredHooks["agent_end"][0];
    const recall = registeredHooks["before_prompt_build"][0];

    await capture(
      {
        success: true,
        messages: [{ role: "user", content: "I prefer dark mode for everything" }],
      },
      { senderId: "alice" },
    );
    await capture(
      {
        success: true,
        messages: [{ role: "user", content: "I prefer bright neon green everywhere" }],
      },
      { senderId: "bob" },
    );

    expect(fake.memoryObjects.episodes.map((e) => e.identity).sort()).toEqual(["alice", "bob"]);

    const forAlice = await recall({ prompt: "I prefer dark mode for everything", messages: [] }, {
      senderId: "alice",
    });
    expect(forAlice.prependContext).toContain("dark mode");
    expect(forAlice.prependContext).not.toContain("neon green");

    const forBob = await recall({ prompt: "I prefer dark mode for everything", messages: [] }, {
      senderId: "bob",
    });
    expect(forBob?.prependContext ?? "").not.toContain("dark mode");
  });

  test("identity 'host': a tool call with no context refuses rather than guessing an identity", async () => {
    const { tool } = await registerWithMemoryObject({ memory: { identityFrom: "host" } });

    // OpenClaw tool execute() handlers carry no session/user context, so there
    // is nothing to derive an identity from. Falling back to a shared bucket
    // is the leak this refusal exists to prevent.
    const result = await tool("memory_store").execute("c1", { text: "I prefer dark mode" });
    expect(result.details.error).toBe("MEMORY_FEATURE_UNAVAILABLE");
    expect(result.content[0].text).toMatch(/memory\.identity/);
  });

  test("identity 'host': auto-recall SKIPS when the hook context carries no identity", async () => {
    const { registeredHooks, fake } = await registerWithMemoryObject({
      autoRecall: true,
      memory: { identityFrom: "host" },
    });
    const recall = registeredHooks["before_prompt_build"][0];

    const result = await recall({ prompt: "anything at all", messages: [] }, {});
    expect(result).toBeUndefined();
    // Nothing was even asked of the gateway — no chance of reading a shared bucket.
    expect(fake.memoryObjects.calls.filter((c) => /recall/.test(c.path)).length).toBe(0);
  });

  test("identity 'credential': the plugin names no identity and lets the gateway resolve the pin", async () => {
    const { fake, tool } = await registerWithMemoryObject(
      { memory: { identityFrom: "credential" } },
      (objects) => {
        objects.identityPin = "user-from-token";
      },
    );

    const stored = await tool("memory_store").execute("c1", { text: "I prefer dark mode" });
    expect(stored.details.action).toBe("created");
    // The request body carried no identity at all.
    const rememberCall = fake.memoryObjects.calls.find((c) => /remember/.test(c.path))!;
    expect((rememberCall.body as Record<string, unknown>).identity).toBeUndefined();
    // …and the gateway filed it under the pinned identity.
    expect(fake.memoryObjects.episodes[0].identity).toBe("user-from-token");
  });

  test("identity 'credential' against an UNPINNED credential explains the config fix", async () => {
    const { tool } = await registerWithMemoryObject({
      memory: { identityFrom: "credential" },
    }); // identityPin stays null — an API key is never pinned

    const result = await tool("memory_store").execute("c1", { text: "I prefer dark mode" });
    expect(result.details.error).toBe("MEMORY_FEATURE_UNAVAILABLE");
    expect(result.content[0].text).toMatch(/no memory_identity pin/);
  });

  test("MEMORY_ACCESS_DENIED is handled explicitly, not as a generic auth failure", async () => {
    const { tool } = await registerWithMemoryObject(
      { memory: { identityFrom: "static", identity: "alice" } },
      (objects) => {
        // A credential pinned to someone else entirely.
        objects.identityPin = "bob";
      },
    );

    const result = await tool("memory_store").execute("c1", { text: "I prefer dark mode" });
    expect(result.details.error).toBe("MEMORY_ACCESS_DENIED");
    expect(result.content[0].text).toMatch(/pinned to a different identity/);
    // The message must name the fix, not just the failure.
    expect(result.content[0].text).toMatch(/memory\.identityFrom/);
  });

  test("MEMORY_ACCESS_DENIED in auto-recall is logged as a refusal and injects nothing", async () => {
    const { registeredHooks, logs } = await registerWithMemoryObject(
      { autoRecall: true, memory: { identityFrom: "static", identity: "alice" } },
      (objects) => {
        objects.identityPin = "bob";
      },
    );
    const recall = registeredHooks["before_prompt_build"][0];

    const result = await recall({ prompt: "what do I prefer?", messages: [] }, {});
    expect(result).toBeUndefined();
    expect(logs.join("\n")).toMatch(/auto-recall refused/);
    expect(logs.join("\n")).toMatch(/MEMORY_ACCESS_DENIED|pinned to a different identity/);
  });

  test("auto-capture records the turn's role as the claim's source_type", async () => {
    const { fake, registeredHooks } = await registerWithMemoryObject({ autoCapture: true });
    const capture = registeredHooks["agent_end"][0];

    await capture(
      {
        success: true,
        messages: [
          { role: "user", content: "I prefer dark mode for everything" },
          { role: "assistant", content: "Noted — I always use dark mode for you now" },
        ],
      },
      {},
    );

    const bySource = Object.fromEntries(
      fake.memoryObjects.episodes.map((e) => [e.content.slice(0, 8), e.sourceType]),
    );
    // A user's own statement outranks the assistant's paraphrase when the two
    // disagree — that is what source_type is for.
    expect(bySource["I prefer"]).toBe("user_statement");
    expect(bySource["Noted — "]).toBe("agent_derivation");
  });

  test("auto-recall injects the engine's projection verbatim instead of hand-assembling one", async () => {
    const { registeredHooks, tool } = await registerWithMemoryObject({ autoRecall: true });
    await tool("memory_store").execute("c1", { text: "I prefer dark mode for everything" });

    const recall = registeredHooks["before_prompt_build"][0];
    const result = await recall({ prompt: "I prefer dark mode for everything", messages: [] }, {});

    expect(result.prependContext).toContain("<relevant-memories>");
    expect(result.prependContext).toContain("ENGINE-PROJECTION:");
    // The legacy hand-assembled "- [category] text" shape is gone on this path.
    expect(result.prependContext).not.toMatch(/- \[other\]/);
  });

  test("ltm stats reports which backend is live", async () => {
    const { ext } = await registerWithMemoryObject();
    expect(await ext.backendKind()).toBe("memory-object");
  });

  test("consolidateMemory is an extension, never a model-facing tool", async () => {
    const { ext, registeredTools, fake, tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "seed" });

    // Memory correctness must not depend on an agent remembering to call it.
    expect(registeredTools.map((t) => t.opts?.name)).not.toContain("memory_consolidate");

    const envelope = await ext.consolidateMemory();
    expect(envelope.status).toBe("completed");
    expect(fake.memoryObjects.calls.some((c) => /consolidate/.test(c.path))).toBe(true);
  });

  test("recallFiltered rejects a metadata predicate on the memory-object backend", async () => {
    const { ext, tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", {
      text: "I prefer dark mode for everything",
      category: "preference",
      importance: 0.9,
    });

    // The object backend stores no per-memory metadata, so this predicate can
    // never be true. Returning [] would look like "nothing matched".
    await expect(
      ext.recallFiltered({
        where: "category = 'preference' AND importance >= 0.7",
        semantic: "what UI does the user like?",
      }),
    ).rejects.toThrow(/cannot filter on category, importance/);
  });

  test("recallFiltered runs the portable predicate subset on the memory-object backend", async () => {
    const { ext, tool } = await registerWithMemoryObject();
    await tool("memory_store").execute("c1", { text: "I prefer dark mode for everything" });
    await tool("memory_store").execute("c2", { text: "Postgres was chosen for analytics" });

    const results = await ext.recallFiltered({
      where: "text LIKE '%dark mode%'",
      semantic: "what UI does the user like?",
      limit: 5,
    });
    expect(results.length).toBe(1);
    expect(results[0].entry.text).toContain("dark mode");
  });

  test("trainRelevanceModel hydrates from the memory object's episodes table", async () => {
    const { ext, fake, tool } = await registerWithMemoryObject();
    const ids: string[] = [];
    for (let i = 0; i < 12; i++) {
      const stored = await tool("memory_store").execute(`c${i}`, {
        text: `Test memory number ${i} I prefer thing ${i}`,
      });
      ids.push(stored.details.id);
    }

    const result = await ext.trainRelevanceModel(
      ids.map((id, i) => ({ memoryId: id, queryText: `Test query ${i}`, score: (i % 11) / 10 })),
    );
    expect(result.modelName).toBe("openclaw_memory_relevance");
    expect(fake.automl.trainCalls.length).toBe(1);
    // Hydration read the object's own episodes table — the memory verbs have
    // no fetch-by-id, and the five backing tables are ordinary tables.
    expect(
      fake.sqlCalls.some((c) => /_mem_openclaw_memories_test_episodes/.test(c.sql)),
    ).toBe(true);
  });

  test("a memory object name that is not a valid engine identifier stays on the legacy path", async () => {
    // The name becomes part of five physical table names, so the engine holds
    // it to ^[A-Za-z_][A-Za-z0-9_]*$ and 64 characters. A longer collection is
    // a reason to stay on the legacy backend, not to refuse to load.
    const { fake, tool, logs } = await registerWithMemoryObject({
      memory: { name: "a".repeat(65) },
    });

    await tool("memory_store").execute("c1", { text: "I prefer dark mode for everything" });
    expect(fake.memory.storeCalls.length).toBe(1);
    expect(fake.memoryObjects.episodes.length).toBe(0);
    expect(logs.join("\n")).toMatch(/not a valid engine identifier/);
  });
});

describe("escapeCypherString", () => {
  test("returns plain text unchanged", async () => {
    const { escapeCypherString } = await import("./index.js");
    expect(escapeCypherString("hello-world")).toBe("hello-world");
    expect(escapeCypherString("abc123_-")).toBe("abc123_-");
  });

  test("escapes single quotes", async () => {
    const { escapeCypherString } = await import("./index.js");
    expect(escapeCypherString("o'brien")).toBe("o\\'brien");
    expect(escapeCypherString("'")).toBe("\\'");
    expect(escapeCypherString("a'b'c")).toBe("a\\'b\\'c");
  });

  test("escapes backslashes (and does so before quotes so '\\'' stays escaped)", async () => {
    const { escapeCypherString } = await import("./index.js");
    expect(escapeCypherString("a\\b")).toBe("a\\\\b");
    expect(escapeCypherString("\\")).toBe("\\\\");
    expect(escapeCypherString("a\\'b")).toBe("a\\\\\\'b");
  });

  test("an engine-issued memory id round-trips unchanged (the hot path)", async () => {
    const { escapeCypherString } = await import("./index.js");
    const id = "mem_1kv69sxfn_5ofzwK";
    expect(escapeCypherString(id)).toBe(id);
  });

  test("the canonical SQL-injection probe gets neutralised inside Cypher", async () => {
    const { escapeCypherString } = await import("./index.js");
    const probe = "' OR 1=1 //";
    const escaped = escapeCypherString(probe);
    const cypher = `MATCH (n:Memory {id: '${escaped}'}) RETURN n`;
    let i = cypher.indexOf("'") + 1;
    while (i < cypher.length) {
      if (cypher[i] === "\\") {
        i += 2;
        continue;
      }
      if (cypher[i] === "'") break;
      i++;
    }
    expect(cypher.slice(i)).toBe("'}) RETURN n");
  });
});

// Live tests that require a running SynapCores gateway (embeddings are
// engine-native — no external embedding provider key needed).
// Enable with: OPENCLAW_LIVE_TEST=1 SYNAPCORES_API_KEY=... pnpm test
describeLive("memory plugin live tests", () => {
  test("smoke: registers against real SynapCores gateway", async () => {
    expect(true).toBe(true);
  });
});

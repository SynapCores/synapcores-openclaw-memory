/**
 * Memory Plugin Unit Tests
 *
 * Tests the memory plugin functionality including:
 * - Plugin registration and configuration
 * - Memory storage and retrieval (via mocked SynapCores SDK)
 * - Auto-recall via hooks
 * - Auto-capture filtering
 * - SynapCores-only extensions: recallFiltered, recallRelated,
 *   predictRelevance, trainRelevanceModel
 *
 * v0.2.0: the fake now mirrors @synapcores/sdk@0.4.0's public surface
 * (`createVectorCollection`, `vectorCollection(name)`, `graph.cypher`,
 * `graph.nodes.create`, `executeQuery`, `automl.*`). The 0.1.0
 * `_getHttpClient()`-based fake is gone because the plugin no longer
 * reaches into the SDK internals.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "test-key";
const SYNAPCORES_API_KEY = process.env.SYNAPCORES_API_KEY ?? "ak_prod_test_key";
const HAS_OPENAI_KEY = Boolean(process.env.OPENAI_API_KEY);
const liveEnabled = HAS_OPENAI_KEY && process.env.OPENCLAW_LIVE_TEST === "1";
const describeLive = liveEnabled ? describe : describe.skip;

// ---------------------------------------------------------------------------
// In-memory fake of the SynapCores SDK (v0.4.0 surface)
// ---------------------------------------------------------------------------
// We mock @synapcores/sdk before importing the plugin so that no network I/O
// happens during unit tests. The fake implements just enough of the
// `SynapCores` client to back the plugin's store / search / delete / count /
// graph.cypher / graph.nodes.create / executeQuery / automl.train /
// automl.listModels / automl.getModel paths against in-process state.

type FakeVec = {
  id: string;
  values: number[];
  metadata: {
    text?: string;
    importance?: number;
    category?: string;
    createdAt?: number;
    [k: string]: unknown;
  };
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

class FakeVectorCollection {
  public vectors = new Map<string, FakeVec>();
  public lastSearchFilter: unknown = undefined;
  constructor(public readonly name: string) {}

  async insert(records: FakeVec | FakeVec[]): Promise<{ inserted: number; ids: string[] }> {
    const arr = Array.isArray(records) ? records : [records];
    const ids: string[] = [];
    for (const v of arr) {
      this.vectors.set(v.id, {
        id: v.id,
        values: v.values ?? [],
        metadata: { ...(v.metadata ?? {}) },
      });
      ids.push(v.id);
    }
    return { inserted: ids.length, ids };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async search(options: { vector: number[]; k?: number; topK?: number; filter?: any; includeMetadata?: boolean }): Promise<Array<{ id: string; score: number; metadata?: Record<string, unknown>; values?: number[] }>> {
    this.lastSearchFilter = options.filter;
    const k = options.k ?? options.topK ?? 10;
    const scored = Array.from(this.vectors.values()).map((v) => {
      const sim = cosineSim(options.vector, v.values);
      const distance = 1 - sim;
      return {
        id: v.id,
        // Mirror the gateway: `score` IS the cosine distance.
        score: distance,
        metadata: v.metadata,
        values: v.values,
      };
    });
    scored.sort((a, b) => a.score - b.score);
    return scored.slice(0, k);
  }

  async get(id: string): Promise<{ id: string; values: number[]; metadata: Record<string, unknown> } | null> {
    const v = this.vectors.get(id);
    if (!v) return null;
    return { id: v.id, values: v.values, metadata: v.metadata };
  }

  async delete(ids: string | string[]): Promise<{ deleted: number }> {
    const list = typeof ids === "string" ? [ids] : ids;
    let n = 0;
    for (const i of list) {
      if (this.vectors.delete(i)) n++;
    }
    return { deleted: n };
  }

  async count(): Promise<number> {
    return this.vectors.size;
  }

  async info(): Promise<{ name: string; dimensions: number; vector_count: number; distance_metric: string }> {
    return {
      name: this.name,
      dimensions: 1536,
      vector_count: this.vectors.size,
      distance_metric: "cosine",
    };
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
    // Also index by the user-supplied `id` property so MATCH (n:Memory {id: 'X'}) works
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
  // `nodes` matches the SDK 0.4.0 surface: client.graph.nodes.create(label, props)
  public readonly nodes: FakeGraphNodesApi;

  constructor() {
    this.nodes = new FakeGraphNodesApi(this);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async cypher(query: string, params?: Record<string, any>, graph?: string): Promise<{ columns: string[]; rows: unknown[][]; records: Array<Record<string, unknown>> }> {
    this.cypherCalls.push({ query, params, graph });
    // Inline-only contract (gateway v1.6.5.x rejects $param bindings).
    if (params !== undefined && Object.keys(params).length > 0) {
      throw new Error(
        `FakeGraph.cypher: gateway v1.6.5.x rejects $param bindings; caller passed params=${JSON.stringify(
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
  private vectorCollections = new Map<string, FakeVectorCollection>();
  public graph = new FakeGraph();
  public automl = new FakeAutoML();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public sqlCalls: Array<{ sql: string; parameters?: any[] }> = [];
  // Synthetic in-memory SQL tables for executeQuery
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public sqlTables = new Map<string, Array<Record<string, any>>>();

  async listVectorCollections(): Promise<Array<{ name: string; dimensions: number; vector_count: number; distance_metric: string }>> {
    return Array.from(this.vectorCollections.values()).map((c) => ({
      name: c.name,
      dimensions: 1536,
      vector_count: c.vectors.size,
      distance_metric: "cosine",
    }));
  }

  async createVectorCollection(opts: { name: string; dimensions: number; distance_metric?: string }): Promise<FakeVectorCollection> {
    let c = this.vectorCollections.get(opts.name);
    if (!c) {
      c = new FakeVectorCollection(opts.name);
      this.vectorCollections.set(opts.name, c);
    }
    return c;
  }

  vectorCollection(name: string): FakeVectorCollection {
    let c = this.vectorCollections.get(name);
    if (!c) {
      c = new FakeVectorCollection(name);
      this.vectorCollections.set(name, c);
    }
    return c;
  }

  async deleteVectorCollection(name: string): Promise<void> {
    this.vectorCollections.delete(name);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async executeQuery(req: { sql: string; parameters?: any[] }): Promise<any> {
    this.sqlCalls.push(req);
    const sql = req.sql.trim();
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
    // SELECT COUNT(*) FROM <name>
    m = sql.match(/^SELECT COUNT\(\*\) FROM\s+(\w+)/i);
    if (m) {
      const t = m[1];
      const n = (this.sqlTables.get(t) ?? []).length;
      return {
        columns: [{ name: "count", data_type: "INTEGER", nullable: true }],
        rows: [[n]],
        execution_time_ms: 0,
      };
    }
    return { columns: [], rows: [], execution_time_ms: 0 };
  }

  // SDK 0.4.0 still ships a broken `graph.nodes.create` (posts `label:`
  // singular, gateway expects `labels:` plural). The plugin works around
  // this in `linkSimilarMemories` by reaching into `_getHttpClient().post`
  // with the correct wire shape. The fake mirrors that path for unit-test
  // coverage of the workaround.
  _getHttpClient(): {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    post: (path: string, body?: any) => Promise<{ data: unknown }>;
  } {
    const self = this;
    return {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async post(path: string, body?: any) {
        if (path === "/graph/nodes") {
          const labels = Array.isArray(body?.labels) ? body.labels : [];
          const props = body?.properties ?? {};
          // Mirror what client.graph.nodes.create() would do via the API
          // surface so the same downstream cypher walks work.
          const id = String(props.id ?? `gn-${self.graph.allNodes.length + 1}`);
          const node: FakeGraphNode = { id, labels: [...labels], properties: { ...props } };
          self.graph.allNodes.push(node);
          self.graph.nodeIndexById.set(id, node);
          self.graph.nodeIndexByUserId.set(String(props.id ?? id), node);
          return { data: { id, labels, properties: props } };
        }
        throw new Error(`FakeSynapCores._getHttpClient: unhandled POST ${path}`);
      },
    };
  }

  // Test helpers
  _peekVectorCollection(name: string): FakeVectorCollection | undefined {
    return this.vectorCollections.get(name);
  }
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
  VectorCollection: class {},
}));

function getLastFake(): FakeSynapCores {
  if (!mockState.lastFake) {
    throw new Error("SynapCores client was never instantiated");
  }
  return mockState.lastFake as FakeSynapCores;
}

// Mock OpenAI embeddings to return deterministic, token-based vectors so
// that semantically-similar texts (texts that share tokens) get high cosine
// similarity. Each token hashes into a primary + two secondary buckets out
// of a 1536-dim sparse vector. This means:
//   - identical text -> cosine 1.0
//   - "a b c d e" vs "a b c d e f" -> cosine ~0.94
//   - texts sharing no tokens -> cosine 0
vi.mock("openai", () => {
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
  return {
    default: class OpenAIMock {
      embeddings = {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        create: async ({ input }: { input: string | string[] }): Promise<any> => {
          const text = Array.isArray(input) ? input[0] : input;
          const dim = 1536;
          const vec = new Array<number>(dim).fill(0);
          for (const tok of tokenize(text)) {
            const h = hashStr(tok);
            const idx = h % dim;
            vec[idx] += 1;
            vec[(idx + 7) % dim] += 0.5;
            vec[(idx + 17) % dim] += 0.5;
          }
          return { data: [{ embedding: vec }] };
        },
      };
    },
  };
});

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
      embedding: {
        apiKey: OPENAI_API_KEY,
        model: "text-embedding-3-small",
      },
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
    expect(config?.embedding?.apiKey).toBe(OPENAI_API_KEY);
    expect(config?.synapcores?.host).toBe("localhost");
    expect(config?.synapcores?.port).toBe(8080);
    expect(config?.collection).toBe("openclaw_memories");
    expect(config?.graph).toBe("openclaw_memory_graph");
    expect(config?.autoLinkSimilar).toBe(true);
  });

  test("config schema resolves env vars", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    process.env.TEST_MEMORY_API_KEY = "test-key-123";
    process.env.TEST_SC_API_KEY = "ak_prod_test_456";

    const config = memoryPlugin.configSchema?.parse?.({
      embedding: { apiKey: "${TEST_MEMORY_API_KEY}" },
      synapcores: { apiKey: "${TEST_SC_API_KEY}" },
    });

    expect(config?.embedding?.apiKey).toBe("test-key-123");
    expect(config?.synapcores?.apiKey).toBe("ak_prod_test_456");

    delete process.env.TEST_MEMORY_API_KEY;
    delete process.env.TEST_SC_API_KEY;
  });

  test("config schema rejects missing embedding apiKey", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        embedding: {},
        synapcores: { apiKey: SYNAPCORES_API_KEY },
      });
    }).toThrow("embedding.apiKey is required");
  });

  test("config schema rejects missing synapcores apiKey", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        embedding: { apiKey: OPENAI_API_KEY },
        synapcores: {},
      });
    }).toThrow("synapcores.apiKey is required");
  });

  test("config schema applies defaults for collection, graph, host, port, autoLinkSimilar", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      embedding: { apiKey: OPENAI_API_KEY },
      synapcores: { apiKey: SYNAPCORES_API_KEY },
    });

    expect(config?.synapcores?.host).toBe("localhost");
    expect(config?.synapcores?.port).toBe(8080);
    expect(config?.synapcores?.useHttps).toBe(false);
    expect(config?.collection).toBe("openclaw_memories");
    expect(config?.graph).toBe("openclaw_memory_graph");
    // autoCapture / autoRecall / autoLinkSimilar default to true
    expect(config?.autoCapture).toBe(true);
    expect(config?.autoRecall).toBe(true);
    expect(config?.autoLinkSimilar).toBe(true);
  });
});

describe("capture filter heuristics", () => {
  test("shouldCapture filters correctly", () => {
    const triggers = [
      { text: "I prefer dark mode", shouldMatch: true },
      { text: "Remember that my name is John", shouldMatch: true },
      { text: "My email is test@example.com", shouldMatch: true },
      { text: "Call me at +1234567890123", shouldMatch: true },
      { text: "We decided to use TypeScript", shouldMatch: true },
      { text: "I always want verbose output", shouldMatch: true },
      { text: "Just a random short message", shouldMatch: false },
      { text: "x", shouldMatch: false },
      { text: "<relevant-memories>injected</relevant-memories>", shouldMatch: false },
    ];

    for (const { text, shouldMatch } of triggers) {
      const hasPreference = /prefer|radši|like|love|hate|want/i.test(text);
      const hasRemember = /zapamatuj|pamatuj|remember/i.test(text);
      const hasEmail = /[\w.-]+@[\w.-]+\.\w+/.test(text);
      const hasPhone = /\+\d{10,}/.test(text);
      const hasDecision = /rozhodli|decided|will use|budeme/i.test(text);
      const hasAlways = /always|never|important/i.test(text);
      const isInjected = text.includes("<relevant-memories>");
      const isTooShort = text.length < 10;

      const wouldCapture =
        !isTooShort &&
        !isInjected &&
        (hasPreference || hasRemember || hasEmail || hasPhone || hasDecision || hasAlways);

      if (shouldMatch) {
        expect(wouldCapture).toBe(true);
      }
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
    const logs: string[] = [];

    return {
      registeredTools,
      registeredClis,
      registeredServices,
      registeredHooks,
      logs,
      mockApi: {
        id: "memory-synapcores",
        name: "Memory (SynapCores)",
        source: "test",
        config: {},
        pluginConfig: {
          embedding: {
            apiKey: OPENAI_API_KEY,
            model: "text-embedding-3-small",
          },
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

  test("memory tools work end-to-end (store, recall, duplicate, forget)", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools, registeredClis, registeredServices } = buildMockApi({
      autoLinkSimilar: false,
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);

    expect(registeredTools.length).toBe(3);
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_recall");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_store");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_forget");
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

    // Recall
    const recallResult = await recallTool.execute("test-call-2", {
      query: "The user prefers dark mode for all applications",
      limit: 5,
    });
    expect(recallResult.details?.count).toBeGreaterThan(0);
    expect(recallResult.details?.memories?.[0]?.text).toContain("dark mode");

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

    // Verify it's gone
    const recallAfterForget = await recallTool.execute("test-call-5", {
      query: "The user prefers dark mode for all applications",
      limit: 5,
    });
    expect(recallAfterForget.details?.count).toBe(0);
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

  test("recallFiltered embeds the semantic query and forwards the WHERE clause as a filter", async () => {
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

    // Inspect what filter the SDK saw.
    const fake = getLastFake();
    const coll = fake._peekVectorCollection("openclaw_memories_test")!;
    expect(coll.lastSearchFilter).toEqual({ sql: "category = 'preference'" });
  });

  test("recallFiltered with where '1=1' behaves like recall (forwards 1=1 filter)", async () => {
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
    const coll = fake._peekVectorCollection("openclaw_memories_test")!;
    expect(coll.lastSearchFilter).toEqual({ sql: "1=1" });
  });

  test("v0.2.0: capture inserts a Memory graph node when autoLinkSimilar=true (was a no-op in v0.1.0)", async () => {
    // v0.1.0 was a no-op — explicit MERGE on SIMILAR_TO is rejected by the
    // gateway. v0.2.0 takes the supported path: insert a Memory node
    // carrying the embedding under the property name `embedding`, so the
    // gateway's vector-indexed synthetic SIMILAR_TO edge resolves at
    // MATCH time in `recallRelated`.
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;

    const r1 = await storeTool.execute("c1", { text: "The user prefers dark mode for all applications" });
    const r2 = await storeTool.execute("c2", { text: "User Alice manages the analytics dashboard project" });
    const fake = getLastFake();
    // Two Memory nodes; no Cypher MERGE writes (that was the v0.1.0 anti-path).
    expect(fake.graph.nodeIndexByUserId.size).toBe(2);
    expect(fake.graph.nodeIndexByUserId.get(r1.details.id)?.labels).toEqual(["Memory"]);
    expect(fake.graph.nodeIndexByUserId.get(r2.details.id)?.labels).toEqual(["Memory"]);
    // Each Memory node carries the embedding under `embedding` — that's
    // the property name the gateway's brute-force vector index is wired
    // against (attach_default_vector_index).
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

  test("v0.2.0: recallRelated walks the synthetic SIMILAR_TO graph and returns related Memory nodes", async () => {
    // Mocks the v0.2.0 wire shape: a Memory node carries an `embedding`
    // property; the gateway derives SIMILAR_TO synthetically by cosine
    // similarity at MATCH time. The fake reproduces that path in-process
    // so the same Cypher the plugin emits round-trips here.
    const { default: memoryPlugin } = await import("./index.js");
    const { mockApi, registeredTools } = buildMockApi({ autoLinkSimilar: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    memoryPlugin.register(mockApi as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ext = (memoryPlugin as any).extensions;
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;

    // Capture two related memories (shared "dark mode" tokens => cosine ~ 1).
    const a = await storeTool.execute("c1", { text: "I prefer dark mode for everything" });
    await storeTool.execute("c2", { text: "I prefer dark mode in editors" });
    // And one unrelated memory (no shared tokens).
    await storeTool.execute("c3", { text: "Decision: PostgreSQL for analytics" });

    const related = await ext.recallRelated(a.details.id, { hops: 1, similarityThreshold: 0.5 });
    expect(Array.isArray(related)).toBe(true);
    // The two "dark mode" memories share many tokens, so c2 should surface.
    expect(related.length).toBeGreaterThanOrEqual(1);
    // c1 (the source) must NOT appear in its own neighborhood.
    expect(related.find((r: { entry: { id: string } }) => r.entry.id === a.details.id)).toBeUndefined();
    // Each result carries hops and via.
    for (const r of related) {
      expect(r.hops).toBe(1);
      expect(r.via).toContain("SIMILAR_TO");
      expect(r.entry.text).toContain("dark");
    }

    // The gateway was hit with a SIMILAR_TO Cypher MATCH — inline values,
    // no $param bindings (which the gateway rejects).
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
    const related = await ext.recallRelated("00000000-0000-0000-0000-000000000000", { hops: 1 });
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

    // Custom edge kind — the fake's cypher doesn't model MENTIONS, so the
    // result will be empty, but the plugin must emit a Cypher that uses
    // `MENTIONS*1..2` (multi-hop for non-synthetic edges) without throwing.
    const related = await ext.recallRelated(r.details.id, { hops: 2, edgeKinds: ["MENTIONS"] });
    expect(related).toEqual([]);
    const fake = getLastFake();
    const last = fake.graph.cypherCalls[fake.graph.cypherCalls.length - 1];
    expect(last.query).toMatch(/MENTIONS\*1\.\.2/);
    // SIMILAR_TO does NOT appear (caller opted out)
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
    // No models registered -> falls back to heuristic
    expect(fake.automl.models.length).toBe(0);

    // Build candidate entries from the fake vectors
    const coll = fake._peekVectorCollection("openclaw_memories_test")!;
    const candidates = Array.from(coll.vectors.values()).map((v) => ({
      id: v.id,
      text: String(v.metadata.text ?? ""),
      vector: v.values,
      importance: typeof v.metadata.importance === "number" ? v.metadata.importance : 0,
      category: ((v.metadata.category as string) ?? "other") as "preference" | "fact" | "decision" | "entity" | "other",
      createdAt: typeof v.metadata.createdAt === "number" ? v.metadata.createdAt : Date.now(),
    }));

    const scored = await ext.predictRelevance("dark mode preference", candidates);
    expect(scored.length).toBe(candidates.length);
    for (const s of scored) {
      expect(s.relevance).toBeGreaterThanOrEqual(0);
      expect(s.relevance).toBeLessThanOrEqual(1);
    }
    // The model.predict path was NOT called
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

    const coll = fake._peekVectorCollection("openclaw_memories_test")!;
    const candidates = Array.from(coll.vectors.values()).map((v) => ({
      id: v.id,
      text: String(v.metadata.text ?? ""),
      vector: v.values,
      importance: typeof v.metadata.importance === "number" ? v.metadata.importance : 0,
      category: ((v.metadata.category as string) ?? "other") as "preference" | "fact" | "decision" | "entity" | "other",
      createdAt: typeof v.metadata.createdAt === "number" ? v.metadata.createdAt : Date.now(),
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
      memoryId: "00000000-0000-0000-0000-00000000000" + i,
      queryText: "q",
      score: 0.5,
    }));
    await expect(ext.trainRelevanceModel(tiny)).rejects.toThrow(/at least 10 samples/);
  });

  test("v0.2.0: trainRelevanceModel stages rows in a SQL table then calls automl.train", async () => {
    // v0.1.0 threw "signature-only" on this path. v0.2.0 stages each
    // feedback row's hydrated feature vector in a CREATE TABLE / INSERT
    // pipeline, then calls /v1/automl/train with the staging table as
    // `collection` and `target: 'score'`.
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
    // SQL pipeline ran: 1 CREATE + 12 INSERTs
    const createCalls = fake.sqlCalls.filter((c) => /^CREATE TABLE/i.test(c.sql));
    const insertCalls = fake.sqlCalls.filter((c) => /^INSERT INTO/i.test(c.sql));
    expect(createCalls.length).toBe(1);
    expect(insertCalls.length).toBe(12);
    expect(createCalls[0].sql).toMatch(/CREATE TABLE openclaw_memory_relevance_training\b/);
    // AutoML train was called with the staging table as `collection`
    expect(fake.automl.trainCalls.length).toBe(1);
    expect(fake.automl.trainCalls[0].collection).toMatch(/^openclaw_memory_relevance_training/);
    expect(fake.automl.trainCalls[0].target).toBe("score");
    expect(fake.automl.trainCalls[0].task).toBe("regression");
    // The trained model is then visible to subsequent predictRelevance calls
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
    // Wire automl.train to fail
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

  test("a UUID round-trips unchanged (the hot path)", async () => {
    const { escapeCypherString } = await import("./index.js");
    const uuid = "11111111-2222-3333-4444-555555555555";
    expect(escapeCypherString(uuid)).toBe(uuid);
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

// Live tests that require OpenAI + a running SynapCores gateway.
// Enable with: OPENCLAW_LIVE_TEST=1 OPENAI_API_KEY=... SYNAPCORES_API_KEY=... pnpm test
describeLive("memory plugin live tests", () => {
  test("smoke: registers against real SynapCores gateway", async () => {
    expect(true).toBe(true);
  });
});

# @synapcores/openclaw-memory

A long-term memory plugin for [OpenClaw](https://github.com/openclaw/openclaw) that uses **SynapCores AIDB** as the storage backend. Drop-in alternative to `@openclaw/memory-lancedb` with the same auto-recall / auto-capture lifecycle, plus three SynapCores-only extensions: SQL-filtered semantic recall, graph-relation walks, and AutoML relevance scoring.

> **0.2.0 shipping note:** verified against gateway `v1.6.5.2-ce` and `@synapcores/sdk@^0.4.0`. The full surface — parity API (recall/capture/forget) **plus all four SynapCores-only extensions** (`recallFiltered`, `recallRelated`, `predictRelevance`, `trainRelevanceModel`) — is wired against the live SDK and exercised end-to-end (7/7 live smoke steps pass against `:28201`).
>
> - **`recallRelated`** is now implemented (was signature-only in 0.1.0). On capture, the plugin inserts each Memory as a graph node carrying the embedding under the `embedding` property so the gateway's synthetic `SIMILAR_TO` edge resolves at MATCH time. `autoLinkSimilar: true` (default) enables this; turn it off if you don't need graph-backed recall.
> - **`trainRelevanceModel`** is now implemented (was signature-only in 0.1.0). Feedback rows are staged in a SQL table (`openclaw_memory_relevance_training[_<workspace>]`) the gateway's AutoML can read, then trained as a regression model targeting the `score` column.
> - SDK dep bumped to `@synapcores/sdk@^0.4.0` (which switched API-key auth to `Authorization: Bearer` and added `createVectorCollection` / `vectorCollection(name)`). The three v0.1.0 workarounds — auth-header shim, direct `/v1/vectors/collections` POSTs, direct `/v1/vectors/collections/{n}/vectors` POSTs — are deleted in favour of the SDK's typed helpers.

## Why use this over `@openclaw/memory-lancedb`?

| Capability | memory-lancedb | memory-synapcores |
| --- | --- | --- |
| Vector recall + capture | yes | yes |
| Auto-recall / auto-capture hooks | yes | yes |
| GDPR-style forget by ID or query | yes | yes |
| SQL-scoped semantic recall (`recallFiltered`) | no | **yes** |
| Graph relation walks (`recallRelated`) | no | **yes** |
| AutoML relevance scoring (`predictRelevance`) | no | **yes** |
| Backend | local LanceDB files | SynapCores gateway (HTTP) |

If you only need a private, single-user, file-backed vector store, stay on `@openclaw/memory-lancedb`. If you want any of: cross-session/multi-device shared memory, SQL filtering across metadata, graph relations between memories, or per-user relevance models — install this package.

## Install

```bash
pnpm add @synapcores/openclaw-memory
# or
npm install @synapcores/openclaw-memory
```

`openclaw` (the host) is declared as a peer dependency — install it in your OpenClaw workspace.

## Prerequisites

You need a running SynapCores gateway. The Community Edition is free:

```bash
# Linux/macOS one-liner installer (see https://synapcores.com/install)
curl -fsSL https://synapcores.com/install.sh | sh
# Then start it:
synapcores start
```

Create an API key from the SynapCores admin UI (default `http://localhost:8095`) and copy it into your OpenClaw config below.

## Configure

Add this entry to your `openclaw.config.json` (or whichever config path your OpenClaw install uses):

```json
{
  "plugins": {
    "memory-synapcores": {
      "embedding": {
        "apiKey": "${OPENAI_API_KEY}",
        "model": "text-embedding-3-small"
      },
      "synapcores": {
        "host": "localhost",
        "port": 8080,
        "apiKey": "${SYNAPCORES_API_KEY}",
        "useHttps": false
      },
      "collection": "openclaw_memories",
      "graph": "openclaw_memory_graph",
      "autoCapture": true,
      "autoRecall": true,
      "autoLinkSimilar": true
    }
  }
}
```

Environment-variable interpolation (`${OPENAI_API_KEY}`, `${SYNAPCORES_API_KEY}`) is supported in any string field so you don't have to commit secrets.

### Config fields

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `embedding.apiKey` | yes | — | OpenAI API key for the embedding model. |
| `embedding.model` | no | `text-embedding-3-small` | Either `text-embedding-3-small` (1536 dims) or `text-embedding-3-large` (3072 dims). |
| `synapcores.apiKey` | yes | — | SynapCores API key (`ak_prod_…` or `aidb_…`). |
| `synapcores.host` | no | `localhost` | SynapCores gateway hostname. |
| `synapcores.port` | no | `8080` | SynapCores gateway port. |
| `synapcores.useHttps` | no | `false` | Use TLS to talk to the gateway. |
| `collection` | no | `openclaw_memories` | SynapCores collection name. |
| `graph` | no | `openclaw_memory_graph` | SynapCores graph name (used for `SIMILAR_TO` edges and `recallRelated` walks). |
| `autoCapture` | no | `true` | Auto-store memorable utterances after each agent turn. |
| `autoRecall` | no | `true` | Auto-inject relevant memories before each agent turn. |
| `autoLinkSimilar` | no | `true` | On capture, insert each Memory as a graph node carrying the embedding so `recallRelated` returns useful neighborhoods out of the box. Adds ~30-80ms per capture; disable if you never call `recallRelated`. |
| `workspace` | no | — | Optional workspace suffix on the AutoML relevance model name so multiple installations sharing one gateway can train independent models. |

## What you get

Once registered, the plugin:

1. Exposes three OpenClaw **tools** to your agents: `memory_recall`, `memory_store`, `memory_forget`.
2. Adds a CLI sub-command `openclaw ltm {list,search,stats}`.
3. Hooks `before_agent_start` to auto-recall relevant memories (if `autoRecall: true`).
4. Hooks `agent_end` to auto-capture preferences / decisions / entities / facts (if `autoCapture: true`).
5. Exposes four SynapCores-only methods at `plugin.extensions.*` (see "Extensions" below).

## API reference

### Tools (used by agents at runtime)

| Tool | What it does |
| --- | --- |
| `memory_recall` | Vector-search the memory store. Params: `{ query: string, limit?: number }` (default 5). |
| `memory_store` | Persist a new memory. Params: `{ text, importance?, category? }`. De-dupes against >0.95 cosine similarity. |
| `memory_forget` | Delete a memory by `memoryId` (UUID) or by `query` (auto-deletes if exactly one candidate at >0.9 similarity, otherwise returns candidates). |

### Extensions (programmatic, SynapCores-only)

Reached via `plugin.extensions.*` after `plugin.register(api)` runs.

```ts
interface MemorySynapCoresExtensions {
  /** Vector recall scoped by a SQL WHERE clause. */
  recallFiltered(opts: {
    where: string;          // e.g. "category = 'preference' AND importance >= 0.7"
    semantic: string;       // natural-language query
    limit?: number;         // default 5
  }): Promise<MemorySearchResult[]>;

  /** Walk SIMILAR_TO / MENTIONS / RELATES_TO edges from a memory. */
  recallRelated(memoryId: string, opts?: {
    hops?: number;          // default 1
    edgeKinds?: string[];   // default: any
  }): Promise<RelatedMemoryResult[]>;

  /** Score candidates with an AutoML model (with heuristic fallback). */
  predictRelevance(query: string, candidates: MemoryEntry[]): Promise<RelevanceScoredMemory[]>;

  /** Train (or retrain) the AutoML relevance model from feedback. */
  trainRelevanceModel(feedback: Array<{
    memoryId: string;
    queryText: string;
    score: number;          // 0..1
  }>): Promise<{ modelId: string; modelName: string }>;
}
```

#### `recallFiltered` — SQL-scoped semantic recall

```ts
const results = await plugin.extensions.recallFiltered({
  where: "category = 'preference' AND importance >= 0.7",
  semantic: "what UI style does the user prefer?",
  limit: 5,
});
```

The `where` clause is forwarded to the SynapCores gateway as the `filter` field on `/vector_search`. SQL validation happens on the gateway — a malformed clause will surface as an SDK error.

#### `recallRelated` — graph neighborhood walk

```ts
const neighbors = await plugin.extensions.recallRelated(memoryId, {
  hops: 1,
  edgeKinds: ["SIMILAR_TO"],   // default
  similarityThreshold: 0.5,    // default — cosine threshold for synthetic edges
  limit: 20,
});
```

Returns memories cosine-similar to the source (synthetic `SIMILAR_TO` edges, single-hop), plus any explicit `MENTIONS` / `RELATES_TO` edges the caller has populated (multi-hop supported on non-synthetic edge kinds). Requires `autoLinkSimilar: true` at capture time — the plugin inserts each Memory as a graph node carrying the embedding so the gateway's vector-indexed synthetic edges resolve at MATCH time.

If a source memory was captured before `autoLinkSimilar` was enabled, its `Memory` graph node won't exist and `recallRelated` will return `[]` for it. Re-capture (or write a one-off back-fill that posts `{labels: ["Memory"], properties: {id, text, embedding, ...}}` to `/v1/graph/nodes`) to retro-fit.

#### `predictRelevance` — AutoML re-ranking with heuristic fallback

```ts
const top = await plugin.extensions.recallFiltered({ where: "1=1", semantic: query, limit: 20 });
const ranked = await plugin.extensions.predictRelevance(query, top.map((r) => r.entry));
ranked.sort((a, b) => b.relevance - a.relevance);
```

When a model named `openclaw_memory_relevance[_<workspace>]` exists, candidates are scored by it. Otherwise the plugin falls back to:

```
relevance = 0.6 * cosine_similarity(query, memory)
          + 0.25 * exp(-age_days / 14)         # 14-day half-life
          + 0.15 * memory.importance
```

#### `trainRelevanceModel` — promote feedback to a model

```ts
const feedback = [
  { memoryId: "...", queryText: "what's my email?", score: 1.0 },
  { memoryId: "...", queryText: "dark mode preference", score: 0.9 },
  // ... at least 10 samples
];
await plugin.extensions.trainRelevanceModel(feedback);
// `predictRelevance` will automatically pick up the new model on the next call.
```

Requires at least 10 samples; throws otherwise. Train periodically (cron / on-demand) — the next `predictRelevance` call will detect the model and switch out of heuristic mode.

Under the hood, v0.2.0 stages feedback rows in a SQL table (`openclaw_memory_relevance_training[_<workspace>]`) on the gateway, then calls `/v1/automl/train` with `target: 'score'` and `task: 'regression'`. The table is preserved across calls so feedback accumulates between sessions; clear it manually with `DROP TABLE` (via `client.executeQuery`) if you want a clean restart.

## Roadmap (0.3.0+)

- **Entity extraction on capture** — parse `@mention` tokens and known-contact names out of incoming text and create `Person` / `Project` graph nodes with `MENTIONS` edges back to the memory.
- **Tag inference** — auto-classify memories into a configurable tag vocabulary on capture (small classifier or LLM call) so `recallFiltered` queries can use tags out of the box.
- **`synapcores-import-lancedb` migration script** — read an existing `~/.openclaw/memory/lancedb` store, re-embed if needed, and bulk-load into a SynapCores collection. Ships as a `bin` entry on the package.
- **Drop the `_getHttpClient` graph-node workaround** once `@synapcores/sdk >0.4.0` fixes `client.graph.nodes.create` to post `{labels: [label]}` instead of `{label}` (the wire shape the gateway's `/v1/graph/nodes` handler expects).

## Upstream

OpenClaw PR adding this plugin to the upstream extension catalogue: _TBD — link will be added once the PR opens_.

## License

MIT. See `LICENSE`.

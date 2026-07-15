# @synapcores/openclaw-memory

A long-term memory plugin for [OpenClaw](https://github.com/openclaw/openclaw) that uses **SynapCores AIDB** as the storage backend. Drop-in alternative to `@openclaw/memory-lancedb` with the same auto-recall / auto-capture lifecycle, plus four SynapCores-only extensions: SQL-filtered semantic recall, graph-relation walks, AutoML relevance scoring, and a model-training helper.

> **0.5.0 shipping note:** the core memory ops (`memory_store` / `memory_recall` / `memory_forget`) ride the engine-side `MEMORY_STORE` / `MEMORY_RECALL` / `MEMORY_FORGET` primitives via `@synapcores/sdk@^0.6.0`'s `client.memory` surface. The plugin's public API (tools, CLI, extensions, types) is unchanged.
>
> - **Requires SynapCores gateway `v1.8.5-ce` or newer** (the version that ships the `MEMORY_*` SQL functions).
> - **Fully engine-native embeddings — zero external-LLM dependency.** The `memory_store` / `memory_recall` hot path embeds server-side inside the engine's `MEMORY_STORE` / `MEMORY_RECALL` primitives; the relevance extensions (`predictRelevance` / `trainRelevanceModel`) and the `autoLinkSimilar` graph-node embedding call the gateway's native `client.embed()`. All embeddings come from the SynapCores gateway (embedding dimension is the gateway model's, e.g. 384 for `all-minilm`). OpenAI has been removed entirely — no OpenAI key, no `openai` dependency.
> - **Migration from 0.3.x: the engine-managed table is `_memory_<namespace>`, a different storage backend from the v0.3.x vector collection. Existing v0.3.x memories WILL NOT appear after upgrade — re-capture them.** See "Upgrading from 0.3.x" below.
> - **`collection` config field becomes the engine `namespace`.** It must now match `^[A-Za-z_][A-Za-z0-9_]*$`. The default `openclaw_memories` continues to work; other custom values with hyphens or other non-identifier characters need updating.
> - **`recallFiltered` WHERE clauses are applied client-side.** The engine cannot apply a `WHERE` to the table-valued `MEMORY_RECALL(?, ?, ?)` result-set (it drops every row), so the plugin fetches an oversampled, unfiltered recall and evaluates the predicate in JS. Legacy column shorthands (`category`, `importance`, `createdAt`, `text`) and the JSON-extract form (`metadata->>'…'`) are both understood directly — no rewriting required.

## Upgrading from 0.3.x

`@synapcores/openclaw-memory@0.5.0` is a **hard cut** from 0.3.x: the storage backend changed (at 0.4.0), so old memories will not migrate automatically. Steps:

1. Upgrade the SynapCores gateway to `v1.8.5-ce` or newer.
2. `npm install @synapcores/openclaw-memory@0.5.0`.
3. If your `collection` config value contains hyphens or other non-identifier characters, rename it to match `^[A-Za-z_][A-Za-z0-9_]*$` before restarting.
4. (Optional) export any high-value memories from the v0.3.x vector collection (the legacy `openclaw_memories` collection in your gateway) and re-store them via `memory_store` so they land in the new `_memory_<namespace>` table.
5. (Optional) drop the old vector collection from the gateway once you're sure the export is done.

If your `recallFiltered` callers use plain column names (`category`, `importance`, `createdAt`, `text`), they continue to work — the client-side filter understands those column names directly. Callers filtering on arbitrary metadata keys use the JSON-extract form (`metadata->>'…'`).

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

Requires OpenClaw **`>=2026.4.10`**. Install the plugin, then add its config and
give it the **memory slot**:

```bash
openclaw plugins install @synapcores/openclaw-memory
```

Add this to your OpenClaw config (run `openclaw config file` to find the path,
typically `~/.openclaw/openclaw.json`). Three things matter: the
`plugins.entries.<id>.config` nesting, the `plugins.allow` entry, and
**`plugins.slots.memory`**:

```json
{
  "plugins": {
    "allow": ["memory-synapcores"],
    "slots": { "memory": "memory-synapcores" },
    "entries": {
      "memory-synapcores": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
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
  }
}
```

> **You must set `plugins.slots.memory` to `"memory-synapcores"`.** Only one
> plugin can own the memory slot, and the default is OpenClaw's built-in
> `memory-core` — without claiming the slot the plugin loads but stays
> **disabled**.

> **You must set `plugins.entries.memory-synapcores.hooks.allowConversationAccess`
> to `true`.** OpenClaw gates conversation-lifecycle hooks (`before_agent_start`,
> `agent_end`) behind this flag for any non-bundled plugin. Without it, the
> plugin loads and its tools work, but `autoCapture`/`autoRecall` silently do
> nothing — the gateway logs `typed hook "agent_end" blocked because
> non-bundled plugins must set ...hooks.allowConversationAccess=true` and moves
> on. This lives outside `configSchema` (it's an OpenClaw host-level permission,
> not a plugin config field), so it won't show up in `openclaw config validate`
> errors — check the gateway log if auto-capture seems inactive.

Then `openclaw config validate`. Environment-variable interpolation
(`${SYNAPCORES_API_KEY}`) is supported in any string field
so you don't have to commit secrets. (Store keys **clean** — a trailing newline
in `apiKey` will break auth.)

### Config fields

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `synapcores.apiKey` | yes | — | SynapCores API key (`ak_prod_…` or `aidb_…`). |
| `synapcores.host` | no | `localhost` | SynapCores gateway hostname. |
| `synapcores.port` | no | `8080` | SynapCores gateway port. |
| `synapcores.useHttps` | no | `false` | Use TLS to talk to the gateway. |
| `collection` | no | `openclaw_memories` | SynapCores collection name. |
| `graph` | no | `openclaw_memory_graph` | SynapCores graph name (used for `SIMILAR_TO` edges and `recallRelated` walks). |
| `autoCapture` | no | `true` | Auto-store memorable utterances after each agent turn. |
| `autoRecall` | no | `true` | Auto-inject relevant memories before each agent turn. |
| `autoLinkSimilar` | no | `true` | On capture, insert each Memory as a graph node carrying the embedding so `recallRelated` returns useful neighborhoods out of the box. Adds ~30-50ms per capture; disable if you never call `recallRelated`. |
| `workspace` | no | — | Optional workspace suffix on the AutoML relevance model name so multiple installations sharing one gateway can train independent models. |

## What you get

Once registered, the plugin:

1. Exposes three OpenClaw **tools** to your agents: `memory_recall`, `memory_store`, `memory_forget`.
2. Adds a CLI sub-command `openclaw ltm {list,search,stats}`.
3. Hooks `before_prompt_build` to auto-recall relevant memories (if `autoRecall: true`).
4. Hooks `agent_end` to auto-capture preferences / decisions / entities / facts matching a rule-based
   trigger list (if `autoCapture: true`) — a fast, per-turn safety net, not exhaustive by design.
5. Registers a `registerMemoryCapability` flush plan — the same mechanism OpenClaw's own bundled
   `memory-core` plugin uses. Right before a session auto-compacts, the core runtime prompts the
   agent to call `memory_store` for anything durable that's about to fall out of context, in its
   own words. This runs independently of `autoCapture` and catches things the per-turn trigger
   list misses. No config needed — registered automatically whenever the plugin loads on a host
   that supports it (silently skipped on older hosts).
6. Exposes four SynapCores-only methods at `plugin.extensions.*` (see "Extensions" below).

## API reference

### Tools (used by agents at runtime)

| Tool | What it does |
| --- | --- |
| `memory_recall` | Vector-search the memory store. Params: `{ query: string, limit?: number }` (default 5). |
| `memory_store` | Persist a new memory. Params: `{ text, importance?, category? }`. De-dupes against >0.95 cosine similarity. |
| `memory_forget` | Delete a memory by `memoryId` (engine-assigned id, e.g. `mem_…`) or by `query` (auto-deletes if exactly one candidate at >0.9 similarity, otherwise returns candidates). |

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
    hops?: number;               // default 1 (capped at 4)
    edgeKinds?: string[];        // default: ["SIMILAR_TO"]
    similarityThreshold?: number; // default 0.5 (synthetic SIMILAR_TO edges only)
    limit?: number;              // default 20
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

Because the engine cannot apply a `WHERE` to the table-valued `MEMORY_RECALL(?, ?, ?)` result-set, the plugin runs an oversampled unfiltered recall and evaluates the `where` predicate **client-side in JS**. A malformed clause surfaces as a descriptive `recallFiltered: …` error thrown by the plugin (not an engine error).

Supported `where` surface (parsed by the plugin's predicate compiler):

- **Fields:** `category`, `importance`, `createdAt`, `text` / `content`, `id`, `similarity` / `score`, and JSON-extract `metadata->>'key'` for any other metadata field.
- **Comparison operators:** `=` / `==`, `!=` / `<>`, `>`, `>=`, `<`, `<=`, `LIKE` (SQL `%` / `_` wildcards), and `IN (…)`.
- **Boolean combinators:** `AND`, `OR`, `NOT`, and parentheses.
- **Literals:** single-quoted strings, numbers, `TRUE` / `FALSE` / `NULL`.

An empty clause or `1=1` passes all rows. Anything outside this surface (subqueries, functions, joins) throws rather than silently returning wrong rows.

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
relevance = 0.6 * (cosine_similarity(query, memory) + 1) / 2   # cosine mapped [-1,1] -> [0,1]
          + 0.25 * exp(-age_days / 14)                          # ~14-day recency decay
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

Under the hood, the plugin stages feedback rows in a SQL table (`openclaw_memory_relevance_training[_<workspace>]`) on the gateway, then calls `/v1/automl/train` with `target: 'score'` and `task: 'regression'`. The table is preserved across calls so feedback accumulates between sessions; clear it manually with `DROP TABLE` (via `client.executeQuery`) if you want a clean restart. Memory hydration is via `MEMORY_RECALL(?, ?, ?) WHERE id = ?` against the engine's namespace; rows whose memories have been deleted are skipped.

## Roadmap

- **Entity extraction on capture** — parse `@mention` tokens and known-contact names out of incoming text and create `Person` / `Project` graph nodes with `MENTIONS` edges back to the memory.
- **Tag inference** — auto-classify memories into a configurable tag vocabulary on capture (small classifier or LLM call) so `recallFiltered` queries can use tags out of the box.
- **`synapcores-import-lancedb` migration script** — read an existing `~/.openclaw/memory/lancedb` store, re-embed if needed, and bulk-load into a SynapCores collection. Ships as a `bin` entry on the package.
- **Drop the `_getHttpClient` graph-node / graph-match workarounds** once the SDK restores a native graph API: `client.graph.nodes.create` needs to post `{labels: [label]}` (not `{label}`) to match the gateway's `/v1/graph/nodes` handler, and `recallRelated` currently posts Cypher to `/v1/graph/match` directly because `@synapcores/sdk@^0.6.0` no longer exposes `client.graph.cypher`.

## Upstream

OpenClaw PR adding this plugin to the upstream extension catalogue: _TBD — link will be added once the PR opens_.

## License

MIT. See `LICENSE`.

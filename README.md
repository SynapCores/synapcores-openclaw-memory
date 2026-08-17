# @synapcores/openclaw-memory

A long-term memory plugin for [OpenClaw](https://github.com/openclaw/openclaw) that uses a **SynapCores AIDB** gateway you host as the storage backend. Drop-in alternative to `@openclaw/memory-lancedb`, plus four SynapCores-only extensions: SQL-filtered semantic recall, graph-relation walks, AutoML relevance scoring, and a model-training helper.

> **What it does with your conversations (read this first).** By default the
> plugin is **passive**: it stores or recalls memory only when you explicitly
> call a tool (`memory_store` / `memory_recall` / `memory_forget`) or the `ltm`
> command. It also offers an **opt-in automatic lifecycle** (`autoCapture` /
> `autoRecall`, both **off by default**). When you turn `autoCapture` on, the
> plugin becomes proactive: at the **end of every agent turn** and again **just
> before OpenClaw compacts the context**, it inspects the conversation, infers
> durable personal facts, preferences, and decisions, and **persists them (in
> the model's own words) to your SynapCores gateway** — and `autoRecall` injects
> matching memories back into the model's context on later turns. This is real
> long-term memory of personal conversation details; enable it deliberately. See
> [Privacy & data handling](#privacy--data-handling).

> **0.7.0 shipping note — the `CREATE MEMORY` backend.** SynapCores engine
> `v1.14.3-ce` introduced a first-class memory object: one logical thing with
> episodes, durable facts, temporal validity, relationships, provenance and an
> authoritative current state, driven by five verbs (`REMEMBER`, `RECALL`,
> `CURRENT`, `FORGET`, `TRACE`). The plugin now **detects it at runtime and uses
> it when it is there**, and keeps running on the legacy `MEMORY_*` primitives
> when it is not. Nothing to configure — `memory.backend` defaults to `auto`.
>
> What you get on `v1.14.3-ce+`:
>
> - **Two new tools.** `memory_current` — the authoritative present value of one
>   attribute, a deterministic state lookup rather than a similarity search, with
>   an explicit `conflict` outcome the agent must handle instead of guessing.
>   `memory_trace` — why a value is believed: the source statement verbatim, when
>   it was recorded versus when it happened, the extraction method, and what it
>   superseded.
> - **Recall that separates authority from similarity.** `memory_recall` and the
>   auto-recall hook now use the engine's `RECALL`, injecting its
>   `context.prompt_ready` projection instead of a hand-assembled list — so
>   current state outranks a semantically similar older episode, and the token
>   budget truncates the low-priority half rather than the authoritative half.
> - **Per-identity isolation, with no configuration.** `memory.identityFrom`
>   defaults to `credential`: the plugin sends no identity and the gateway
>   resolves it from the token's `memory_identity` pin, so the plugin cannot
>   address another identity because it never names one. **This needs an
>   ordinary user token** — API keys and admin tokens are never pinned; with
>   one of those, set `memory.identityFrom: "static"`. See
>   [Identity](#identity--who-a-memory-belongs-to).
> - **The engine owns dedup.** The old client-side "is this a near-duplicate?"
>   probe before every store is gone on this backend; consolidation, supersession
>   and conflict policy are the database's job.
> - **`confidence` and `relevance` stay separate.** Relevance is the retrieval
>   score; confidence is evidence strength. `MemorySearchResult.score` remains
>   relevance, and `confidence` is its own optional field.
>
> Trade-offs and the full mapping (including what the legacy path can do that
> the new one cannot — `importance` / `category` are **not persisted** on the
> memory object) are in [MIGRATION.md](./MIGRATION.md).

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
          "autoCapture": false,
          "autoRecall": false,
          "autoLinkSimilar": false
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

## Privacy & data handling

This plugin is **privacy-safe by default and passive out of the box.**

- **Automatic capture and recall are opt-in.** `autoCapture`, `autoRecall`, and
  `autoLinkSimilar` all default to `false`. With the defaults, the plugin never
  reads or writes memory on its own — it stores or recalls **only** when you
  explicitly invoke a tool (`memory_store`, `memory_recall`, `memory_forget`) or
  the `ltm` command.
- **What "opt-in automatic capture" actually does.** When you set
  `autoCapture: true`, the plugin subscribes to the `agent_end` conversation
  hook and, **at the end of every agent turn**, inspects that turn and stores the
  durable personal facts, preferences, and decisions it infers — paraphrased in
  the model's own words, not raw transcripts — to your gateway. Separately, the
  **pre-compaction flush** fires **just before OpenClaw discards older context**
  and prompts the agent to persist anything still worth keeping. Both require the
  host permission `plugins.entries.memory-synapcores.hooks.allowConversationAccess: true`.
  `autoRecall: true` then injects matching stored memories back into the model's
  context on later turns. Turn these on only where proactive persistence of
  personal conversation details is acceptable.
- **Pre-compaction flush is part of that opt-in.** The plugin only registers a
  pre-compaction memory-flush capability when `autoCapture` is enabled; with the
  default off, the agent is never prompted to store facts automatically.
- **You control your data.** Review memories with `memory_recall` / `ltm`, and
  delete them by id or query with `memory_forget` (GDPR-style forget).
- **No third-party egress or telemetry.** The plugin talks to exactly one
  endpoint — the **SynapCores gateway you configure** (`synapcores.host` /
  `apiKey`). It sends no analytics, usage data, or conversation content anywhere
  else. Your memories live in your own self-hosted database.
- **Prefer HTTPS + a controlled gateway.** Set `synapcores.useHttps: true` for
  any non-localhost gateway so memory content is encrypted in transit.
- **Don't store secrets or regulated data** in shared or sensitive workspaces,
  and confirm you can review/delete memories before enabling auto-capture there.

### Config fields

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `synapcores.apiKey` | yes | — | SynapCores API key (`ak_prod_…` or `aidb_…`). |
| `synapcores.host` | no | `localhost` | SynapCores gateway hostname. |
| `synapcores.port` | no | `8080` | SynapCores gateway port. |
| `synapcores.useHttps` | no | `false` | Use TLS to talk to the gateway. |
| `collection` | no | `openclaw_memories` | SynapCores collection name. Also the default name of the `CREATE MEMORY` object. |
| `graph` | no | `openclaw_memory_graph` | SynapCores graph name (used for `SIMILAR_TO` edges and `recallRelated` walks). |
| `memory.backend` | no | `auto` | `auto` \| `memory-object` \| `legacy`. `auto` probes once and picks; `memory-object` fails loudly on an engine older than v1.14.3-ce; `legacy` never probes. |
| `memory.name` | no | = `collection` | Name of the `CREATE MEMORY` object. Must match `^[A-Za-z_][A-Za-z0-9_]*$`, ≤64 chars — it becomes part of five table names. |
| `memory.identityField` | no | `user_id` | `IDENTITY <field>` used when the plugin creates the object. Ignored afterwards. |
| `memory.identityFrom` | no | `credential` | `credential` \| `static` \| `host` — see [Identity](#identity--who-a-memory-belongs-to). |
| `memory.identity` | no | — (`default` in `static` mode) | Identity value for `static` mode; tool-call fallback for `host` mode. Ignored in `credential` mode. |
| `memory.autoCreate` | no | `true` | Create the memory object on first use when the gateway supports it. |
| `memory.requireIsolation` | no | `false` | Fail instead of falling back to the legacy backend when per-identity isolation was asked for. |
| `memory.tokenBudget` | no | `512` | `token_budget` for prompt-injection `RECALL`. |
| `memory.sourceType` | no | `user_statement` | Claim authority for `memory_store` writes. |
| `autoCapture` | no | `true` | Auto-store memorable utterances after each agent turn. |
| `autoRecall` | no | `true` | Auto-inject relevant memories before each agent turn. |
| `autoLinkSimilar` | no | `true` | On capture, insert each Memory as a graph node carrying the embedding so `recallRelated` returns useful neighborhoods out of the box. Adds ~30-50ms per capture; disable if you never call `recallRelated`. |
| `workspace` | no | — | Optional workspace suffix on the AutoML relevance model name so multiple installations sharing one gateway can train independent models. |

## Identity — who a memory belongs to

The legacy backend keyed everything on one flat namespace shared by the whole
install. The `CREATE MEMORY` object splits that into an object plus a per-call
**identity**, and the engine guarantees reads and writes never cross identities.

The plugin never *guesses* an identity, because guessing wrong means one user
reading another user's memory. OpenClaw tool handlers receive no session or
user context (the host's `execute()` signature has none) — which is exactly why
the default lets the **gateway** supply what the host cannot:

| `memory.identityFrom` | Behaviour | Use it when |
| --- | --- | --- |
| `credential` (default) | The plugin sends **no** identity; the gateway resolves it from the token's `memory_identity` claim. The plugin cannot address another identity because it never names one. | Your credential is an ordinary user token. Needs no configuration. |
| `static` | Every call uses `memory.identity` (default `"default"`). One identity for the whole install — the same isolation the legacy shared namespace gave. | Your credential is an **API key** or an admin token (the gateway never pins those), and one host serves one person. |
| `host` | Derived per turn from the OpenClaw hook context (`senderId` → `accountId` → `sessionKey` → `sessionId`). Tool calls, which have no context, use `memory.identity` if set and otherwise **refuse**. | A multi-user host — a Discord/Slack bot serving many people through one backend credential. |

**Which credentials the gateway pins.** Ordinary interactive logins get
`memory_identity` set to the user's own id. **Admin tokens and API keys stay
unpinned on purpose**: an admin must be able to administer (and `FORGET` on
behalf of) other identities, and one service key legitimately serves many end
users, supplying the identity per request.

> **If you followed the README's setup and created a FullAccess API key**, that
> key is unpinned, so the default `credential` mode will refuse with
> `MEMORY_CREDENTIAL_UNPINNED` and tell you so at startup. Set
> `memory.identityFrom: "static"` (or `"host"` for a multi-user bot) — one line
> — or configure an ordinary user's token instead. The plugin refuses rather
> than quietly writing everything into a shared scope you did not choose.

A credential pinned to one identity that names a different one is refused with
HTTP 403 `MEMORY_ACCESS_DENIED`. The plugin surfaces that as its own error with
the fix in the message — it is never retried against another identity and never
reported as "the gateway is unreachable".

Identity is defence in depth, not the only boundary: each registered CE user
gets its own tenant, and memory objects are tenant-keyed, so two registered
users cannot reach each other's memory regardless of this setting.

**On a pre-v1.14.3 engine there is no identity dimension at all**: the legacy
namespace is shared and this whole section is moot — the plugin stays on the
legacy backend and never probes for a pin. If you *explicitly* asked for `host`
or `credential` on such a gateway it logs a prominent warning; set
`memory.requireIsolation: true` to make it a hard failure instead.

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
| `memory_recall` | Retrieve relevant memory. Params: `{ query: string, limit?: number }` (default 5). On the memory-object backend this is `RECALL`: the reply text is the engine's `context.prompt_ready` projection, and `details` keeps `currentState` separate from the ranked list. On legacy it is a vector search. |
| `memory_store` | Persist a memory. Params: `{ text, importance?, category? }`. Legacy de-dupes client-side against >0.95 cosine similarity; the memory-object backend does not, because the engine owns dedup. **`importance` and `category` are not persisted on the memory-object backend** (see MIGRATION.md §6). |
| `memory_forget` | Delete by `memoryId`, or by `query`. On the memory-object backend a query issues `FORGET … ABOUT` and returns the per-category counts the engine removed; on legacy it auto-deletes a single >0.9 match and otherwise returns candidates. |
| `memory_current` | **v1.14.3-ce+.** The authoritative current value of one attribute. Params: `{ attribute: string }`. Deterministic, never a similarity search. Returns the value with its authority and `valid_from`, or `not_found`, or `conflict` with both competing claims — which the engine deliberately does not resolve. |
| `memory_trace` | **v1.14.3-ce+.** Why a value is believed. Params: `{ attribute: string }`. Returns the source statement verbatim, recorded-vs-event time, extraction method, confidence and what it superseded. |

`memory_current` and `memory_trace` are registered unless `memory.backend` is
`legacy`; on an older engine they answer with a stable
`MEMORY_FEATURE_UNAVAILABLE` result rather than failing.

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

  /** Which backend this gateway resolved to. Runs detection on first call. */
  backendKind(): Promise<"legacy" | "memory-object">;

  /**
   * Force a consolidation cycle NOW (memory-object backend only).
   *
   * Optional control, never a prerequisite — the engine schedules its own
   * consolidation and current state is already correct the moment a write
   * returns. Deliberately NOT a model-facing tool: memory correctness must not
   * depend on an agent remembering to call it. Reach for it in a test, before
   * a report, or after a bulk import.
   */
  consolidateMemory(): Promise<MemoryEnvelope>;
}
```

On the memory-object backend, `recallFiltered` supports only the predicate
fields that backend actually has — `text` / `content`, `id`, `score` /
`similarity`, `confidence`. A predicate naming `category`, `importance`,
`createdAt` or `metadata->>'…'` **throws** rather than returning an empty
result set, because the memory object stores no per-memory metadata. Set
`memory.backend: "legacy"` to keep the metadata-filtering surface.

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

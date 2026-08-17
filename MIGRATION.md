# Migration plan — legacy `MEMORY_*` → `CREATE MEMORY` (SRS v1.2)

Status: **implemented in v0.7.0**. This document is the plan the implementation
followed, plus the record of what the new backend cannot do and how the plugin
keeps those cases working.

Everything marked **VERIFIED** below was exercised against a live SynapCores
engine (`v1.14.3`-class build, `http://127.0.0.1:18193`) while writing this.
Everything marked **REASONED** was derived from the engine source
(`crates/aidb-gateway/src/routes/memory.rs`, `crates/aidb-mcp-server/src/tools/sql_manual.rs`,
`AIDB_SQL_MANUAL.md` §"Persistent agent memory (v1.14.3+)") but not run.

---

## 1. What actually changed in the engine

The old surface is a flat, per-namespace key/vector store:

```sql
SELECT MEMORY_STORE('ns', 'text', '{"importance":0.8}');
SELECT * FROM MEMORY_RECALL('ns', 'query', 5);
SELECT MEMORY_FORGET('ns', 'mem_…');
```

`v1.14.3` adds a first-class memory **object**:

```sql
CREATE MEMORY openclaw_memories IDENTITY user_id;
REMEMBER openclaw_memories FOR user_id = 'u1' 'I prefer dark mode';
RECALL    openclaw_memories FOR user_id = 'u1' ABOUT 'ui preferences';
CURRENT   openclaw_memories FOR user_id = 'u1' ATTRIBUTE subscription_plan;
TRACE     openclaw_memories FOR user_id = 'u1' ATTRIBUTE subscription_plan;
FORGET    openclaw_memories FOR user_id = 'u1' ABOUT 'home address';
```

Both surfaces still exist. The old one is not deprecated by this plugin — it is
the fallback path for engines older than `v1.14.3`.

The differences that drive the design:

| | legacy `MEMORY_*` | `CREATE MEMORY` |
| --- | --- | --- |
| Scope unit | one flat `namespace` | one object + a per-call **identity** |
| Write | `MEMORY_STORE` inserts a row | `REMEMBER` persists an episode, then the engine derives facts, current state, temporal supersession, relationships and provenance |
| Read | cosine top-K, one flat list | `RECALL` returns `current_state` **separately** from `semantic` / `episodic` / `relationships` / `temporal` / `provenance`, plus `context.prompt_ready` |
| Exact fact | similarity search and hope | `CURRENT` — deterministic, `ok` \| `not_found` \| `conflict` |
| Provenance | none | `TRACE` |
| Dedup / upsert policy | the CLIENT's job (`MEMORY_UPSERT` policies, the plugin's own >0.95 similarity check) | the ENGINE's job (consolidation, supersession, conflict policy) |
| Response | a scalar or a result-set | a JSON envelope: `{status, operation, memory, identity, request_id, generated_at, data, warnings, error}` |
| Free-form metadata | yes (`metadata` TEXT blob) | **no** — see §6 |

The plugin talks to the object surface over **REST** (`/v1/memory/*`), not SQL.
That is deliberate: the gateway module's own header calls the OpenClaw plugin
its first consumer, it escapes every literal server-side (`lit()` / `ident()`),
and — the part that matters — it is where the identity pin is enforced
(`resolve_identity`, MEM-SEC-001). Composing `REMEMBER … FOR user_id = '…'`
client-side would move both the escaping and the authorization decision into
the plugin, which is exactly backwards.

---

## 2. Capability detection

`memory.backend` (config) selects the strategy:

| value | behaviour |
| --- | --- |
| `"auto"` (default) | probe, then pick; fall back to legacy on a negative probe |
| `"memory-object"` | probe, and **fail loudly** if the engine cannot serve it |
| `"legacy"` | never probe; use `MEMORY_*` |

The probe is one `GET /v1/memory` (the `SHOW MEMORIES` catalog), run lazily on
the first memory operation and memoized for the life of the plugin — the same
lazy shape the existing gateway preflight uses.

Signals, in order:

* **VERIFIED** a `v1.14.3` engine answers `200` with a JSON array of
  `{memory_name, agent, identity_field, episodic, …}`.
* **VERIFIED** any unrouted `/v1/*` path answers `404 text/plain` on this build
  (`is_api_path()` in `static_assets.rs` explicitly refuses to serve the SPA
  shell under `/v1/`), which the SDK raises as `NotFoundError`. So a pre-1.14.3
  engine — which has no `/v1/memory` route at all — produces a clean negative.
* Belt-and-braces: the probe also requires the payload to be an **array**. An
  older gateway whose SPA fallback did answer `200 text/html` under `/v1/`
  would otherwise read as a false positive.
* An auth failure or a connection failure is **not** a capability signal and is
  rethrown — silently dropping to the legacy backend because the API key is
  wrong would hide a misconfiguration behind a working-looking fallback.

`GET /v1/memory/` **with a trailing slash 404s** (**VERIFIED**) — it misses the
route and lands on the SPA fallback. The probe must use the unslashed form.

If the object named by `memory.name` (default: the `collection` value) is not in
the catalog and `memory.autoCreate` is on (default), the plugin issues
`POST /v1/memory {name, identity_field}` — i.e. `CREATE MEMORY <name> IDENTITY
<field>`, every other option left at the engine's defaults, per the manual's
"`CREATE MEMORY customer_memory IDENTITY user_id;` is a complete, working
object". A `MEMORY_ALREADY_EXISTS` here is treated as success (another OpenClaw
instance won the race).

The resolved decision is logged once at `info`:

```
memory-synapcores: memory backend = memory-object (object 'openclaw_memories', IDENTITY user_id, identity source: static)
memory-synapcores: memory backend = legacy MEMORY_* namespace 'openclaw_memories' (engine predates CREATE MEMORY / v1.14.3)
```

---

## 3. Identity — the key design decision

The legacy API keys on one flat `namespace` shared by everything the OpenClaw
host does. `CREATE MEMORY` splits that into an object (`IDENTITY user_id`) and a
per-call identity **value**, and the engine guarantees reads and writes never
cross identities.

The constraint that shapes the mapping: **OpenClaw tool `execute()` handlers
receive no session or user context.** The signature is
`(toolCallId, params, signal?, onUpdate?)` — that is the host's own type
(`openclaw/dist/types-*.d.ts`), not a plugin choice. Hook handlers *do* get a
`PluginHookAgentContext` (`senderId`, `accountId`, `sessionKey`, `sessionId`),
but tools do not. Deriving a per-user identity in a hook and caching it for a
later tool call would be a race in any host that runs two sessions at once, and
losing that race means one user reading another user's memory. So the plugin
never guesses an identity — it resolves one of three ways, and refuses when it
cannot.

`memory.identityFrom`:

* **`"static"` (default)** — every call uses `memory.identity` (default
  `"default"`). One identity for the whole install. This is *exactly* the
  isolation the legacy shared namespace gave, so upgrading changes nothing
  about who can see what; it is honest rather than better. Right for the normal
  OpenClaw deployment: one host, one person.
* **`"credential"`** — the plugin sends **no** identity at all and the gateway
  resolves it from the token's `memory_identity` claim. This is the strongest
  option: the plugin *cannot* address another identity, because it never names
  one. Requires a JWT minted with the pin. If the configured credential is
  unpinned, the gateway answers `400 missing_field` and the plugin raises an
  actionable error naming both fixes.
* **`"host"`** — per-turn, derived from the hook context in priority order
  `senderId` → `accountId` → `sessionKey` → `sessionId`. Correct for a
  multi-user host (a Discord/Slack bot). Tool calls, which have no context,
  use `memory.identity` if it is set and otherwise **refuse the call** with an
  actionable message. Refusing is the point: the alternative is a shared
  bucket, which is the leak.

Identity values are trimmed, capped at 128 chars, and rejected when empty. They
land in a SQL string literal on the engine side (`lit()`, quote-doubling), so
no client-side escaping is required or attempted — **REASONED**, from
`memory.rs`'s own module docs and the `lit` helper.

**Legacy backend + identity isolation.** The legacy namespace has no identity
dimension. Renaming the namespace per identity would strand every existing
install's memories, so the plugin does not do it. Instead: selecting
`identityFrom: "host"` or `"credential"` while the resolved backend is legacy
logs a one-time prominent warning, and `memory.requireIsolation: true` turns
that warning into a hard failure for operators who cannot ship without it.

### `MEMORY_ACCESS_DENIED` is handled explicitly

A pinned credential naming a different identity is refused with
HTTP 403 + `error.code = "MEMORY_ACCESS_DENIED"` before any SQL is composed
(**REASONED** — read in `resolve_identity`; not reproduced live because this
build exposes no route that mints a pinned JWT, see §7).

The `@synapcores/sdk` collapses **any** 403 into `AuthenticationError` with
`code: "AUTH_ERROR"`, which would make a memory-authorization refusal
indistinguishable from a bad API key. The gateway's real code survives in the
error body, so the plugin digs it out of `err.details.error.code` (the gateway
error envelope is `{error: {code, message, fields}, meta}` — **VERIFIED** on the
`MEMORY_NOT_FOUND` path). A denial is surfaced as its own error type with a
message naming the identity that was refused and the two ways to fix it; it is
never retried against another identity and never degraded into "gateway
unreachable".

---

## 4. Operation mapping

| plugin operation | legacy | memory-object |
| --- | --- | --- |
| `memory_store` tool | `MEMORY_STORE` + a client-side >0.95 similarity dedup probe | `REMEMBER` — **no dedup probe**; the engine owns dedup, supersession and conflict policy |
| `memory_recall` tool | `MEMORY_RECALL` top-K, flat list | `RECALL`; returns `context.prompt_ready` as the tool text, and `details` keeps `current_state` / `semantic` / `episodic` separate |
| auto-recall hook | hand-assembled `- [category] text` lines | `RECALL` with `prompt_ready = true`, `token_budget` from config; the engine's projection is injected verbatim |
| auto-capture hook | `MEMORY_STORE` per capturable turn, with a dedup probe per turn | `REMEMBER` per capturable turn, `source_type` from the turn's role |
| `memory_forget` (by id) | `MEMORY_FORGET(ns, id)` | `FORGET … MEMORY '<id>'` |
| `memory_forget` (by query) | search → 1 candidate above 0.9 auto-deletes, else return a candidate list | `FORGET … ABOUT '<query>'`; the engine resolves the scope and reports per-category counts |
| `memory_current` (new) | *unavailable* | `CURRENT … ATTRIBUTE <a>` — handles `ok` / `not_found` / **`conflict`** |
| `memory_trace` (new) | *unavailable* | `TRACE … ATTRIBUTE <a>` |
| `ltm list` / `stats` | `SELECT COUNT(*) FROM _memory_<ns>` | `SELECT COUNT(*) FROM _mem_<name>_episodes WHERE identity = …` |

Things deliberately dropped on the new path:

* **The client-side dedup probe before every store.** `memory_store` used to run
  a recall at `minScore 0.95` and return `action: "duplicate"`. That is the
  engine's job now (`REMEMBER` → consolidation collapses duplicate facts,
  conflict policy resolves contradictions), and doing it in the client both
  costs a round-trip and second-guesses the engine's supersession logic. The
  tool now reports the engine's own `changes` counters instead.
* **Hand-assembling recall context.** `context.prompt_ready` replaces the
  `- [category] text` join in the auto-recall hook.
* **Similarity search for exact facts.** `memory_current` exists so a model
  asking "what plan is this customer on?" gets a deterministic answer with an
  authority and a `valid_from`, not the top-1 of a cosine ranking.

Things deliberately **not** exposed to the model:

* `CONSOLIDATE`. The manual is explicit that consolidation is the engine's job
  and that calling it to make an answer correct is a bug worth reporting. It is
  available at `plugin.extensions.consolidateMemory()` for tests and
  before-a-report use, and nowhere else.
* `RECONCILE` / `SUPERSEDE` / `RELATE` / `HISTORY` / `SEARCH` /
  `EXPLAIN MEMORY`. Optional control, not part of the day-to-day contract.

### `confidence` vs `relevance`

They are different fields and the plugin never merges them.
`MemorySearchResult.score` continues to mean **relevance** (retrieval score) —
it is the field the legacy `similarity` mapped to and every existing caller
reads it that way. Evidence strength arrives as a separate, optional
`confidence` on the same result. Tool `details` carry both, named.

---

## 5. Response envelope

Every operational verb returns the SRS §5.8 envelope. The gateway wraps it in
its usual `{data, meta}`, and the SDK's response interceptor already unwraps
that layer, so `response.data` **is** the envelope.

`status` handling:

| status | plugin behaviour |
| --- | --- |
| `ok` / `accepted` / `completed` | normal result |
| `not_found` | a real answer, not an error — `memory_current` says so plainly |
| `conflict` | surfaced with `data.competing_claims`; the tool reports both claims and states that the engine will not pick |
| `error` | mapped through `error.code`; `retryable` read from `error.fields[retryable]` |

`conflict` is HTTP **200** by design (the gateway's own comment: an unresolved
contradiction is a successfully-computed answer the caller must handle), so it
arrives as a normal response body, not an SDK exception — **VERIFIED** for
`not_found`, **REASONED** for `conflict` (no contradiction could be staged on
the probe engine within the time budget).

---

## 6. What the old path can do that the new one cannot

1. **Arbitrary per-memory metadata.** `MEMORY_STORE(ns, text, metadata)` stores
   a free-form JSON blob; the plugin puts `{importance, category, createdAt}` in
   it. `REMEMBER` has **no** metadata option — **VERIFIED**: the engine answers
   `Unknown REMEMBER option 'metadata'. Valid: source_type, source_id,
   confidence, event_time, session_id, session_end, attribute, value, extract,
   classify, consolidate`.
   *Consequence:* on the memory-object backend, `importance` and `category` are
   **not persisted**. `memory_store` still accepts both (the tool signature is
   unchanged) and they still flow into the `Memory` graph node written by
   `autoLinkSimilar`, so `recallRelated` results keep carrying them — but they
   do not round-trip through recall.
   *Kept working:* nothing silently changes meaning. Entries recalled from the
   object backend report `importance: 0` and `category: "other"`, and the
   README says so.

2. **`recallFiltered` predicates over `category` / `importance` /
   `createdAt` / `metadata->>'…'`.** Those fields do not exist on the new
   backend. Rather than return a confidently empty result set, `recallFiltered`
   **throws** a descriptive error when the predicate names a field the active
   backend cannot supply, naming the two ways out (`memory.backend: "legacy"`,
   or filter on a supported field). Predicates restricted to the portable
   subset — `text`/`content`, `id`, `score`/`similarity`, and (new) `confidence`
   — run against `RECALL` output. Failing loudly beats being quietly wrong.

3. **Fetch-by-id (`db.get`), used by `trainRelevanceModel`.** There is no
   "get one memory" verb. Implemented against the object's own
   `_mem_<name>_episodes` table, which the manual documents as an ordinary
   table you may `SELECT` from — **VERIFIED**: columns
   `id, identity, content, embedding, source_type, source_id, session_id,
   event_time, recorded_at, confidence, authority_rank, actor,
   derivation_status, derivation_error, extractor_version, consolidated,
   suppressed, metadata`. `importance`/`category` come back at their defaults
   (see 1), so a relevance model trained on the object backend has two
   constant features. `trainRelevanceModel` warns when that is the case.

4. **A stable client-visible id for every stored item.** `MEMORY_STORE` returns
   one `mem_…` id. `REMEMBER` returns an `episode_id` **plus** a list of derived
   `memory_ids` — one write, several records. The plugin reports the
   `episode_id` as the memory's id, because that is the thing `FORGET … MEMORY
   '<id>'` and the graph node key need to agree on.

5. **Namespaces longer than 64 characters, or a `collection` that is not a
   valid identifier.** The gateway holds memory names to
   `^[A-Za-z_][A-Za-z0-9_]*$` and ≤64 chars. A longer `collection` is
   legacy-only; auto-detection logs why and stays on the legacy path.

Not a regression, but worth stating: **existing legacy memories are not
migrated.** A memory written through `MEMORY_STORE` lives in `_memory_<ns>` and
is invisible to `RECALL`. There is no server-side migration verb. Operators who
want their history on the new backend must re-store it; `memory.backend:
"legacy"` keeps the old data reachable in the meantime.

---

## 7. Engine-side findings

Recorded here because they were found while doing this work, not because the
plugin needs them fixed:

* **Documentation gap — `REMEMBER` options.** The engine accepts `source_id`,
  `classify` and `consolidate`, none of which appear in the `REMEMBER` syntax
  line in `sql_manual.rs` or in `AIDB_SQL_MANUAL.md`. **VERIFIED** from the
  engine's own error text.
* **`GET /v1/memory/` (trailing slash) 404s** while `GET /v1/memory` works.
  Harmless once known, but every other `/v1` collection route in the gateway is
  reachable both ways, so a client written from the docs can trip on it.
  **VERIFIED**.
* **No route mints a `memory_identity`-pinned JWT.** `Claims.memory_identity`
  is enforced everywhere it matters, but `/v1/auth/login` always emits
  `memory_identity: null` and nothing else in `aidb-gateway` sets it to
  `Some(_)` — so the pin is only reachable for deployments that mint their own
  JWTs with the gateway's signing secret. `identityFrom: "credential"` is
  implemented and documented against that reality. **VERIFIED** (login token
  decoded; `grep memory_identity` over `crates/aidb-gateway/src`).
* **`context.prompt_ready` is repetitive** on this build: the projection
  concatenated each fact and then each episode containing the same sentence
  (`"subscription plan is Enterprise. subscription plan is Enterprise I prefer
  dark mode… I prefer dark mode…"`). It is compact and usable, but it spends
  tokens restating itself. **VERIFIED** on the probe engine.
* **SDK gap (`@synapcores/sdk@0.6.1`)** — every HTTP 403 becomes
  `AuthenticationError`/`AUTH_ERROR`, discarding the gateway's
  `MEMORY_ACCESS_DENIED`. The code is still recoverable from
  `err.details.error.code`, which is what the plugin does. A future SDK could
  preserve `errorCode` on the thrown error and save every client this dance.

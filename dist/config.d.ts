/**
 * Which engine surface backs the plugin's memory operations.
 *
 * - `memory-object`: the v1.14.3+ `CREATE MEMORY` object and its five verbs
 *   (REMEMBER / RECALL / CURRENT / FORGET / TRACE), over `/v1/memory/*`.
 * - `legacy`: the flat `MEMORY_STORE` / `MEMORY_RECALL` / `MEMORY_FORGET`
 *   primitives (engine v1.8.5+). Still fully supported — it is the fallback
 *   for engines that predate the memory object.
 * - `auto` (default): probe the gateway once and pick.
 */
export declare const MEMORY_BACKENDS: readonly ["auto", "memory-object", "legacy"];
export type MemoryBackendMode = (typeof MEMORY_BACKENDS)[number];
/**
 * Where the per-call memory identity comes from. See MIGRATION.md §3.
 *
 * `credential` is the default: the plugin sends no identity and the gateway
 * resolves it from the token's `memory_identity` pin, so the plugin cannot
 * address another identity because it never names one. `static` and `host`
 * remain for credentials the gateway does not pin (API keys, admin tokens) and
 * for engines that predate pin minting.
 */
export declare const MEMORY_IDENTITY_SOURCES: readonly ["credential", "static", "host"];
export type MemoryIdentitySource = (typeof MEMORY_IDENTITY_SOURCES)[number];
export type MemoryObjectConfig = {
    backend: MemoryBackendMode;
    /** Memory-object name. Defaults to `collection`, so one name covers both backends. */
    name?: string;
    /** `IDENTITY <field>` used when the plugin has to CREATE the object. */
    identityField: string;
    identityFrom: MemoryIdentitySource;
    /**
     * True when the operator named `identityFrom` explicitly.
     *
     * Needed because the default is now `credential`: a plain install on a
     * pre-v1.14.3 gateway must not be warned that it "asked for isolation it
     * cannot have" when it asked for nothing at all.
     */
    identityFromExplicit: boolean;
    /** Static identity value (used by `identityFrom: "static"`, and as the tool-call fallback for `"host"`). */
    identity?: string;
    /** CREATE the memory object when the gateway has the feature but not the object. */
    autoCreate: boolean;
    /** Turn the "legacy backend cannot isolate identities" warning into a hard failure. */
    requireIsolation: boolean;
    /** `token_budget` passed to RECALL for the auto-recall hook's prompt injection. */
    tokenBudget: number;
    /** `source_type` (claim authority) for memories written through the memory_store tool. */
    sourceType: string;
    /**
     * `CREATE MEMORY ... [FOR AGENT a] IDENTITY f WITH (...)` tuning, applied
     * ONLY when the plugin creates the object (autoCreate). Pre-built from the
     * operator's `memory.create` block into the engine's request shape so the
     * REST layer passes it straight to `POST /v1/memory`: `agent` becomes
     * `FOR AGENT`, and `options` is the `WITH (...)` map. Both empty by default,
     * so an unconfigured install still creates the object on the engine's own
     * defaults — but an operator can now pin `embedding_model` / `dimensions` to
     * match the deployment's embedding space, toggle `llm_extraction`, and tune
     * the episodic/semantic/temporal/consolidation knobs, instead of every
     * memory object silently inheriting whatever the engine happened to default.
     */
    create: MemoryCreateRequest;
};
/** The engine-shaped CREATE MEMORY request the plugin sends when autoCreating. */
export type MemoryCreateRequest = {
    /** `FOR AGENT <agent>` — a top-level field on `POST /v1/memory`, not a WITH key. */
    agent?: string;
    /** The `WITH (...)` map, already in the engine's snake_case key form. */
    options: Record<string, unknown>;
};
export type MemoryConfig = {
    synapcores: {
        host: string;
        port: number;
        apiKey: string;
        useHttps: boolean;
    };
    collection?: string;
    graph?: string;
    memory: MemoryObjectConfig;
    autoCapture?: boolean;
    autoRecall?: boolean;
    /**
     * When true (default: false), the plugin auto-creates `SIMILAR_TO` graph
     * edges at capture time between a newly-stored memory and its top-K most
     * similar existing memories (cosine > 0.7). Adds ~30-50ms per capture but
     * makes `recallRelated` return useful results out of the box. Only has an
     * effect when `autoCapture` is also enabled.
     */
    autoLinkSimilar?: boolean;
    /**
     * Optional workspace identifier. Currently used only as a suffix on the
     * AutoML relevance model name (`openclaw_memory_relevance_<workspace>`)
     * so multiple OpenClaw installations sharing one SynapCores gateway can
     * each train their own relevance model.
     */
    workspace?: string;
};
export declare const MEMORY_CATEGORIES: readonly ["preference", "fact", "decision", "entity", "other"];
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];
export declare const memoryConfigSchema: {
    parse(value: unknown): MemoryConfig;
    uiHints: {
        "synapcores.host": {
            label: string;
            placeholder: string;
            help: string;
        };
        "synapcores.port": {
            label: string;
            placeholder: string;
            help: string;
        };
        "synapcores.apiKey": {
            label: string;
            sensitive: boolean;
            placeholder: string;
            help: string;
        };
        "synapcores.useHttps": {
            label: string;
            advanced: boolean;
            help: string;
        };
        collection: {
            label: string;
            placeholder: string;
            advanced: boolean;
            help: string;
        };
        graph: {
            label: string;
            placeholder: string;
            advanced: boolean;
            help: string;
        };
        "memory.backend": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.name": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.identityField": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.identityFrom": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.identity": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.autoCreate": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.requireIsolation": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.tokenBudget": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.sourceType": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.create.embeddingModel": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.create.dimensions": {
            label: string;
            advanced: boolean;
            help: string;
        };
        "memory.create.llmExtraction": {
            label: string;
            advanced: boolean;
            help: string;
        };
        autoCapture: {
            label: string;
            help: string;
        };
        autoRecall: {
            label: string;
            help: string;
        };
        autoLinkSimilar: {
            label: string;
            advanced: boolean;
            help: string;
        };
        workspace: {
            label: string;
            advanced: boolean;
            help: string;
        };
    };
};
//# sourceMappingURL=config.d.ts.map
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
export const MEMORY_BACKENDS = ["auto", "memory-object", "legacy"] as const;
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
export const MEMORY_IDENTITY_SOURCES = ["credential", "static", "host"] as const;
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

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_HOST = "localhost";
const DEFAULT_PORT = 8080;
const DEFAULT_COLLECTION = "openclaw_memories";
const DEFAULT_GRAPH = "openclaw_memory_graph";
const DEFAULT_IDENTITY_FIELD = "user_id";
const DEFAULT_IDENTITY = "default";
// The engine's RECALL default is generous; 512 keeps per-turn prompt injection
// affordable on the small local models CE ships with.
const DEFAULT_TOKEN_BUDGET = 512;
// Authority of a claim written via the memory_store tool. The engine's ladder
// is system_of_record > verified_user_statement > user_statement > tool_result
// > prior_user_statement > agent_derivation > agent_inference. A tool call is
// the agent writing down what the user said, so `user_statement` is the honest
// default; auto-capture overrides it per turn from the message role.
const DEFAULT_SOURCE_TYPE = "user_statement";

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function parseEnum<T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T,
  label: string,
): T {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value === "string" && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  throw new Error(`${label} must be one of: ${allowed.join(", ")}`);
}

/**
 * Parse the `memory` block — the CREATE MEMORY (SRS v1.2) backend controls.
 *
 * Every field has a default that reproduces the pre-0.7.0 behaviour as closely
 * as the new backend allows, so an existing config keeps working untouched:
 * backend auto-detects, and identity is a single static value (which is the
 * same isolation the flat legacy namespace gave — see MIGRATION.md §3).
 */
function parseMemoryBlock(value: unknown): MemoryObjectConfig {
  if (value !== undefined && (typeof value !== "object" || value === null || Array.isArray(value))) {
    throw new Error("memory config must be an object");
  }
  const mem = (value ?? {}) as Record<string, unknown>;
  assertAllowedKeys(
    mem,
    [
      "backend",
      "name",
      "identityField",
      "identityFrom",
      "identity",
      "autoCreate",
      "requireIsolation",
      "tokenBudget",
      "sourceType",
    ],
    "memory config",
  );

  const identityFrom = parseEnum(
    mem.identityFrom,
    MEMORY_IDENTITY_SOURCES,
    "credential",
    "memory.identityFrom",
  );
  const identity =
    typeof mem.identity === "string" && mem.identity.trim().length > 0
      ? mem.identity.trim()
      : undefined;

  // `static` has to resolve to *something*, and an empty string is not a valid
  // identity on the engine side (resolve_identity rejects it), so fill the
  // documented default rather than failing an otherwise-valid install.
  const staticIdentity = identityFrom === "static" ? (identity ?? DEFAULT_IDENTITY) : identity;

  const tokenBudget =
    mem.tokenBudget === undefined
      ? DEFAULT_TOKEN_BUDGET
      : typeof mem.tokenBudget === "number" &&
          Number.isInteger(mem.tokenBudget) &&
          mem.tokenBudget > 0
        ? mem.tokenBudget
        : (() => {
            throw new Error("memory.tokenBudget must be a positive integer");
          })();

  return {
    backend: parseEnum(mem.backend, MEMORY_BACKENDS, "auto", "memory.backend"),
    identityFromExplicit: mem.identityFrom !== undefined,
    name: typeof mem.name === "string" && mem.name.length > 0 ? mem.name : undefined,
    identityField:
      typeof mem.identityField === "string" && mem.identityField.length > 0
        ? mem.identityField
        : DEFAULT_IDENTITY_FIELD,
    identityFrom,
    identity: staticIdentity,
    autoCreate: mem.autoCreate !== false,
    requireIsolation: mem.requireIsolation === true,
    tokenBudget,
    sourceType:
      typeof mem.sourceType === "string" && mem.sourceType.length > 0
        ? mem.sourceType
        : DEFAULT_SOURCE_TYPE,
  };
}

function parsePort(value: unknown): number {
  if (value === undefined) {
    return DEFAULT_PORT;
  }
  if (typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 65535) {
    return value;
  }
  throw new Error("synapcores.port must be an integer between 1 and 65535");
}

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      [
        "synapcores",
        "collection",
        "graph",
        "memory",
        "autoCapture",
        "autoRecall",
        "autoLinkSimilar",
        "workspace",
      ],
      "memory config",
    );

    const synapcores = (cfg.synapcores ?? {}) as Record<string, unknown>;
    assertAllowedKeys(
      synapcores,
      ["host", "port", "apiKey", "useHttps"],
      "synapcores config",
    );

    const host =
      typeof synapcores.host === "string" && synapcores.host.length > 0
        ? synapcores.host
        : DEFAULT_HOST;
    const port = parsePort(synapcores.port);
    const useHttps = synapcores.useHttps === true;

    // The API key may come from config OR from the environment. The README has
    // always documented `${SYNAPCORES_API_KEY}`, but requiring the config key
    // meant the environment variable alone could never satisfy it.
    //
    // Resolution order: explicit config value, then SYNAPCORES_API_KEY.
    // A MISSING key is NOT fatal here. It used to throw, which — because
    // OpenClaw validates plugin config at install time — made
    // `openclaw plugins install` abort with a Node stack trace before the
    // plugin could ever be configured. There is no install-then-configure
    // path if this throws. The gateway preflight reports the missing key at
    // runtime with an actionable message instead.
    const configuredKey =
      typeof synapcores.apiKey === "string" ? resolveEnvVars(synapcores.apiKey) : "";
    const apiKey = configuredKey || process.env.SYNAPCORES_API_KEY || "";

    return {
      synapcores: {
        host,
        port,
        apiKey,
        useHttps,
      },
      collection: typeof cfg.collection === "string" ? cfg.collection : DEFAULT_COLLECTION,
      graph: typeof cfg.graph === "string" ? cfg.graph : DEFAULT_GRAPH,
      memory: parseMemoryBlock(cfg.memory),
      // Privacy-safe by default: all automatic capture/recall/linking of
      // conversation-derived facts is OPT-IN. With these off the plugin is
      // passive — it only reads or writes memory when the user explicitly
      // invokes a tool (memory_store / memory_recall / memory_forget) or the
      // `ltm` command. Set each to `true` to enable the automatic lifecycle.
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall === true,
      autoLinkSimilar: cfg.autoLinkSimilar === true,
      workspace: typeof cfg.workspace === "string" ? cfg.workspace : undefined,
    };
  },
  uiHints: {
    "synapcores.host": {
      label: "SynapCores Host",
      placeholder: DEFAULT_HOST,
      help: "Hostname of the SynapCores gateway",
    },
    "synapcores.port": {
      label: "SynapCores Port",
      placeholder: String(DEFAULT_PORT),
      help: "Port of the SynapCores gateway",
    },
    "synapcores.apiKey": {
      label: "SynapCores API Key",
      sensitive: true,
      placeholder: "ak_prod_... or aidb_...",
      help: "API key for the SynapCores gateway (or use ${SYNAPCORES_API_KEY})",
    },
    "synapcores.useHttps": {
      label: "Use HTTPS",
      advanced: true,
      help: "Connect to the SynapCores gateway over TLS",
    },
    collection: {
      label: "Memory Collection",
      placeholder: DEFAULT_COLLECTION,
      advanced: true,
      help: "SynapCores collection name used to store memories",
    },
    graph: {
      label: "Memory Graph",
      placeholder: DEFAULT_GRAPH,
      advanced: true,
      help: "SynapCores graph name used for memory relations (SIMILAR_TO edges + recallRelated walks).",
    },
    "memory.backend": {
      label: "Memory Backend",
      advanced: true,
      help: "auto (default) probes the gateway and uses the v1.14.3+ CREATE MEMORY object when available, falling back to the legacy MEMORY_* namespace. Force with 'memory-object' or 'legacy'.",
    },
    "memory.name": {
      label: "Memory Object Name",
      advanced: true,
      help: "Name of the CREATE MEMORY object. Defaults to the collection name. Must match ^[A-Za-z_][A-Za-z0-9_]*$ and be at most 64 characters.",
    },
    "memory.identityField": {
      label: "Memory Identity Field",
      advanced: true,
      help: "IDENTITY field used when the plugin creates the memory object (default user_id). Ignored once the object exists — the gateway reads the field from the object.",
    },
    "memory.identityFrom": {
      label: "Memory Identity Source",
      advanced: true,
      help: "credential (default) sends no identity and lets the gateway resolve the token's memory_identity pin — the plugin cannot address another identity because it never names one. Requires an ordinary user token; API keys and admin tokens are never pinned, so use static (one identity for this install) or host (derive per turn from the OpenClaw sender/session) with those.",
    },
    "memory.identity": {
      label: "Memory Identity",
      advanced: true,
      help: "Identity value used when identityFrom is 'static' (defaults to 'default'), and the tool-call fallback when identityFrom is 'host'. Ignored in credential mode.",
    },
    "memory.autoCreate": {
      label: "Auto-Create Memory Object",
      advanced: true,
      help: "Create the memory object on first use when the gateway supports it but the object does not exist yet.",
    },
    "memory.requireIsolation": {
      label: "Require Identity Isolation",
      advanced: true,
      help: "Fail startup rather than fall back to the legacy backend when per-identity isolation was requested. The legacy MEMORY_* namespace is shared by every identity.",
    },
    "memory.tokenBudget": {
      label: "Recall Token Budget",
      advanced: true,
      help: "token_budget passed to RECALL when injecting memory context into a prompt (default 512).",
    },
    "memory.sourceType": {
      label: "Memory Source Type",
      advanced: true,
      help: "Claim authority recorded for memories written via memory_store (default user_statement). Higher authority wins conflicts: system_of_record > verified_user_statement > user_statement > tool_result > agent_derivation > agent_inference.",
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
    autoLinkSimilar: {
      label: "Auto-Link Similar Memories",
      advanced: true,
      help: "On capture, draw SIMILAR_TO graph edges to existing memories with cosine similarity > 0.7 so `recallRelated` returns useful neighborhoods (adds ~30-50ms per capture).",
    },
    workspace: {
      label: "Workspace ID",
      advanced: true,
      help: "Optional workspace name suffixed onto the AutoML relevance model so multiple installations sharing one gateway can train independent models.",
    },
  },
};

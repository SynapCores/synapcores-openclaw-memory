export type MemoryConfig = {
  synapcores: {
    host: string;
    port: number;
    apiKey: string;
    useHttps: boolean;
  };
  collection?: string;
  graph?: string;
  autoCapture?: boolean;
  autoRecall?: boolean;
  /**
   * When true (default), the plugin auto-creates `SIMILAR_TO` graph edges at
   * capture time between a newly-stored memory and its top-K most similar
   * existing memories (cosine > 0.7). Adds ~30-50ms per capture but makes
   * `recallRelated` return useful results out of the box. Set to false if
   * you only need vector recall or you plan to wire your own edge-building
   * pipeline.
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
        "autoCapture",
        "autoRecall",
        "autoLinkSimilar",
        "workspace",
      ],
      "memory config",
    );

    const synapcores = cfg.synapcores as Record<string, unknown> | undefined;
    if (!synapcores || typeof synapcores.apiKey !== "string") {
      throw new Error("synapcores.apiKey is required");
    }
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

    return {
      synapcores: {
        host,
        port,
        apiKey: resolveEnvVars(synapcores.apiKey),
        useHttps,
      },
      collection: typeof cfg.collection === "string" ? cfg.collection : DEFAULT_COLLECTION,
      graph: typeof cfg.graph === "string" ? cfg.graph : DEFAULT_GRAPH,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
      autoLinkSimilar: cfg.autoLinkSimilar !== false,
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

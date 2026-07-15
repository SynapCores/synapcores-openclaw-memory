import type { OpenClawPluginDefinition } from "openclaw/plugin-sdk/core";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// Lightweight CLI-metadata entry: lets `openclaw plugins` list the `ltm`
// command + its subcommands without loading the full plugin runtime. The
// actual command implementation (list / search / stats) is wired via
// `api.registerCli` inside the main plugin entry (index.ts).
const cliMetadataEntry: OpenClawPluginDefinition = definePluginEntry({
  id: "memory-synapcores",
  name: "Memory (SynapCores)",
  description: "SynapCores-backed long-term memory provider",
  register(api) {
    api.registerCli(() => {}, {
      descriptors: [
        {
          name: "ltm",
          description: "Inspect and query SynapCores-backed memory",
          hasSubcommands: true,
        },
      ],
    });
  },
});

export default cliMetadataEntry;

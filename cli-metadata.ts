import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// Lightweight CLI-metadata entry: lets `openclaw plugins` list the `ltm`
// command + its subcommands without loading the full plugin runtime. The
// actual command implementation (list / search / stats) is wired via
// `api.registerCli` inside the main plugin entry (index.ts).
// Annotated with `ReturnType<typeof definePluginEntry>` so the emitted default
// export declaration references only the public `definePluginEntry` symbol,
// avoiding TS2742 (inferred type pointing at an internal openclaw module path).
const cliMetadata: ReturnType<typeof definePluginEntry> = definePluginEntry({
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

export default cliMetadata;

# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities to the maintainers privately (open a
GitHub security advisory on this repository) rather than filing a public issue.

## Published-package attack surface

This plugin publishes a deliberately small surface. The `files` whitelist in
`package.json` ships only:

- `dist/` (compiled JavaScript + type declarations)
- `openclaw.plugin.json`, `README.md`, `LICENSE`

Its declared **runtime** dependencies are only:

- `@synapcores/sdk` (`^0.6.0`)
- `typebox` (`^1.1.38`)

`openclaw` is a **peer** dependency (provided by the host), and `vitest` /
`typescript` / `@types/node` are **dev** dependencies. None of these — nor their
transitive trees — are installed into a consumer's runtime when they depend on
this package.

## Residual `npm audit` findings (dev-only, do not ship)

As of 2026-07-15 the repository has **4 residual advisories**, all confined to
the `vitest` dev-toolchain. They are **not present in the published package's
runtime dependency tree** and are never installed by consumers of this plugin.

| Package    | Severity | Advisory                                                                 | Where it comes from                     |
|------------|----------|--------------------------------------------------------------------------|-----------------------------------------|
| `vitest`   | critical | [GHSA-5xrq-8626-4rwp](https://github.com/advisories/GHSA-5xrq-8626-4rwp) | dev dependency (test runner)            |
| `vite`     | high     | [GHSA-fx2h-pf6j-xcff](https://github.com/advisories/GHSA-fx2h-pf6j-xcff) (+ GHSA-4w7w-66w2-5vf9, GHSA-v6wh-96g9-6wx3) | transitive via `vitest`                 |
| `esbuild`  | moderate | [GHSA-67mh-4wv8-2f99](https://github.com/advisories/GHSA-67mh-4wv8-2f99) | transitive via `vitest` → `vite`        |
| `vite-node`| moderate | (no direct advisory; flagged via vulnerable `vite`)                      | transitive via `vitest`                 |

### Why these are not remediated in-place

All four collapse to a single fix: bumping `vitest` from `1.6.x` to `3.2.7`,
which npm reports as `isSemVerMajor: true` — a **breaking major upgrade of a
dev-only test runner**. Forcing it (`npm audit fix --force`) would destabilize
the test toolchain without changing anything a consumer installs. The upgrade is
deferred to a dedicated tooling PR rather than bundled into a security-hygiene
lockfile pass.

Additional risk context:

- The critical `vitest` advisory (GHSA-5xrq-8626-4rwp) is only exploitable when
  the **Vitest UI/API server** is running (`--ui` / `--api`). This project runs
  tests headless via `vitest run`; it never starts that server.
- The `vite` / `esbuild` advisories concern the **local dev server** (dev-server
  request handling, `server.fs.deny` bypass on Windows). No dev server is run in
  CI or in the shipped package.

## Remediated in this pass

- **`ws`** — [GHSA-96hv-2xvq-fx4p](https://github.com/advisories/GHSA-96hv-2xvq-fx4p)
  (high, memory-exhaustion DoS). Updated in `package-lock.json` from `8.20.1` to
  the patched `8.21.1`. `ws` reaches this package only transitively through
  `@synapcores/sdk` (`ws: ^8.16.0`); the patched `8.21.1` is within that range,
  so consumers resolve to a fixed version on a fresh install with no breaking
  change.

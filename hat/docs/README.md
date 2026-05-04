# Architecture Decision Records

* [ADR-001 — directory layout](adr-001-directory-layout.md)
* [ADR-002 — raw vs curated memory separation](adr-002-raw-vs-curated.md)
* [ADR-003 — model backend protocol](adr-003-backend-protocol.md)
* [ADR-004 — model lifecycle (catalog, hot-swap, unload)](adr-004-model-lifecycle.md)

## Per-package READMEs

For directory-level docs see the README at the top of each subpackage:

* [src/hat/api/README.md](../src/hat/api/README.md) — REST surface, streaming, deps
* [src/hat/core/README.md](../src/hat/core/README.md) — paper algorithms, Cortex contract
* [src/hat/memory/README.md](../src/hat/memory/README.md) — raw vs curated stores
* [src/hat/models/README.md](../src/hat/models/README.md) — backends, catalog, manager
* [src/hat/ui/README.md](../src/hat/ui/README.md) — vanilla HTML/CSS/JS web app
* [src/hat/data/README.md](../src/hat/data/README.md), [src/hat/eval/README.md](../src/hat/eval/README.md), [src/hat/services/README.md](../src/hat/services/README.md) — stubs

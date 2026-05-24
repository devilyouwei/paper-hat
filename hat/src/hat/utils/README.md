# `utils/` — cross-cutting helpers

Small, dependency-light utilities shared across the codebase.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker; no exports. |
| `logging.py` | Loguru wrapper. `setup()` configures a single stderr (and optional file) sink with a consistent format; `get_logger(__name__)` returns a per-module bound logger. Idempotent — importing from many places never duplicates sinks. Honours `HAT_LOG_LEVEL`. |

Typical use:

```python
from hat.utils.logging import get_logger

log = get_logger(__name__)
log.info("loaded model {}", path)
```

"""HTTP controllers.

Thin FastAPI routers that handle request/response shaping, status-code
mapping, and dependency injection. All business logic lives in
:mod:`hat.api.services`. ``main.py`` mounts each router under its prefix.
"""


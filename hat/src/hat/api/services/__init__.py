"""Service layer.

Pure business logic invoked by HTTP controllers. Each module owns one
domain (chat, models, embedding models, curated memory, sessions, OpenAI
compatibility). Singletons (the wake/sleep loop, session store, raw log)
live in :mod:`.container`.
"""

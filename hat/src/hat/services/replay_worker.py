"""Background worker that consumes a job queue and runs SWS cycles. Stub.

In production this would be a separate process (Celery / RQ / asyncio Task)
that owns the GPU and is the only writer of model checkpoints.
"""

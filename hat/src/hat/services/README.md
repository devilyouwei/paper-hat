# `services/` — long-running orchestration

* `sws_scheduler.py` — fires an SWS cycle every N interactions (paper §3.8).
* `replay_worker.py` — background process that owns the GPU and is the sole
  writer of model checkpoints. Stub.
* `job_queue.py` — in-process queue placeholder; swap for Redis/RQ in production.

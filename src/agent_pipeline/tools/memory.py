"""Working memory: short-term, task-scoped state owned by the Harness.

Backed by an in-memory dict for dev/test; the same interface accepts a
checkpointer-backed store (Postgres) in production. Scoped by request_id so one
pipeline run cannot read another's scratch.
"""


class WorkingMemory:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, object]] = {}

    def save(self, scope: str, key: str, value: object) -> None:
        self._store.setdefault(scope, {})[key] = value

    def load(self, scope: str, key: str) -> object | None:
        return self._store.get(scope, {}).get(key)

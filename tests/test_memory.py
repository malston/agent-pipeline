"""Working memory is task-scoped short-term state the Harness owns.

Each pipeline run (request_id) is an isolated scope so one task cannot read
another's scratch.
"""
from agent_pipeline.tools.memory import WorkingMemory


def test_save_then_load_returns_value():
    mem = WorkingMemory()
    mem.save("r1", "candidates", ["s1", "s2"])
    assert mem.load("r1", "candidates") == ["s1", "s2"]


def test_load_missing_key_returns_none():
    mem = WorkingMemory()
    assert mem.load("r1", "nope") is None


def test_scopes_are_isolated():
    mem = WorkingMemory()
    mem.save("r1", "k", "one")
    mem.save("r2", "k", "two")
    assert mem.load("r1", "k") == "one"
    assert mem.load("r2", "k") == "two"

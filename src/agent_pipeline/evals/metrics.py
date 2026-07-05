"""Retrieval-quality metrics. Deterministic, no Model, no keys."""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top-k retrieved ids."""
    if k <= 0:
        raise ValueError(f"recall_at_k needs k >= 1, got {k}")
    if not relevant:
        raise ValueError("recall_at_k needs a non-empty relevant set")
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant id (1-indexed); 0.0 if none is relevant."""
    for index, source_id in enumerate(retrieved, start=1):
        if source_id in relevant:
            return 1.0 / index
    return 0.0


def set_recall(predicted: list[str], relevant: set[str]) -> float:
    """Fraction of the relevant ids that were predicted (order-independent)."""
    if not relevant:
        raise ValueError("set_recall needs a non-empty relevant set")
    return len(set(predicted) & relevant) / len(relevant)


def set_precision(predicted: list[str], relevant: set[str]) -> float:
    """Fraction of the predicted ids that are relevant; 1.0 if nothing predicted."""
    predicted_set = set(predicted)
    if not predicted_set:
        return 1.0
    return len(predicted_set & relevant) / len(predicted_set)

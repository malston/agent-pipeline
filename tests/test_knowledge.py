"""Semantic retrieval is the RAG quality gate for the whole pipeline.

Real embeddings, real vector store, no mocks. If these pass, A1 can retrieve
relevant evidence; if they fail, every downstream stage is starved.
"""


def test_search_ranks_semantically_relevant_passage_first(knowledge):
    # Query shares no keywords with the mitochondrion sentence; retrieval must
    # be semantic, not lexical.
    passages = knowledge.search("how do cells produce energy", k=3)
    assert passages[0].source_id in {"mito", "photo"}
    assert passages[-1].source_id == "econ"  # off-topic doc ranks last


def test_search_respects_k(knowledge):
    passages = knowledge.search("how do cells produce energy", k=1)
    assert len(passages) == 1


def test_search_scores_within_unit_interval(knowledge):
    passages = knowledge.search("inflation and interest rates", k=3)
    assert all(0.0 <= p.score <= 1.0 for p in passages)


def test_search_filters_by_metadata(knowledge):
    passages = knowledge.search(
        "energy", k=3, filters={"topic": "economics"}
    )
    assert {p.source_id for p in passages} == {"econ"}

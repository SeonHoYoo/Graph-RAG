from utils.graph import search_query_graph_bindings


def test_binding_search_question_vs_doc():
    query_triples = [
        '(ENT2) [SEP] plays [SEP] (ENT1) [PREP] in "A League of Their Own"',
        '(ENT2) [SEP] has_attribute [SEP] "acclaimed role model business woman"',
    ]
    fact_triples = [
        'Mae Mordabito [SEP] is played by [SEP] Madonna [PREP] in "A League of Their Own"',
        "Madonna [SEP] is [SEP] a role model for businesswomen",
    ]
    result = search_query_graph_bindings(
        query_triples,
        fact_triples,
        top_k=3,
        beam_size=10,
        cand_per_query=10,
        min_token_jaccard=0.4,
    )
    assert result["bindings"]
    top1 = result["bindings"][0]
    assert top1["binding"].get("ENT1")
    assert top1["binding"].get("ENT2")
    assert "madonna" in top1["binding"]["ENT2"].lower()


def test_binding_search_wrong_entity_low_support():
    query_triples = [
        'Kathleen Turner [SEP] plays [SEP] Rosie the Riveter [PREP] in "A League of Their Own"',
        'Kathleen Turner [SEP] has_attribute [SEP] "acclaimed role model businesswoman"',
    ]
    fact_triples = [
        'Mae Mordabito [SEP] is played by [SEP] Madonna [PREP] in "A League of Their Own"',
        "Madonna [SEP] is [SEP] a role model for businesswomen",
    ]
    result = search_query_graph_bindings(
        query_triples,
        fact_triples,
        top_k=3,
        beam_size=10,
        cand_per_query=10,
        min_token_jaccard=0.4,
    )
    top1 = result["bindings"][0] if result["bindings"] else None
    if top1:
        assert top1["score"] < 0.6


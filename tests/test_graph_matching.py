from utils.graph import Graph


def test_gold_vs_doc_has_overlap_token_jaccard():
    gold_triples = [
        "Madonna [SEP] is [SEP] a role model for businesswomen",
        "Madonna [SEP] has generated [SEP] over $1.2 billion in sales",
    ]
    doc_triples = [
        "Madonna [SEP] has received acclaim as [SEP] a role model for businesswomen",
        "Madonna [SEP] has generated [SEP] over $1.2 billion in sales",
    ]
    gold_graph = Graph([], gold_triples)
    doc_graph = Graph([], doc_triples)
    result = gold_graph.compare_with(
        doc_graph,
        match_mode="token_jaccard",
        min_token_jaccard=0.5,
        include_definitions=False,
        ignore_ent_placeholders=True,
    )
    assert result["triplet_overlap"] >= 1


def test_question_template_matches_doc_with_placeholder_and_attribute_equivalence():
    question_triples = [
        '(ENT2) [SEP] plays [SEP] (ENT1) [PREP] in "A League of Their Own"',
        '(ENT2) [SEP] has_attribute [SEP] "acclaimed role model business woman"',
    ]
    doc_triples = [
        'Mae Mordabito [SEP] is played by [SEP] Madonna [PREP] in "A League of Their Own"',
        "Madonna [SEP] is [SEP] a role model for businesswomen",
    ]
    question_graph = Graph([], question_triples)
    doc_graph = Graph([], doc_triples)
    result = question_graph.compare_with(
        doc_graph,
        match_mode="token_jaccard",
        min_token_jaccard=0.5,
        include_definitions=False,
        ignore_ent_placeholders=True,
    )
    assert result["triplet_overlap"] >= 1


def test_wrong_entity_stays_unmatched():
    cot_triples = [
        'Kathleen Turner [SEP] plays [SEP] Rosie the Riveter [PREP] in "A League of Their Own"',
        'Kathleen Turner [SEP] has_attribute [SEP] "acclaimed role model businesswoman"',
    ]
    doc_triples = [
        'Mae Mordabito [SEP] is played by [SEP] Madonna [PREP] in "A League of Their Own"',
        "Madonna [SEP] is [SEP] a role model for businesswomen",
    ]
    cot_graph = Graph([], cot_triples)
    doc_graph = Graph([], doc_triples)
    result = cot_graph.compare_with(
        doc_graph,
        match_mode="token_jaccard",
        min_token_jaccard=0.5,
        include_definitions=False,
        ignore_ent_placeholders=True,
    )
    assert result["triplet_overlap"] == 0


"""TASI core: triple alignment with structural importance."""

from graphqa.tasi.embedding import SentenceEncoder, get_default_encoder
from graphqa.tasi.align import (
    align_triple,
    free_matching,
    pairwise_alignment_matrix,
)
from graphqa.tasi.ppr import (
    build_nx_graph,
    compute_ppr,
    triple_weight,
    triples_to_entities,
)
from graphqa.tasi.consistency import propagation_consistency
from graphqa.tasi.core import tasi, TasiResult

__all__ = [
    "SentenceEncoder",
    "get_default_encoder",
    "align_triple",
    "free_matching",
    "pairwise_alignment_matrix",
    "build_nx_graph",
    "compute_ppr",
    "triple_weight",
    "triples_to_entities",
    "propagation_consistency",
    "tasi",
    "TasiResult",
]

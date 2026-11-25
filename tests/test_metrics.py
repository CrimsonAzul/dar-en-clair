from darenclair.evaluation.metrics import (
    citation_precision,
    extraire_citations,
    recall_at_k,
    reciprocal_rank,
)
from darenclair.retrieval.hybrid import rrf


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, k=2) == 0.5
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a"], {"x"}) == 0.0


def test_rrf_favorise_accord_entre_listes():
    fusion = rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert fusion[0][0] == "b"
    assert len(rrf([["a", "b"], ["c"]], top_n=2)) == 2


def test_citations():
    rep = "Oui [Loi 67-12, art. 34] et aussi [CGI 2026, art 63]."
    assert extraire_citations(rep) == {("loi 67-12", "34"), ("cgi 2026", "63")}
    assert citation_precision(rep, [("Loi 67-12", "34")]) == 0.5
    assert citation_precision("pas de source", [("Loi 67-12", "34")]) == 0.0

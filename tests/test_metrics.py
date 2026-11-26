from darenclair.evaluation.metrics import (
    articles_uniques,
    bootstrap_difference,
    bootstrap_ic,
    citation_precision,
    citation_rappel,
    citations_inventees,
    moyenne,
    recall_at_k,
    reciprocal_rank,
    sources_citees,
)
from darenclair.retrieval.hybrid import rrf


def test_morceaux_ramenes_a_l_article():
    ids = ["cgi:art_63#p1", "cgi:art_63#p2", "loi:art_5", "cgi:art_64#p1"]
    assert articles_uniques(ids) == ["cgi:art_63", "loi:art_5", "cgi:art_64"]
    # Sans deduplication, art_64 serait 4e ; ici il est 3e donc dans le top 3
    assert recall_at_k(ids, {"cgi:art_64"}, k=3) == 1.0


def test_recall_et_rr():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, k=2) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a"], {"x"}) == 0.0


def test_questions_pieges_exclues():
    assert recall_at_k(["a"], set(), k=5) is None
    assert moyenne([1.0, None, 0.0]) == 0.5


def test_rrf_favorise_accord_entre_listes():
    fusion = rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert fusion[0][0] == "b"
    assert len(rrf([["a", "b"], ["c"]], top_n=2)) == 2


def test_citations_par_numero_de_source():
    contexte = ["loi_67-12:art_34", "cgi_2026:art_63#p2"]
    rep = "Oui [S1]. Pour l'impot voir [S2] et [S7]."
    citees = sources_citees(rep, contexte)
    assert citees == {"loi_67-12:art_34", "cgi_2026:art_63"}
    assert citations_inventees(rep, contexte) == 1
    assert citation_precision(citees, {"loi_67-12:art_34"}) == 0.5
    assert citation_rappel(citees, {"loi_67-12:art_34", "loi_67-12:art_35"}) == 0.5
    assert citation_precision(set(), {"x"}) == 0.0


def test_bootstrap():
    m, bas, haut = bootstrap_ic([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert bas <= m <= haut
    d, bas, haut = bootstrap_difference([0] * 30, [1] * 30)
    assert d == 1 and bas == 1


def test_bootstrap_par_grappes_plus_large():
    # 10 articles, 6 questions chacun, reussies ou ratees ensemble : 10 mesures, pas 60.
    valeurs = [float(i % 2) for i in range(10) for _ in range(6)]
    groupes = [i for i in range(10) for _ in range(6)]
    _, bas, haut = bootstrap_ic(valeurs)
    _, bas_g, haut_g = bootstrap_ic(valeurs, groupes=groupes)
    assert haut_g - bas_g > 1.5 * (haut - bas)
    d, bas, haut = bootstrap_difference([0.0] * 60, [1.0] * 60, groupes=groupes)
    assert d == 1 and bas == 1

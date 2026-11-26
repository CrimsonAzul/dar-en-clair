import pytest

from darenclair.retrieval.rerank import choisir_seuil, doit_refuser


def test_seuil_separe_normales_et_pieges():
    seuil = choisir_seuil([0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3])
    assert seuil == pytest.approx(0.45)  # milieu entre le plus haut piege et la plus basse question normale


def test_seuil_garde_la_plus_grande_marge():
    # 0.5 et 0.85 donnent la meme precision equilibree ; 0.5 est entoure de plus de marge.
    seuil = choisir_seuil([0.9, 0.8, 0.6], [0.4, 0.82])
    assert seuil == pytest.approx(0.5)


def test_refus():
    assert doit_refuser([], 0.5)
    assert doit_refuser([("a", 0.2)], 0.5)
    assert not doit_refuser([("a", 0.7)], 0.5)

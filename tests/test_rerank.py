from darenclair.retrieval.rerank import choisir_seuil, doit_refuser


def test_seuil_separe_normales_et_pieges():
    seuil = choisir_seuil([0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3])
    assert 0.3 < seuil <= 0.6


def test_refus():
    assert doit_refuser([], 0.5)
    assert doit_refuser([("a", 0.2)], 0.5)
    assert not doit_refuser([("a", 0.7)], 0.5)

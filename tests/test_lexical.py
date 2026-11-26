from darenclair.retrieval.lexical import BM25, raciniser, tokeniser


def test_loyer_ne_devient_pas_loi():
    assert raciniser("loyer") != raciniser("loi")
    assert raciniser("loyers") == raciniser("loyer")


def test_accents_et_pluriels():
    assert tokeniser("propriété") == tokeniser("propriete")
    assert tokeniser("Locataires") == tokeniser("locataire")
    assert tokeniser("années") == tokeniser("annee")


def test_mots_composes():
    assert raciniser("plusvalue") in tokeniser("les plus-values immobilières")
    assert tokeniser("plus-value")[0] != tokeniser("moins-value")[0]
    assert "6712" in tokeniser("la loi 67-12")
    bm = BM25({"tpi": "Le profit foncier et la plus-value de cession.", "autre": "La valeur du bien."})
    assert bm.search("plus-value")[0] == "tpi"


def test_mots_vides_retires():
    assert tokeniser("le loyer de la maison") == [raciniser("loyer"), raciniser("maison")]


DOCS = {
    "bail": "Le loyer est revise tous les trois ans, l'augmentation ne peut depasser 8 %.",
    "vente": "La vente d'un immeuble immatricule est inscrite a la conservation fonciere.",
    "copro": "Le syndic execute les decisions de l'assemblee generale des coproprietaires.",
}


def test_question_en_langage_courant():
    # Aucun document ne contient tous les mots : BM25 doit quand meme trouver le bon
    bm = BM25(DOCS)
    assert bm.search("Mon propriétaire peut-il augmenter le loyer chaque année ?")[0] == "bail"
    assert bm.search("qui applique les décisions de l'assemblée ?")[0] == "copro"


def test_filtre_ids():
    bm = BM25(DOCS)
    assert bm.search("loyer vente", ids_autorises={"vente"}) == ["vente"]

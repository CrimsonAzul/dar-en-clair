from darenclair.ingest.parse import split_articles

TEXTE = """Dahir portant promulgation de la loi.
Article premier
La presente loi s'applique aux baux d'habitation.
Article 2 : Le bail est conclu par ecrit.
Article 12-1
Disposition ajoutee.
Article 12 bis.
Autre disposition.
"""


def test_un_chunk_par_article():
    chunks = split_articles(TEXTE, "Loi 67-12", "location", "2013")
    assert [c.article for c in chunks] == ["1", "2", "12-1", "12 bis"]
    assert chunks[0].id == "loi_67-12:art_1"
    assert chunks[1].texte == "Le bail est conclu par ecrit."
    assert all(c.domaine == "location" for c in chunks)


def test_preambule_ignore():
    chunks = split_articles(TEXTE, "Loi 67-12", "location", "2013")
    assert "Dahir" not in chunks[0].texte


def test_numero_repete_suffixe():
    chunks = split_articles("Article 3\nA\nArticle 3\nB\n", "Loi X", "vente", "2020")
    assert [c.id for c in chunks] == ["loi_x:art_3", "loi_x:art_3#2"]

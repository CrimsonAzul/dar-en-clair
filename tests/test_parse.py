from darenclair.ingest.parse import decouper, nettoyer, split_articles

TEXTE = """Dahir portant promulgation de la loi.
TITRE PREMIER
Dispositions generales
Article premier
La presente loi s'applique aux baux d'habitation.
Article 2.- Le bail est conclu par ecrit, sous reserve de
l'article 3 et de
Article 12 de la loi sur les obligations.
CHAPITRE II
Des obligations du bailleur
Article 12-1
Disposition ajoutee.
12
BULLETIN OFFICIEL N 6208
Article 12 bis : Autre disposition.
"""


def chunks():
    return split_articles(TEXTE, "Loi 67-12", "location", "2013")


def test_un_chunk_par_article():
    c = chunks()
    assert [x.article for x in c] == ["1", "2", "12-1", "12 bis"]
    assert c[0].id == "loi_67-12:art_1"
    assert all(x.domaine == "location" for x in c)


def test_renvoi_en_debut_de_ligne_ne_coupe_pas():
    # "Article 12 de la loi..." est la suite de l'article 2, pas un nouvel article
    assert "Article 12 de la loi sur les obligations" in chunks()[1].texte


def test_separateurs_retires():
    c = chunks()
    assert c[1].texte.startswith("Le bail")
    assert c[3].texte == "Autre disposition."


def test_contexte_hierarchique():
    c = chunks()
    assert c[0].contexte == "TITRE PREMIER : Dispositions generales"
    assert c[2].contexte.endswith("CHAPITRE II : Des obligations du bailleur")
    assert c[2].texte_index().startswith("Loi 67-12 > TITRE PREMIER")


def test_preambule_et_parasites_ignores():
    c = chunks()
    assert "Dahir" not in c[0].texte
    assert c[2].texte == "Disposition ajoutee."


def test_cesure():
    assert nettoyer("le loca-\ntaire paie") == "le locataire paie"


def test_numero_repete():
    c = split_articles("Article 3\nA\nArticle 3\nB\n", "Loi X", "vente", "2020")
    assert [x.id for x in c] == ["loi_x:art_3", "loi_x:art_3#dup2"]


def test_article_long_decoupe_avec_meme_article_id():
    long = "Article 5\n" + " ".join(f"Phrase numero {i} du texte." for i in range(200))
    c = split_articles(long, "CGI", "fiscalite", "2026", slug="cgi_2026", max_mots=100)
    assert len(c) > 1
    assert {x.article_id for x in c} == {"cgi_2026:art_5"}
    assert c[0].id == "cgi_2026:art_5#p1"
    assert all(len(x.texte.split()) <= 100 for x in c)


def test_decouper_recouvrement():
    phrases = [f"Phrase {i} ici." for i in range(10)]
    m = decouper(" ".join(phrases), max_mots=9)
    assert m[0].split(". ")[-1] in m[1]  # derniere phrase reprise au debut du morceau suivant

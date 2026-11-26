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


def test_cesure_et_ligatures():
    assert nettoyer("les plus-\nvalues ci-\n dessus") == "les plus-values ci-dessus"
    assert nettoyer("déﬁnies") == "définies"


def test_formats_d_en_tete_reels():
    texte = (
        "Chapitre Premier : Dispositions generales\n"
        "Article Premier\nPremier texte.\n"
        "Article 4 2\nTexte avec appel de note.\n"
        "Article 59 duodecies\nTexte latin.\n"
        "Article 2 - Taxes au profit des communes\nTexte 47-06.\n"
        "Article.127- Taxes recouvrees\nCoquille du texte source.\n"
        "Article17\nPas un en-tete.\n"
    )
    c = split_articles(texte, "Loi 18-00", "copropriete", "2016", saut_renumerotation=0)
    assert [x.article for x in c] == ["1", "4", "59 duodecies", "2", "127"]
    assert c[0].contexte == "Chapitre Premier : Dispositions generales"
    assert c[3].texte.startswith("Taxes au profit")


def test_livre_et_section_premiere():
    texte = "LIVRE PREMIER\nDes droits reels\nTITRE II\nDu loyer\nSection première. - Champ\nArticle 7\nTexte."
    c = split_articles(texte, "Loi 39-08", "vente", "2011")
    assert c[0].contexte == "LIVRE PREMIER : Des droits reels > TITRE II : Du loyer > Section première. - Champ"


def test_appels_de_note_colles():
    assert nettoyer("des immeubles bâtis1 tel") == "des immeubles bâtis tel"
    assert nettoyer("De la copropriété18 : Section") == "De la copropriété : Section"
    assert nettoyer("non immatriculés.3") == "non immatriculés."
    # a ne pas toucher : unites, nombres, references
    assert nettoyer("une surface de 120 m2 au taux de 1.5") == "une surface de 120 m2 au taux de 1.5"
    assert nettoyer("loi n° 18-00 et Mouharam1435") == "loi n° 18-00 et Mouharam1435"


def test_preface_annexes_et_abroges():
    lignes = ["Note de presentation.", "ARTICLE 5", "de la loi de finances 43-06 (preface)."]
    lignes += [f"Article {n}\nTexte {n}." for n in ["premier", *range(2, 41)]]
    lignes += ["Article 41\n(abrogé)", "ANNEXE", "ARTICLE 8", "Article de loi de finances."]
    c = split_articles("\n".join(lignes), "CGI", "fiscalite", "2026", slug="cgi_2026")
    ids = [x.id for x in c]
    assert ids[0] == "cgi_2026:art_1"
    assert "cgi_2026:art_5" in ids and not any("#dup" in i for i in ids)  # preface ignoree
    assert "cgi_2026:art_41" not in ids  # abroge
    assert ids[-1] == "cgi_2026:art_40" and "ANNEXE" not in c[-1].texte  # annexe coupee


def test_numero_de_page_avec_tirets():
    assert nettoyer("fin.\n- 12 -\nsuite") == "fin.\n\nsuite"


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

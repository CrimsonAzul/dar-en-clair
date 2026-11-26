from darenclair.ingest.ocr import corriger_intitule, lignes_depuis_tsv


def test_corrections_d_intitules():
    assert corriger_intitule("Articie 8") == "Article 8"
    assert corriger_intitule("ARTlCLE 12") == "ARTICLE 12"
    assert corriger_intitule(".Article 71") == "Article 71"
    assert corriger_intitule("! Article 132") == "Article 132"
    assert corriger_intitule("Section HI. \u2014 Le stage") == "Section III. \u2014 Le stage"
    assert corriger_intitule("Chapitre Il") == "Chapitre II"


def test_texte_ordinaire_intact():
    for t in ["Annexe 3 du present texte", "Partie 2", "Le notaire 12 fois", "Section 2. - Champ"]:
        assert corriger_intitule(t) == t


ENTETE = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def mot(bloc, ligne, x, y, texte, conf=95, h=40):
    return f"5\t1\t{bloc}\t1\t{ligne}\t1\t{x}\t{y}\t100\t{h}\t{conf}\t{texte}\n"


def test_tsv_ordre_des_colonnes_et_bruit():
    # colonne droite (bloc 2) plus haute sur la page que la fin de la colonne gauche (bloc 1)
    tsv = ENTETE + mot(1, 1, 100, 500, "Article") + mot(1, 1, 300, 500, "6") \
        + mot(2, 1, 1500, 200, "Article") + mot(2, 1, 1700, 200, "10") \
        + mot(1, 2, 100, 560, "‘Ç", conf=5)
    lignes = lignes_depuis_tsv(tsv, hauteur_px=3000)
    assert [l.texte for l in lignes] == ["Article 6", "Article 10"]
    assert abs(lignes[0].y0 - 500 / 3000) < 1e-9

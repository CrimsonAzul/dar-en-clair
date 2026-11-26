from darenclair.ingest.pdf import Ligne, assembler, est_sommaire, retirer_entetes, retirer_notes


def page(n: int, corps: list[str], notes: list[str] = ()) -> list[Ligne]:
    lignes = [Ligne("Agence Nationale de la Conservation Fonciere", 0.03, 0.05, 11)]
    lignes.append(Ligne(str(n), 0.06, 0.08, 11))
    y = 0.2
    for t in corps:
        lignes.append(Ligne(t, y, y + 0.02, 11))
        y += 0.03
    y = 0.8
    for t in notes:
        lignes.append(Ligne(t, y, y + 0.02, 9))
        y += 0.02
    return lignes


def test_entetes_retires_mais_pas_les_articles_en_haut_de_page():
    pages = [page(i, [f"Article {i}", f"Texte {i}."]) for i in range(1, 6)]
    pages[2].insert(0, Ligne("Article 99", 0.04, 0.06, 11))  # article tout en haut d'une page
    textes = [[l.texte for l in p] for p in retirer_entetes(pages)]
    assert all("Agence Nationale de la Conservation Fonciere" not in p for p in textes)
    assert "Article 99" in textes[2]
    assert all(f"Article {i}" in textes[i - 1] for i in range(1, 6))


def test_notes_de_bas_de_page_retirees():
    p = page(1, ["Article 1", "Le texte de l'article."], notes=["1 - Publie au BO n 5054.", "2 - Modifie par la loi 106-12."])
    p.append(Ligne("9", 0.9, 0.92, 11))  # numero de page sous les notes
    textes = [l.texte for l in retirer_notes(p, corps=11)]
    assert "Le texte de l'article." in textes
    assert not any("Publie au BO" in t or "Modifie par" in t for t in textes)


def test_petite_police_en_haut_de_page_conservee():
    p = [Ligne("Tableau en petite police", 0.3, 0.32, 9), Ligne("Texte normal", 0.5, 0.52, 11)]
    assert len(retirer_notes(p, corps=11)) == 2


def test_sommaires_detectes():
    toc = [Ligne(t, 0.1 + i * 0.02, 0.11 + i * 0.02, 11) for i, t in enumerate(
        ["TABLE DES MATIERES", "TITRE PREMIER : DES DROITS", "9", "Chapitre premier", "9"])]
    assert est_sommaire(toc)
    pointilles = [Ligne(f"Article {i}.- Intitule -------------------- {i}", 0.1, 0.12, 11) for i in range(8)]
    assert est_sommaire(pointilles)
    corps = [Ligne(t, 0.2, 0.22, 11) for t in ["Article 5", "Le bail est conclu par ecrit.", "12"]]
    assert not est_sommaire(corps)


def test_assembler_bout_a_bout():
    pages = [page(i, [f"Article {i}", f"Texte {i}."], notes=["1 - Note."]) for i in range(1, 5)]
    texte = assembler(pages)
    assert "Agence" not in texte and "Note." not in texte
    assert "Article 3\nTexte 3." in texte

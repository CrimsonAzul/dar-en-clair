from darenclair.finetune.donnees import cote
from darenclair.finetune.questions import filtrer, langue_de


def test_filtre_questions():
    gardees = filtrer([
        "Mon proprio peut-il augmenter le loyer chaque annee ?",
        "Que dit l'article 34 sur le loyer ?",  # cite l'article
        "Selon ce texte, qui paie la taxe ?",  # parle de l'extrait
        "Que prevoit la loi n° 67-12 ?",  # cite la loi
        "ماذا تقول هذه المادة عن الرسم؟",  # هذه المادة
        "Quel est le maximum d'augmentation du loyer selon cette loi ?",  # parle de la source
        "Comment识别并翻译 Artículo 385 del tribunal ?",  # derapage chinois / espagnol
        "Loyer ?",  # trop court
        "mon proprio peut-il augmenter le loyer chaque annee ?",  # doublon
    ])
    assert gardees == ["Mon proprio peut-il augmenter le loyer chaque annee ?"]


def test_langue_et_separation_stable():
    assert langue_de("من يؤدي رسم السكن؟") == "ar"
    assert langue_de("Qui paie la taxe d'habitation ?") == "fr"
    assert cote("loi_67-12:art_32") == cote("loi_67-12:art_32")
    parts = [cote(f"cgi_2026:art_{i}") for i in range(1, 400)]
    assert 0.04 < parts.count("val") / len(parts) < 0.18  # environ 10 % en validation

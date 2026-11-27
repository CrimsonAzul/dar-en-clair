"""Pipeline et generation sans GPU ni base : recherche dense, reranker et LLM factices."""

from darenclair.generation.llm import Sortie
from darenclair.generation.outils import executer
from darenclair.generation.prompt import HORS_CORPUS, langue_question, message
from darenclair.generation.repondre import repondre
from darenclair.ingest.parse import Chunk
from darenclair.retrieval.pipeline import Recherche


def chunk(id_, texte, domaine="location"):
    loi, art = id_.split(":")
    return Chunk(id=id_, article_id=id_.split("#")[0], loi=loi, article=art.removeprefix("art_"),
                 domaine=domaine, edition="", contexte="", texte=texte)


CHUNKS = [
    chunk("loi_67-12:art_20", "La caution ne peut exceder deux mois de loyer."),
    chunk("loi_67-12:art_34", "Augmentation du loyer : 8 % pour les locaux a usage d'habitation."),
    chunk("cgi_2026:art_63", "Exoneration du profit sur l'habitation principale.", "fiscalite"),
]


def faux_dense(question, k, domaine):
    ids = ["loi_67-12:art_34", "cgi_2026:art_63", "loi_67-12:art_20"]
    return [i for i in ids if not domaine or i.startswith("loi")][:k]


def faux_reranker(question, passages, top_n):
    # Score eleve seulement si la question et le passage parlent tous deux de caution
    pertinent = "caution" in question
    classes = sorted(((i, 0.9 if pertinent and "caution" in t else 0.1) for i, t in passages),
                     key=lambda x: -x[1])
    return classes[:top_n] if top_n else classes


class FauxLLM:
    def __init__(self, texte):
        self.texte, self.vu = texte, None

    def repondre(self, systeme, msg, outils, executer):
        self.vu = msg
        return Sortie(self.texte)


def recherche(seuil=0.5):
    return Recherche(CHUNKS, faux_dense, seuil_refus=seuil, rerank=faux_reranker)


def test_hybride_fusionne_dense_et_bm25():
    r = recherche()
    assert r.lexicale("caution deux mois")[0] == "loi_67-12:art_20"
    assert set(r.hybride("caution deux mois")) == {c.id for c in CHUNKS}


def test_filtre_par_domaine():
    r = recherche()
    assert all(i.startswith("loi") for i in r.hybride("loyer exoneration", domaine="location"))


def test_reranker_puis_refus():
    r = recherche()
    res = r.chercher("Le proprietaire demande trois mois de caution")
    assert res.classes[0][0] == "loi_67-12:art_20" and not res.refus
    assert r.chercher("Quel temps fera-t-il demain ?").refus


def test_reponse_avec_citations_et_passages_numerotes():
    llm = FauxLLM("Non, deux mois au plus [S1]. Voir aussi [S9].")
    rep = repondre("Le proprietaire demande trois mois de caution", recherche(), llm, n=2)
    assert "[S1] Loi" not in llm.vu and "[S1] loi_67-12, art. 20" in llm.vu
    assert not rep.refus and rep.citations_inventees == 1
    assert rep.passages[0]["cite"] and not rep.passages[1]["cite"]


def test_refus_par_seuil_et_par_llm():
    assert repondre("Quel temps fera-t-il demain ?", recherche(), FauxLLM("x")).motif_refus == "seuil"
    rep = repondre("Le proprietaire demande trois mois de caution", recherche(), FauxLLM(HORS_CORPUS))
    assert rep.refus and rep.motif_refus == "llm"
    # Le modele ecrit parfois une reponse puis conclut HORS_CORPUS : c'est un refus.
    rep = repondre("Le proprietaire demande trois mois de caution", recherche(),
                   FauxLLM(f"Peut-etre deux mois [S1]. {HORS_CORPUS}"))
    assert rep.refus


class FauxLLMCalcul(FauxLLM):
    def __init__(self, texte, extraction):
        super().__init__(texte)
        self.extraction, self.extraits, self.outils_vus = extraction, 0, None

    def extraire(self, systeme, texte, schema):
        self.extraits += 1
        return self.extraction

    def repondre(self, systeme, msg, outils, executer):
        self.outils_vus = outils
        return super().repondre(systeme, msg, outils, executer)


def test_calcul_decide_par_le_code_et_montant_ajoute_si_oublie():
    from darenclair.generation.repondre import generer

    llm = FauxLLMCalcul("Le taux est de 10 % [S1].", {"calcul": "impot_revenus_fonciers", "loyer_mensuel": 4000})
    rep = generer("Je loue 4 000 DH par mois, combien d'impôt ?", CHUNKS[:1], [0.9], llm)
    appel = rep.appels_outils[0]
    assert appel["par"] == "code" and appel["arguments"] == {"loyers_bruts_annuels": 48_000}
    assert appel["resultat"]["resultat"]["montant"] == 4_800
    assert "a reprendre tel quel" in llm.vu and llm.outils_vus == []  # le modele ne choisit plus l'outil
    assert "4 800 DH" in rep.texte  # absent de la reponse du modele : ajoute par le code


def test_pas_de_calcul_sans_chiffre_ni_avec_montant_invente():
    from darenclair.generation.repondre import generer

    sans_chiffre = FauxLLMCalcul("x [S1]", {"calcul": "impot_revenus_fonciers", "loyer_mensuel": 4000})
    generer("Combien d'impôt sur mes loyers ?", CHUNKS[:1], [0.9], sans_chiffre)
    assert sans_chiffre.extraits == 0
    invente = FauxLLMCalcul("x [S1]", {"calcul": "impot_revenus_fonciers", "loyer_mensuel": 5000})
    assert generer("Je loue 4 000 DH par mois", CHUNKS[:1], [0.9], invente).appels_outils == []
    terrain = FauxLLMCalcul("x [S1]", {"calcul": "droits_enregistrement", "prix_achat": 500000, "nature_bien": "inconnu"})
    assert generer("J'achète un bien à 500 000 DH", CHUNKS[:1], [0.9], terrain).appels_outils == []


def test_droits_enregistrement_prix_dans_l_un_ou_l_autre_champ():
    from darenclair.generation.calcul import appel_depuis_extraction

    q = "J'achète un appartement de 1 000 000 DH"
    e = {"calcul": "droits_enregistrement", "prix_achat": None, "prix_vente": 1_000_000, "nature_bien": "local_construit"}
    assert appel_depuis_extraction(e, q) == ("droits_enregistrement", {"prix": 1_000_000.0, "nature": "local_construit"})


def test_calcul_vente_avec_forfait():
    from darenclair.generation.calcul import appel_depuis_extraction

    q = "Je vends 1 000 000 DH un terrain payé 900 000 DH"
    e = {"calcul": "impot_profit_immobilier", "prix_vente": 1_000_000, "prix_achat": 900_000}
    assert appel_depuis_extraction(e, q) == (
        "impot_profit_immobilier", {"prix_cession": 1_000_000.0, "prix_acquisition": 900_000.0})


def test_langue_question():
    assert langue_question("هل يمكنني البقاء في الشقة؟") == "ar"
    assert langue_question("Can my landlord raise the rent?") == "en"
    assert langue_question("Mon proprietaire peut-il augmenter le loyer ?") == "fr"


def test_message_numerote_les_passages():
    texte = message("Q ?", CHUNKS[:2])
    assert texte.index("[S1]") < texte.index("[S2]") and texte.endswith("Question : Q ?")


def test_bloc_de_reflexion_retire():
    from darenclair.generation.llm import sans_pensee

    assert sans_pensee("<think>\nje calcule...\n</think>\n\nDeux mois [S1].") == "Deux mois [S1]."
    assert sans_pensee(None) == ""


def test_montant_present_quel_que_soit_le_format():
    from darenclair.evaluation.generation import montant_present

    assert montant_present("soit 36 000 DH [S1]", 36_000)
    assert montant_present("soit 36.000 DH", 36_000) and montant_present("36,000 dirhams", 36_000)
    assert not montant_present("soit 3 600 DH", 36_000)


def test_calcul_correct_exige_l_outil():
    from darenclair.evaluation.generation import calcul_correct

    appel = {"outil": "droits_enregistrement", "arguments": {}, "resultat": {"resultat": {"montant": 36_000}}}
    bon = {"calcul_attendu": 36_000, "reponse": {"texte": "36 000 DH [S1]", "appels_outils": [appel]}}
    sans_outil = {"calcul_attendu": 36_000, "reponse": {"texte": "36 000 DH [S1]", "appels_outils": []}}
    assert calcul_correct(bon) and not calcul_correct(sans_outil)


def test_outils_fiscaux():
    assert executer("droits_enregistrement", {"prix": 900_000, "nature": "local_construit"})["resultat"]["montant"] == 36_000
    assert "erreur" in executer("droits_enregistrement", {"prix": 1, "nature": "villa"})
    assert "erreur" in executer("outil_invente", {})

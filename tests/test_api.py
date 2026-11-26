"""API avec un moteur factice : pas de base, de GPU ni de LLM."""

from fastapi.testclient import TestClient

from darenclair.api.main import app, moteur
from darenclair.generation.repondre import Reponse


class FauxMoteur:
    def repondre(self, question, domaine):
        if "meteo" in question:
            return Reponse("Je ne trouve pas de reponse.", True, "seuil")
        passage = {"numero": 1, "id": "loi_67-12:art_20", "article_id": "loi_67-12:art_20",
                   "reference": "Loi 67-12, art. 20", "contexte": "", "langue": "fr",
                   "texte": "La caution ne peut exceder deux mois de loyer.", "score": 0.93, "cite": True}
        return Reponse("Deux mois au plus [S1].", False, passages=[passage])


app.dependency_overrides[moteur] = FauxMoteur
client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_reponse_citee():
    r = client.post("/ask", json={"question": "Combien de caution ?", "domaine": "location"}).json()
    assert not r["refus"] and r["passages"][0]["reference"] == "Loi 67-12, art. 20"
    assert "pas un conseil" in r["avertissement"]


def test_ask_refus_et_validation():
    assert client.post("/ask", json={"question": "la meteo demain ?"}).json()["refus"]
    assert client.post("/ask", json={"question": "ok", "domaine": "voitures"}).status_code == 422

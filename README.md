# Dar en Clair

Assistant de questions-reponses sur l'immobilier au Maroc : location, vente, fiscalite immobiliere et copropriete. Il repond en citant les articles de loi utilises, refuse quand le corpus ne couvre pas la question, et confie les calculs d'impots a du code Python teste plutot qu'au modele.

> Reponses informatives, pas un conseil juridique ou fiscal.

## Corpus

| Domaine | Textes |
|---|---|
| Location | Loi 67-12 (baux d'habitation et professionnels), Loi 49-16 (baux commerciaux) |
| Vente | Loi 39-08 (Code des droits reels), Loi 32-09 (notariat) |
| Fiscalite | Code General des Impots 2026 (parties immobilieres), Loi 47-06 (fiscalite locale) |
| Copropriete | Loi 18-00 |

Sources et versions : [data/raw/SOURCES.md](data/raw/SOURCES.md).

## Architecture

```
PDF -> nettoyage -> 1 chunk par article (+ morceaux de 350 mots si long)
                    texte indexe = "Loi > Titre > Chapitre > Article N" + contenu

question ─┬─> bge-m3 dense (pgvector, recherche exacte), top 30 ─┐
          └─> BM25 francais (en memoire), top 30 ─────────────────┴─> RRF ─> top 30
                                                                           │
                                       reranker bge-reranker-v2-m3 <───────┘
                                                  │
                         score max < seuil ? ── oui ──> refus explicite
                                                  │ non
                                     top 5 passages numerotes [S1]..[S5]
                                                  │
                     outils fiscaux (Python) <── LLM ──> reponse + citations [S2]
                                                         -> "Loi 67-12, art. 34"
```

Choix et raisons :
- **Contexte dans le chunk** : un article seul ("le preneur doit...") est ambigu ; le chemin Titre > Chapitre aide la recherche dense et BM25.
- **BM25 en Python** plutot que le plein texte Postgres : vrai IDF, questions en langage courant (pas besoin que tous les mots soient presents), insensible aux accents. Raciniseur leger maison car Snowball transforme "loyer" en "loi".
- **RRF** : fusion par rangs, pas de calibration de scores entre BM25 et cosinus.
- **Refus par seuil sur le reranker**, seuil choisi sur le jeu dev : un LLM repond presque toujours quelque chose.
- **Citations par numero de passage** ([S2]) puis conversion en article : pas de reference inventee ou mal recopiee.
- **Calculs dans du code** : taux dans `config/fiscal_2026.yaml`, refuses tant qu'ils ne sont pas verifies.

## Lancer

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[ml,dev]"
cp .env.example .env            # ajouter GEMINI_API_KEY
docker compose up -d db
python -m darenclair.ingest.parse data/raw/loi_67-12.pdf --loi "Loi 67-12" --slug loi_67-12 --domaine location --edition 2013 --out data/processed/loi_67-12.jsonl
pytest
```

## Evaluation

Methode dans [eval/README.md](eval/README.md) : 100 questions annotees au niveau article, separees en dev (reglages) et test (chiffres publies), intervalles de confiance par bootstrap.

| Configuration (jeu test) | recall@5 | MRR |
|---|---|---|
| BM25 | | |
| Dense (bge-m3) | | |
| Hybride (RRF) | | |
| Hybride + reranker | | |

## Avancement

- [x] Decoupage, BM25, RRF, metriques, seuil de refus, calculateur fiscal (taux a remplir)
- [ ] Textes telecharges, decoupes et verifies a la main
- [ ] 100 questions annotees
- [ ] Embeddings + pgvector, puis hybride, puis reranker
- [ ] Generation avec citations et outils
- [ ] API, interface Next.js

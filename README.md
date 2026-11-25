# Dar en Clair

Assistant de questions-reponses sur l'immobilier au Maroc : location, vente, fiscalite immobiliere et copropriete. Il repond en citant l'article de loi utilise, et confie les calculs d'impots a du code Python teste plutot qu'au modele.

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
question ─┬─> bge-m3 (dense, pgvector) ──┐
          └─> plein texte Postgres ──────┴─> RRF ─> reranker bge-v2-m3 ─> top 5
                                                                          │
                          outils fiscaux (Python) <─── LLM (Gemini) <─────┘
                                                        │
                                       reponse + citations [Loi, art.]
```

## Lancer

```bash
cp .env.example .env            # ajouter GEMINI_API_KEY
docker compose up -d db
pip install -e ".[ml,dev]"
python -m darenclair.ingest.parse data/raw/loi_67-12.pdf --loi "Loi 67-12" --domaine location --edition 2013 --out data/processed/loi_67-12.jsonl
pytest
```

## Evaluation

Voir [eval/README.md](eval/README.md). Resultats a venir :

| Configuration | recall@5 | MRR |
|---|---|---|
| Dense (bge-m3) | | |
| Plein texte | | |
| Hybride (RRF) | | |
| Hybride + reranker | | |

## Avancement

- [x] Squelette, schema SQL, metriques, RRF, calculateur fiscal (taux a remplir)
- [ ] Textes telecharges et decoupes par article
- [ ] 100 questions annotees
- [ ] Recherche dense, puis hybride, puis reranker
- [ ] Generation avec citations et outils
- [ ] API, interface Next.js

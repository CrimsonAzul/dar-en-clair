# Dar en Clair

Assistant de questions-reponses sur l'immobilier au Maroc : location, vente, fiscalite immobiliere et copropriete. Il repond en citant les articles de loi utilises, refuse quand le corpus ne couvre pas la question, et confie les calculs d'impots a du code Python teste plutot qu'au modele.

> Reponses informatives, pas un conseil juridique ou fiscal.

## Corpus

| Domaine | Textes |
|---|---|
| Location | Loi 67-12 (baux d'habitation et professionnels), Loi 49-16 (baux commerciaux) |
| Vente | Loi 39-08 (Code des droits reels), Loi 32-09 (notariat) |
| Fiscalite | Code General des Impots 2026, Loi 47-06 (fiscalite locale, version consolidee 2021 en arabe) |
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
- **Corpus multilingue sans traduction** : chaque texte est indexe dans sa meilleure version (la loi 47-06 consolidee n'existe qu'en arabe). bge-m3 et le reranker sont multilingues : une question en francais, arabe ou anglais trouve l'article quelle que soit sa langue. Seuls les 3 a 5 articles cites sont traduits, au moment de la reponse, et marques comme traduction automatique.
- **PDF difficiles** : notes de bas de page reperees par la taille de police, en-tetes par leur position, sommaires et annexes ecartes, OCR Tesseract pour les scans, ligatures arabes lam-alef reparees grace a la largeur nulle de l'alef dans le glyphe.

## Lancer

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[ml,dev]"
cp .env.example .env            # ajouter GEMINI_API_KEY
docker compose up -d db
python -m darenclair.ingest.corpus   # decoupe tous les PDF de data/raw selon config/corpus.yaml
                                     # PDF scannes : OCR Tesseract (winget install UB-Mannheim.TesseractOCR)
                                     # + modele fra de tessdata_best dans .tessdata/
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

## Fine-tuning LoRA de bge-m3 (GPU de 6 Go)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[train]"
ollama pull qwen2.5:7b
python -m darenclair.finetune.questions   # 3 questions en langage courant par chunk (LLM local)
python -m darenclair.finetune.donnees     # separation par article (90/10) + negatifs difficiles BM25
python -m darenclair.finetune.entrainer   # LoRA r=16, GradCache (lot effectif 64), bf16, gradient checkpointing
python -m darenclair.finetune.evaluer     # base vs LoRA, niveau article, IC bootstrap par article
```

Les questions de validation viennent d'articles jamais vus a l'entrainement (ni comme positifs, ni comme negatifs). Le chiffre a publier reste celui du jeu de test ecrit a la main : `python -m darenclair.finetune.evaluer --questions eval/questions.jsonl --split test`.

Premier essai : 6 448 questions generees par qwen2.5:7b (3 309 en francais, 3 139 en arabe), 84 % des questions d'entrainement gardees par le filtre de coherence, 4 935 triplets, 2 epoques en 56 min, pic de VRAM 2,8 Go. Un seul entrainement, aucun reglage fait sur la validation.

Validation : 552 questions sur 117 articles jamais vus. Intervalles a 95 % par bootstrap sur les articles (les questions d'un meme article ne sont pas independantes).

| | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| bge-m3 de base | 0.418 | 0.705 | 0.795 | 0.548 |
| + LoRA | 0.562 | 0.824 | 0.897 | 0.675 |
| Ecart | +0.143 [+0.093, +0.190] | +0.120 [+0.084, +0.156] | +0.101 [+0.065, +0.140] | +0.128 [+0.092, +0.162] |

Le gain le plus net est entre langues : question en francais sur un article arabe, recall@5 de 0.463 a 0.683 ; question en arabe sur un article francais, de 0.654 a 0.777. Arabe vers arabe : pas d'ecart significatif (53 questions sur 24 articles, trop peu pour conclure).

Limite : validation et entrainement viennent du meme generateur. Le modele a pu apprendre le style des questions de qwen2.5 en plus du vocabulaire juridique, donc le gain sur de vraies questions sera sans doute plus faible. C'est le jeu de test ecrit a la main qui tranchera.

## Avancement

- [x] Decoupage, BM25, RRF, metriques, seuil de refus, calculateur fiscal (taux a remplir)
- [ ] Textes telecharges, decoupes et verifies a la main
- [ ] 100 questions annotees
- [ ] Embeddings + pgvector, puis hybride, puis reranker
- [ ] Generation avec citations et outils
- [ ] API, interface Next.js

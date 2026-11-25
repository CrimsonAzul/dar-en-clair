# Jeu d'evaluation

A ecrire **avant** de coder la recherche. Objectif : 100 questions dans `questions.jsonl`, au format de `questions.example.jsonl`.

| Champ | Valeurs |
|---|---|
| `type` | `mono` (60), `multi` (25, au moins deux textes), `piege` (15, la bonne reponse est un refus) |
| `split` | `dev` (40) pour regler les parametres, `test` (60) pour les chiffres publies |
| `langue` | `fr` (environ 80), `ar` ou `en` (environ 20) |
| `domaine` | `location`, `vente`, `fiscalite`, `copropriete`, `multi`, `hors_sujet` |
| `sources` | ids d'**articles** pertinents, ex. `loi_67-12:art_34` (jamais `#p2`), vide pour les pieges |

## Regles

- **Langage courant.** Une question qui reprend les mots de l'article avantage BM25 et gonfle les scores. Ecrire comme un locataire ou un vendeur ("mon proprio veut me mettre dehors"), et faire ecrire 20 a 30 questions par des proches **sans** leur montrer les textes, puis trouver soi-meme les articles.
- **Dev et test separes.** Tous les reglages (k, taille des morceaux, seuil de refus, prompt) se font sur `dev`. On ne regarde `test` qu'a la fin, une fois par configuration. Repartir les types et les domaines de facon equilibree entre les deux.
- **Pieges credibles.** Pas seulement hors sujet (TVA sur les voitures), aussi des questions proches mais non couvertes (bail d'un terrain agricole, succession).
- **Calculs.** Pour une question fiscale chiffree, noter le resultat attendu, calcule a la main depuis le texte.
- **Petit echantillon.** Avec 60 questions de test, comparer les configurations avec `bootstrap_difference` : si l'intervalle contient 0, ne pas annoncer de gain.
- **Juge LLM.** Pour la fidelite des reponses, verifier le juge sur 20 reponses notees a la main avant de lui faire confiance.

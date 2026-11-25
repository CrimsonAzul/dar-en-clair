# Jeu d'evaluation

A ecrire **avant** de coder la recherche. Objectif : 100 questions dans `questions.jsonl`, au format de `questions.example.jsonl`.

| Champ | Valeurs |
|---|---|
| `type` | `mono` (60), `multi` (25, au moins deux textes), `piege` (15, la bonne reponse est un refus) |
| `langue` | `fr` (environ 80), `ar` ou `en` (environ 20) |
| `domaine` | `location`, `vente`, `fiscalite`, `copropriete`, `multi`, `hors_sujet` |
| `sources` | ids des chunks pertinents, ex. `loi_67-12:art_34` (vide pour les pieges) |

Regles :
- Formuler les questions comme un vrai locataire ou vendeur, pas en recopiant l'article.
- Repartir les questions a peu pres egalement entre les 4 domaines.
- Pour les questions fiscales avec un montant, noter aussi le resultat de calcul attendu.

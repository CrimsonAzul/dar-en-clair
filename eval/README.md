# Jeu d'evaluation

A ecrire **avant** de coder la recherche. Objectif : 100 questions dans `questions.jsonl`, au format de `questions.example.jsonl`.

Etat actuel : les 100 questions de `questions.jsonl` sont un premier jet ecrit avec un assistant IA (Claude), pas par de vrais utilisateurs. Methode : les questions ont ete ecrites d'abord, a partir de situations courantes et sans lire les articles, puis annotees en lisant les textes du corpus (jamais a partir des resultats du moteur de recherche, pour ne pas favoriser ce qu'il trouve deja). Le split dev/test est stratifie par type et par langue. Mieux encore : remplacer 20 a 30 questions par des questions de proches, comme indique plus bas.

Relecture du 26 novembre 2026, avant tout calcul sur ces questions : chaque reponse attendue a ete comparee au texte des articles cites (assistant IA, a la demande de l'auteur), et les taux du CGI relus dans le PDF officiel. Deux corrections :
- q044 (dev) : l'explication ignorait le forfait de 15 % de frais d'acquisition (CGI art. 65-II). Le montant attendu, 30 000 DH, ne change pas ; l'article 65 est ajoute aux sources.
- q086 (test, piege) : "comment se partage une maison heritee" touchait la procedure de partage de la loi 39-08 (art. 25 a 27, 313 a 332), donc ce n'etait pas un vrai piege. Reformulee pour porter sur les parts successorales (Code de la famille, hors corpus).

Les ids d'articles sont verifies automatiquement (tous existent dans le corpus).

Commandes, dans cet ordre (le seuil de refus est choisi sur dev, puis relu tel quel sur test) :

```bash
python -m darenclair.finetune.evaluer --questions eval/questions.jsonl --split test   # dense : base contre LoRA
python -m darenclair.evaluation.recherche --split dev     # 6 configurations + choix du seuil de refus
python -m darenclair.evaluation.recherche --split test    # memes configurations, seuil relu
python -m darenclair.evaluation.generation --split dev    # refus, citations, calculs (LLM)
python -m darenclair.evaluation.generation --split test
```

Les sorties detaillees (classements, reponses completes) vont dans `eval/results/`, non versionne.

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

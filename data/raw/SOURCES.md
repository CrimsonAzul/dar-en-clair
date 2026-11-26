# Textes du corpus

Les PDF ne sont pas versionnes (voir `.gitignore`). Metadonnees de decoupage : `config/corpus.yaml`.
Etat au 25/11/2026, apres `python -m darenclair.ingest.corpus`.

| Fichier | Texte | Domaine | Source | Version | Etat |
|---|---|---|---|---|---|
| `loi_67-12.pdf` | Loi 67-12, baux d'habitation et a usage professionnel | location | aute.gov.ma | Traduction non officielle, verifiee le 02/12/2018 | OK, 75 articles |
| `loi_49-16.pdf` | Loi 49-16, baux commerciaux, industriels et artisanaux | location | adala.justice.gov.ma (Ministere de la Justice) | Dahir du 18/07/2016 | OK, 38 articles |
| `loi_39-08.pdf` | Loi 39-08, Code des droits reels | vente | ANCFCC | Version modifiee et completee | OK, 334 articles |
| `loi_32-09.pdf` | Loi 32-09, profession de notaire | vente | fourni (BO n° 6062 du 05/07/2012, scan) | 2011 | OK par OCR, 134 articles ; petites erreurs possibles ("I est" pour "Il est") |
| `cgi_2026.pdf` | Code General des Impots | fiscalite | DGI | Edition 2026 (LF 50-25) | OK, 353 articles (code entier, annexes des lois de finances ecartees) |
| `loi_47-06ar.pdf` | Loi 47-06, fiscalite des collectivites territoriales | fiscalite | Direction des finances des collectivites territoriales | **Version consolidee 2021** (loi 07-20 du 31/12/2020), **en arabe** | OK, 182 articles ; annexe (liste des professions) ecartee |
| `loi_47-06.pdf` | Loi 47-06 en francais | | DGI | Version d'origine de 2007, perimee | Garde pour reference, **non indexe** |
| `loi_18-00.pdf` | Loi 18-00, statut de la copropriete des immeubles batis | copropriete | fourni | Modifiee par la loi 106-12 (2016) | OK, 101 articles |

`loi_49-16_scan.pdf` : premiere version fournie, scannee, remplacee par la version texte de adala. Non utilisee.

Limites a citer dans le README : la loi 47-06 est indexee en arabe (seule version a jour), les questions en francais la trouvent par la recherche dense multilingue (pas par BM25) ; la loi 67-12 est une traduction non officielle (le texte officiel est en arabe) ; les numeros absents du CGI (32, 97, 98, 100, 165, 178, 197, 202, 276) sont des articles abroges ou absents de l'edition.

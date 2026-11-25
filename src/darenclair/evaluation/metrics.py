"""Metriques de recherche et de generation, au niveau ARTICLE.

Un article long est coupe en plusieurs chunks (id "...#p2") : on ramene chaque resultat a son
article et on deduplique avant de mesurer, sinon deux morceaux du meme article occupent deux
places du top k et le recall est fausse.

Les questions sans source (pieges) renvoient None et sont exclues des moyennes de recherche :
elles se mesurent a part, par le taux de refus.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence


def article_de(chunk_id: str) -> str:
    return re.sub(r"#p\d+$", "", chunk_id)


def articles_uniques(chunk_ids: Iterable[str]) -> list[str]:
    vus: dict[str, None] = {}
    for c in chunk_ids:
        vus.setdefault(article_de(c), None)
    return list(vus)


def recall_at_k(recuperes: Sequence[str], pertinents: Iterable[str], k: int) -> float | None:
    """Part des articles pertinents presents dans les k premiers articles recuperes."""
    pertinents = set(pertinents)
    if not pertinents:
        return None
    top = set(articles_uniques(recuperes)[:k])
    return len(pertinents & top) / len(pertinents)


def reciprocal_rank(recuperes: Sequence[str], pertinents: Iterable[str]) -> float | None:
    pertinents = set(pertinents)
    if not pertinents:
        return None
    for rang, art in enumerate(articles_uniques(recuperes), start=1):
        if art in pertinents:
            return 1.0 / rang
    return 0.0


def moyenne(valeurs: Iterable[float | None]) -> float:
    v = [x for x in valeurs if x is not None]
    return sum(v) / len(v) if v else 0.0


def bootstrap_ic(valeurs: Sequence[float | None], n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Moyenne et intervalle de confiance a 95 % par bootstrap. Avec 60 questions de test,
    un ecart de 2 ou 3 points entre deux configurations n'est souvent pas significatif."""
    v = [x for x in valeurs if x is not None]
    if not v:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    moyennes = sorted(sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n))
    return sum(v) / len(v), moyennes[int(n * alpha / 2)], moyennes[int(n * (1 - alpha / 2)) - 1]


def bootstrap_difference(a: Sequence[float | None], b: Sequence[float | None], n: int = 2000, seed: int = 0):
    """Difference moyenne b - a sur les MEMES questions (bootstrap apparie) et son IC a 95 %.
    Si l'intervalle contient 0, on ne peut pas dire que b est meilleur que a."""
    paires = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not paires:
        return 0.0, 0.0, 0.0
    diffs = [y - x for x, y in paires]
    rng = random.Random(seed)
    tirages = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), tirages[int(n * 0.025)], tirages[int(n * 0.975) - 1]


# Le LLM cite les passages par leur numero dans le contexte : [S1], [S2]...
# Plus fiable que de lui faire recopier "Loi 67-12, art. 34" (formats variables, inventions).
SOURCE_RE = re.compile(r"\[S(\d+)\]")


def sources_citees(reponse: str, contexte: Sequence[str]) -> set[str]:
    """Convertit les [S2] de la reponse en ids d'articles, d'apres l'ordre des passages fournis.
    Un numero hors limites (source inventee) est ignore ici et compte comme erreur plus bas."""
    res = set()
    for n in SOURCE_RE.findall(reponse):
        i = int(n) - 1
        if 0 <= i < len(contexte):
            res.add(article_de(contexte[i]))
    return res


def citations_inventees(reponse: str, contexte: Sequence[str]) -> int:
    return sum(1 for n in SOURCE_RE.findall(reponse) if not 1 <= int(n) <= len(contexte))


def citation_precision(citees: set[str], attendues: Iterable[str]) -> float:
    """Part des articles cites qui sont pertinents. 0 si la reponse ne cite rien."""
    return len(citees & set(attendues)) / len(citees) if citees else 0.0


def citation_rappel(citees: set[str], attendues: Iterable[str]) -> float | None:
    attendues = set(attendues)
    return len(citees & attendues) / len(attendues) if attendues else None

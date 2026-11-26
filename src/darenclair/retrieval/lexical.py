"""BM25 en memoire avec un tokeniseur adapte au francais juridique.

Pourquoi pas le plein texte de Postgres : ts_rank n'a pas d'IDF (ce n'est pas du BM25),
websearch_to_tsquery exige TOUS les mots de la question (une question en langage courant
ne renvoie alors presque rien), et la config 'french' ne gere pas les accents oublies.
Le corpus fait quelques milliers de chunks : un index en memoire se construit en moins d'une seconde.

Pourquoi pas Snowball : il racinise "loyer" en "loi", ce qui melange loyer et loi partout.
On utilise un raciniseur leger (dans l'esprit de celui de Savoy), suffisant en recherche d'information.

Arabe (loi 47-06) : normalisation orthographique classique en recherche d'information (alef
hamza -> alef, ta marbuta -> ha, alef maqsura -> ya, voyelles supprimees) et raciniseur
Light10 (Larkey et al.) : prefixes "ال", "وال", "بال"... et suffixes "ات", "ون", "ها"...
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping

STOPWORDS = frozenset(
    """
    a au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur
    leurs lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa
    se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y ete etre est
    sont ont avoir fait faire peut peuvent doit doivent si ainsi dont lequel laquelle
    lesquels lesquelles cela ceci celui celle ceux celles tout tous toute toutes autre
    autres plus moins tres comme quand quel quelle quels quelles chaque
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def sans_accents(texte: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace("œ", "oe")


def raciniser(mot: str) -> str:
    """Raciniseur leger : pluriels et quelques suffixes frequents, sans toucher aux mots courts."""
    if len(mot) <= 3 or mot.isdigit():
        return mot
    if mot.endswith("aux") and len(mot) > 5:
        return mot[:-3] + "al"
    for suffixe in ("ations", "ation", "ements", "ement", "ments", "ment"):
        if mot.endswith(suffixe) and len(mot) - len(suffixe) >= 4:
            return mot[: -len(suffixe)]
    if mot[-1] in "sx":
        mot = mot[:-1]
    if mot.endswith("er") and len(mot) > 5:
        return mot[:-2]
    if mot.endswith("e") and len(mot) > 4:  # "locataire" -> "locatair", "annee" -> "anne"
        mot = mot[:-1]
    return mot


ARABE_RE = re.compile("[ء-ي]+")
# Mots vides arabes, deja normalises (على -> علي, إلى -> الي, التي -> التي)
STOPWORDS_AR = frozenset(
    "في من علي الي عن ان او ما لا لم لن هذا هذه ذلك تلك التي الذي الذين اللذين كل مع بعد قبل "
    "كما غير بين حتي عند ثم قد هو هي هم انه انها اي ب ل و ف ك اذا لدي".split()
)
PREFIXES_AR = ("وال", "بال", "كال", "فال", "لل", "ال")
SUFFIXES_AR = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def normaliser_arabe(mot: str) -> str:
    """Apres sans_accents (qui retire deja hamza, madda et voyelles) : ة -> ه, ى -> ي."""
    return mot.replace("ـ", "").replace("ة", "ه").replace("ى", "ي")


def raciniser_arabe(mot: str) -> str:
    """Light10 : un prefixe d'article, un "و" initial, puis les suffixes, sans descendre sous 2 lettres."""
    if len(mot) >= 4 and mot.startswith("و"):
        mot = mot[1:]
    for p in PREFIXES_AR:
        if mot.startswith(p) and len(mot) - len(p) >= 2:
            mot = mot[len(p):]
            break
    for s in SUFFIXES_AR:  # un seul passage dans l'ordre, comme Light10 (pas de sur-racinisation)
        if mot.endswith(s) and len(mot) - len(s) >= 2:
            mot = mot[: -len(s)]
    return mot


def tokeniser(texte: str) -> list[str]:
    """Un mot compose donne un jeton entier ("plus-value" -> "plusvalu", "67-12" -> "6712")
    en plus de ses parties : sinon "plus" (mot vide) disparait et il ne reste que "value".
    Les mots arabes passent par la normalisation et le raciniseur Light10."""
    normalise = sans_accents(texte.lower())
    jetons: list[str] = []
    for mot in ARABE_RE.findall(normalise):
        mot = normaliser_arabe(mot)
        if mot not in STOPWORDS_AR and len(mot) > 1:
            jetons.append(raciniser_arabe(mot))
    for mot in TOKEN_RE.findall(normalise):
        parties = mot.split("-")
        if len(parties) > 1:
            jetons.append(raciniser("".join(parties)))
        jetons += [raciniser(p) for p in parties if p not in STOPWORDS and (len(p) > 1 or p.isdigit())]
    return jetons


class BM25:
    def __init__(self, docs: Mapping[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = [Counter(tokeniser(docs[i])) for i in self.ids]
        self.longueurs = [sum(c.values()) for c in self.tf]
        self.moy = sum(self.longueurs) / max(len(self.ids), 1)
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        # IDF de Lucene : toujours positif
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, question: str) -> dict[str, float]:
        termes = [t for t in tokeniser(question) if t in self.idf]
        res: dict[str, float] = {}
        for i, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.longueurs[i] / self.moy)
            for t in termes:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                res[self.ids[i]] = s
        return res

    def search(self, question: str, k: int = 20, ids_autorises: Iterable[str] | None = None) -> list[str]:
        scores = self.scores(question)
        if ids_autorises is not None:
            autorises = set(ids_autorises)
            scores = {i: s for i, s in scores.items() if i in autorises}
        return [i for i, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]

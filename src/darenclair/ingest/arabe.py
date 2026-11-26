"""Regles propres aux textes de loi en arabe (ex. loi 47-06, version consolidee 2021).

- en-tetes d'articles : "المادة 5", "المادة الأولى", "المادة 120 المكررة" (bis), "مكرر مرتين" (ter) ;
- intitules : القسم (partie) > الباب (titre) > الفصل (chapitre) > الفرع (section) ;
- renvois : un "المادة 19" seul en debut de ligne peut etre un renvoi coupe par le retour a la
  ligne ("المادة 19 / من القانون رقم 12.90", "المادة 152 / أعلاه") : on regarde la ligne suivante ;
- mots coupes par une espace parasite dans le PDF source ("ع لى", "الب ناء") : recolles quand le
  mot entier est frequent ailleurs dans le meme texte et qu'un des morceaux n'est pas un mot.
"""

from __future__ import annotations

import re
from collections import Counter

LETTRE = "ء-ي"
MOT_RE = re.compile(f"[{LETTRE}]+")
CHIFFRES = str.maketrans("٠١٢٣٤٥٦٧٨٩"
                         "۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
TATWEEL = "ـ"


def est_arabe(texte: str) -> bool:
    arabe = len(re.findall(f"[{LETTRE}]", texte))
    latin = len(re.findall(r"[A-Za-z]", texte))
    return arabe > 3 * max(latin, 1)


WAW = "و"  # "و" (et) : seule lettre isolee qui soit un vrai mot


def recoller_mots(texte: str, frequence_min: int = 3) -> str:
    """Recolle "a b" en "ab" quand "ab" est un mot frequent du texte et que :
    - un morceau est une lettre isolee (sauf "و"), jamais un mot en arabe ; ou
    - un morceau est rare ; ou
    - le mot entier est bien plus frequent que les morceaux (coupure repetee, "ا لمادة")."""
    freq = Counter(MOT_RE.findall(texte))
    morceaux = re.split(f"([{LETTRE}]+)", texte)  # indices impairs = mots arabes
    i = 1
    while i + 2 < len(morceaux):
        a, espace, b = morceaux[i], morceaux[i + 1], morceaux[i + 2]
        entier = a + b
        lettre_isolee = (len(a) == 1 and a != WAW) or (len(b) == 1 and b != WAW)
        if espace == " " and freq[entier] >= frequence_min and (
            lettre_isolee
            or min(freq[a], freq[b]) < frequence_min
            or freq[entier] >= 5 * max(freq[a], freq[b])
        ):
            morceaux[i : i + 3] = [entier]
            continue  # le mot recolle peut encore l'etre avec le suivant
        # Morceaux inverses par le sens de lecture : "لى ع" pour "على" (seulement avec une
        # lettre isolee, qui n'est jamais un mot : sinon deux vrais mots seraient permutes)
        if espace == " " and lettre_isolee and freq[b + a] >= frequence_min and freq[entier] == 0:
            morceaux[i : i + 3] = [b + a]
            continue
        i += 2
    return "".join(morceaux)


# "المادة" seul sur sa ligne et numero sur la ligne suivante : on recolle avant que le filtre
# des numeros de page ne supprime la ligne "120".
ARTICLE_NUMERO_SEPARES_RE = re.compile("(المادة)[ \t]*\n[ \t]*(\\d+)[ \t]*(?=\n)")


def normaliser(texte: str) -> str:
    texte = texte.translate(CHIFFRES).replace(TATWEEL, "")
    if not est_arabe(texte):
        return texte
    texte = recoller_mots(texte)
    return ARTICLE_NUMERO_SEPARES_RE.sub(r"\1 \2", texte)


SUFFIXES = (
    ("مكرر ثلاث مرات", "quater"),  # مكرر ثلاث مرات
    ("مكرر مرتين", "ter"),  # مكرر مرتين
    ("المكررة", "bis"),  # المكررة
    ("المكرر", "bis"),  # المكرر
    ("مكررة", "bis"),  # مكررة
    ("مكرر", "bis"),  # مكرر
)
PREMIERE = ("الأولى", "الاولى")  # الأولى
ARTICLE_AR_RE = re.compile(
    "^[ \t]*المادة"  # المادة
    rf"[ \t]*\n?[ \t]*(\d+|{'|'.join(PREMIERE)})"
    rf"(?:[ \t]*\n?[ \t]*({'|'.join(re.escape(s) for s, _ in SUFFIXES)}))?"
    "[ \t]*$",
    re.MULTILINE,
)

# Premier mot d'une ligne qui suit un "المادة N" de renvoi : من (de), أعلاه (ci-dessus), أدناه
# (ci-dessous), بعده (apres), السالف/السالفة (precite), المشار (vise), المنصوص (prevu), و, أو...
SUITES_RENVOI = set(
    "من أعلاه أدناه بعده "
    "بعدها قبله السالف "
    "السالفة المشار المذكور "
    "المذكورة المنصوص بموجب "
    "و أو إلى في على عن ومن وفي".split()
)


def est_renvoi(texte: str, position: int) -> bool:
    suite = texte[position:position + 200].lstrip()
    if not suite or suite[0].isdigit():
        return True
    m = MOT_RE.search(suite[:40])
    return bool(m) and m.start() <= 3 and m.group() in SUITES_RENVOI


def numero(m: re.Match) -> str:
    n = "1" if m.group(1) in PREMIERE else m.group(1)
    suffixe = dict(SUFFIXES).get(m.group(2) or "")
    return f"{n} {suffixe}" if suffixe else n


ORDINAUX = (
    "الحادي عشر|الثاني عشر|"
    "الثالث عشر|الرابع عشر|"
    "الخامس عشر|الأول|الأولى|"
    "الثاني|الثانية|الثالث|"
    "الثالثة|الرابع|الرابعة|"
    "الخامس|الخامسة|السادس|"
    "السادسة|السابع|السابعة|"
    "الثامن|الثامنة|التاسع|"
    "التاسعة|العاشر|العاشرة|"
    "التمهيدي"
)
NIVEAUX_AR = {
    "القسم": 0,  # القسم
    "الجزء": 0,  # الجزء
    "الباب": 1,  # الباب
    "الفصل": 2,  # الفصل
    "الفرع": 3,  # الفرع
}
TITRE_AR_RE = re.compile(
    rf"^[ \t]*({'|'.join(NIVEAUX_AR)})(?:[ \t]+|[ \t]*\n[ \t]*)({ORDINAUX})"
    "(?:[ \t]+(?:المكرر|مكرر))?[ \t]*$",
    re.MULTILINE,
)

# Debut d'annexe : "الملحق" / "ملحق بالقانون رقم ..." (ou "ANNEXE" en francais), ligne courte.
ANNEXE_RE = re.compile(
    "^[ \t]*(?:ال ?ملحق|ملحق بالقانون"
    "|ANNEXES?|Annexes?)\\b[^\n]{0,60}$",
    re.MULTILINE,
)

ABROGE_AR_RE = re.compile(
    "^[(\\s.]*(نسخت|منسوخة|ملغاة|ألغيت)[)\\s.]*(.*)$"
)  # نسخت, منسوخة, ملغاة, ألغيت

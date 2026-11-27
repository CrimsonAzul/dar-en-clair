"""Calculs decides par le code, pas laisses au choix du modele.

Sur le jeu dev, les petits modeles locaux appelaient rarement les outils fiscaux (ou le mauvais).
Le calcul suit donc un chemin fixe :
1. Filtre : sans chiffre dans la question, pas de calcul.
2. Extraction a format impose (schema JSON, decodage contraint) : le modele dit QUEL calcul est
   demande et recopie les montants de la question. Il ne fait aucune operation.
3. Le code verifie que chaque montant extrait figure bien dans la question, fait lui-meme les
   conversions (loyer mensuel x 12) et appelle l'outil. Les exonerations ne sont jamais
   deduites ici : leurs conditions (habitation principale depuis 5 ans...) se discutent dans la
   reponse, avec les articles.
4. Le resultat est donne au modele pour sa reponse ; s'il l'oublie, le code l'ajoute.
"""

from __future__ import annotations

import re

from darenclair.tools.fiscal import NATURES_BIEN

CHIFFRE_RE = re.compile("[0-9٠-٩]")

# Tous les champs sont obligatoires (null si absent) : avec des champs facultatifs, le decodage
# contraint laissait parfois tomber le prix (vu sur dev avec qwen3.5:9b pour les droits d'enregistrement).
_MONTANT = {"type": ["number", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "calcul": {
            "type": "string",
            "enum": ["aucun", "impot_profit_immobilier", "impot_revenus_fonciers", "droits_enregistrement"],
        },
        "prix_vente": _MONTANT,
        "prix_achat": _MONTANT,
        "loyer_mensuel": _MONTANT,
        "loyers_annuels": _MONTANT,
        "nature_bien": {"type": "string", "enum": [*NATURES_BIEN, "inconnu"]},
    },
    "required": ["calcul", "prix_vente", "prix_achat", "loyer_mensuel", "loyers_annuels", "nature_bien"],
}

SYSTEME = """Tu lis une question sur l'immobilier au Maroc. Reponds uniquement en JSON.
calcul :
- "impot_revenus_fonciers" si la question demande l'impot a payer sur des loyers encaisses ;
- "droits_enregistrement" si elle demande les droits ou frais d'enregistrement d'un achat ;
- "impot_profit_immobilier" si elle demande l'impot a payer sur la vente d'un bien ;
- "aucun" dans tous les autres cas, ou s'il manque un montant necessaire.
Champs a remplir selon le calcul (null pour tous les autres) :
- impot_revenus_fonciers : loyer_mensuel (loyer par mois) ou loyers_annuels (total par an) ;
- droits_enregistrement : prix_achat et nature_bien ;
- impot_profit_immobilier : prix_vente et prix_achat.
Recopie les montants en dirhams tels qu'ils figurent dans la question (1 000 000 ou 1,000,000
donnent 1000000), sans faire aucune operation. nature_bien : local_construit pour un appartement,
une maison, une villa ou un local ; terrain_nu pour un terrain ; inconnu sinon."""

# Textes montres a l'usager : accents compris.
LIBELLES = {
    "impot_revenus_fonciers": ("impôt sur les loyers", "tax on rental income", "الضريبة على مداخيل الكراء"),
    "droits_enregistrement": ("droits d'enregistrement", "registration duties", "رسوم التسجيل"),
    "impot_profit_immobilier": ("impôt sur le profit de la vente", "tax on the sale profit",
                                "الضريبة على الربح العقاري"),
}
ARTICLES = {
    "impot_revenus_fonciers": "CGI art. 73-II-B-5° et C-4°",
    "droits_enregistrement": "CGI art. 133-I-F et G",
    "impot_profit_immobilier": "CGI art. 65, 73-II-F-6° et 144-II",
}


def montant_present(texte: str, montant: float) -> bool:
    """36 000, 36.000, 36,000 ou 36000 : on compare les nombres sans separateurs de milliers."""
    nombres = {re.sub(r"[\s .,]", "", m) for m in re.findall(r"\d[\d\s .,]*\d|\d", texte)}
    return str(int(montant)) in nombres


def a_calculer(question: str) -> bool:
    return bool(CHIFFRE_RE.search(question))


def appel_depuis_extraction(extraction: dict, question: str) -> tuple[str, dict] | None:
    """Traduit l'extraction en appel d'outil, ou None si un montant manque ou n'est pas dans la question."""

    def montant(cle: str) -> float | None:
        v = extraction.get(cle)
        if isinstance(v, bool) or not isinstance(v, int | float) or v <= 0:
            return None
        return float(v) if montant_present(question, v) else None  # pas de montant invente

    calcul = extraction.get("calcul")
    if calcul == "impot_revenus_fonciers":
        if (annuel := montant("loyers_annuels")) is not None:
            return calcul, {"loyers_bruts_annuels": annuel}
        if (mensuel := montant("loyer_mensuel")) is not None:
            return calcul, {"loyers_bruts_annuels": mensuel * 12}
    elif calcul == "droits_enregistrement":
        # Un seul prix compte ici, celui de la transaction : le modele le range parfois en prix_vente.
        prix = montant("prix_achat") or montant("prix_vente")
        nature = extraction.get("nature_bien")
        if prix is not None and nature in NATURES_BIEN:
            return calcul, {"prix": prix, "nature": nature}
    elif calcul == "impot_profit_immobilier":
        vente, achat = montant("prix_vente"), montant("prix_achat")
        if vente is not None and achat is not None:
            return calcul, {"prix_cession": vente, "prix_acquisition": achat}
    return None


def dh(montant: float) -> str:
    entier = round(montant) == montant
    return f"{montant:,.{0 if entier else 2}f}".replace(",", " ")


def bloc_pour_le_modele(appel: dict) -> str:
    r = appel["resultat"]["resultat"]
    detail = ", ".join(f"{k} {dh(v) if isinstance(v, float) and v >= 1 else v}" for k, v in r["detail"].items())
    return (
        f"Calcul fait par l'outil, a reprendre tel quel sans rien recalculer : "
        f"{LIBELLES[appel['outil']][0]} = {dh(r['montant'])} DH ({detail} ; formule : {r['formule']} ; "
        f"{ARTICLES[appel['outil']]})."
    )


def ligne_si_oubli(appel: dict, langue: str) -> str:
    montant = dh(appel["resultat"]["resultat"]["montant"])
    fr, en, ar = LIBELLES[appel["outil"]]
    if langue == "en":
        return f"Calculated amount: {montant} DH ({en}, {ARTICLES[appel['outil']]})."
    if langue == "ar":
        return f"المبلغ المحسوب: {montant} درهم ({ar})."
    return f"Montant calculé par l'outil : {montant} DH ({fr}, {ARTICLES[appel['outil']]})."

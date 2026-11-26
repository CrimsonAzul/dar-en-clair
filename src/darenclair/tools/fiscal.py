"""Calculs fiscaux immobiliers. Le LLM appelle ces fonctions au lieu de calculer lui-meme.

Les taux ne sont pas codes en dur : ils viennent de config/fiscal_2026.yaml. charger_taux()
refuse de renvoyer une section tant qu'elle contient une valeur vide ou n'est pas marquee
verifie: true. Mieux vaut un outil qui refuse qu'un impot faux affiche avec assurance.
Les regles de calcul suivent le texte du CGI 2026 ; les articles sont cites a chaque etape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DEFAUT = Path(__file__).resolve().parents[3] / "config" / "fiscal_2026.yaml"

# Art. 63 : exonerations du profit foncier. Leurs conditions (duree d'occupation, total des
# cessions de l'annee, lien de parente) ne se calculent pas : elles sont decidees en amont.
EXONERATIONS = {
    "63-II-A": "cessions de l'annee d'une valeur totale inferieure ou egale a 140 000 DH",
    "63-II-B": "habitation principale depuis au moins 5 ans",
    "63-II-C": "droits indivis d'immeubles agricoles entre coheritiers",
    "63-II-D": "logement social, habitation principale depuis au moins 4 ans",
    "63-III": "cession a titre gratuit entre ascendants, descendants, epoux, freres et soeurs",
}

NATURES_BIEN = ("local_construit", "terrain_nu")


class TauxNonVerifies(RuntimeError):
    pass


def chemin_config() -> Path:
    # En Docker le paquet est installe dans site-packages : le chemin vient de FISCAL_CONFIG.
    return Path(os.getenv("FISCAL_CONFIG", CONFIG_DEFAUT))


def charger_taux(section: str, path: Path | None = None) -> dict[str, float]:
    import yaml

    with (path or chemin_config()).open(encoding="utf-8") as f:
        donnees = yaml.safe_load(f)[section]
    vides = [k for k, v in donnees.items() if v is None]
    if not donnees.get("verifie") or vides:
        raise TauxNonVerifies(f"section '{section}' non verifiee ou incomplete : {vides}")
    return {k: float(v) for k, v in donnees.items() if k != "verifie"}


@dataclass
class Resultat:
    montant: float
    detail: dict[str, float] = field(default_factory=dict)
    formule: str = ""


def _positifs(**valeurs: float) -> None:
    for nom, v in valeurs.items():
        if v < 0:
            raise ValueError(f"{nom} doit etre positif ou nul")


def profit_immobilier(
    prix_cession: float,
    prix_acquisition: float,
    *,
    taux: float,
    taux_minimum: float,
    forfait_frais_acquisition: float,
    seuil_minimum_habitation_principale: float,
    frais_acquisition_justifies: float = 0.0,
    investissements: float = 0.0,
    coefficient_reevaluation: float = 1.0,
    interets_pret: float = 0.0,
    frais_cession: float = 0.0,
    exoneration: str | None = None,
) -> Resultat:
    """Impot sur le profit d'une cession immobiliere (TPI).

    frais  = max(prix_acquisition x forfait, frais justifies)               art. 65-II
    cout   = (prix_acquisition + frais + investissements + interets) x coef  art. 65-II
    profit = max(0, prix_cession - frais_cession - cout)                     art. 65-I
    impot  = max(profit x taux, prix_cession x taux_minimum)                 art. 73-II-F-6°, 144-II-1°

    Habitation principale exoneree (63-II-B) : minimum de taux_minimum sur la part du prix
    au-dessus du seuil (144-II-2°). Autres exonerations : pas d'impot.
    """
    _positifs(
        prix_cession=prix_cession, prix_acquisition=prix_acquisition,
        frais_acquisition_justifies=frais_acquisition_justifies, investissements=investissements,
        interets_pret=interets_pret, frais_cession=frais_cession,
    )
    if coefficient_reevaluation < 1:
        raise ValueError("coefficient_reevaluation doit etre >= 1")
    if exoneration is not None and exoneration not in EXONERATIONS:
        raise ValueError(f"exoneration inconnue : {exoneration} (attendu : {sorted(EXONERATIONS)})")

    if exoneration == "63-II-B":
        fraction = max(0.0, prix_cession - seuil_minimum_habitation_principale)
        return Resultat(
            round(fraction * taux_minimum, 2),
            {"fraction_au_dessus_du_seuil": round(fraction, 2)},
            "habitation principale exoneree : minimum sur la part du prix au-dessus du seuil (144-II-2°)",
        )
    if exoneration:
        return Resultat(0.0, {}, f"cession exoneree ({exoneration} : {EXONERATIONS[exoneration]})")

    frais = max(prix_acquisition * forfait_frais_acquisition, frais_acquisition_justifies)
    cout = (prix_acquisition + frais + investissements + interets_pret) * coefficient_reevaluation
    profit = max(0.0, prix_cession - frais_cession - cout)
    impot_profit = profit * taux
    minimum = prix_cession * taux_minimum
    return Resultat(
        round(max(impot_profit, minimum), 2),
        {
            "frais_acquisition_retenus": round(frais, 2),
            "cout_revalue": round(cout, 2),
            "profit": round(profit, 2),
            "impot_sur_profit": round(impot_profit, 2),
            "cotisation_minimale": round(minimum, 2),
        },
        "max(profit x taux, prix de cession x taux minimum)",
    )


def impot_revenus_fonciers(
    loyers_bruts_annuels: float, *, taux_reduit: float, taux_normal: float, seuil: float
) -> Resultat:
    """Impot sur les loyers : taux sur le montant brut annuel, selon qu'il est inferieur au seuil
    (73-II-B-5°) ou egal ou superieur (73-II-C-4°)."""
    _positifs(loyers_bruts_annuels=loyers_bruts_annuels)
    taux = taux_reduit if loyers_bruts_annuels < seuil else taux_normal
    return Resultat(
        round(loyers_bruts_annuels * taux, 2),
        {"loyers_bruts_annuels": round(loyers_bruts_annuels, 2), "taux": taux},
        "loyers bruts annuels x taux",
    )


def droits_enregistrement(
    prix: float, nature: str, *, taux_local_construit: float, taux_terrain_nu: float
) -> Resultat:
    """Droits d'enregistrement a l'achat : local construit (133-I-F-1°) ou terrain nu (133-I-G-1°)."""
    _positifs(prix=prix)
    if nature not in NATURES_BIEN:
        raise ValueError(f"nature inconnue : {nature} (attendu : {NATURES_BIEN})")
    taux = taux_local_construit if nature == "local_construit" else taux_terrain_nu
    return Resultat(round(prix * taux, 2), {"taux": taux}, "prix x taux")


def frais_acquisition(
    prix: float,
    *,
    taux_enregistrement: float,
    taux_conservation: float,
    frais_fixes_conservation: float = 0.0,
    taux_honoraires_notaire: float = 0.0,
    taux_tva_honoraires: float = 0.0,
) -> Resultat:
    """Frais a la charge de l'acheteur : enregistrement, conservation fonciere, notaire."""
    _positifs(prix=prix)
    enregistrement = prix * taux_enregistrement
    conservation = prix * taux_conservation + frais_fixes_conservation
    honoraires = prix * taux_honoraires_notaire
    tva = honoraires * taux_tva_honoraires
    detail = {
        "enregistrement": round(enregistrement, 2),
        "conservation_fonciere": round(conservation, 2),
        "honoraires_notaire": round(honoraires, 2),
        "tva_honoraires": round(tva, 2),
    }
    return Resultat(round(sum(detail.values()), 2), detail, "somme des frais")

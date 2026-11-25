"""Calculs fiscaux immobiliers. Le LLM appelle ces fonctions au lieu de calculer lui-meme.

Les taux ne sont PAS codes en dur ici : ils viennent de config/fiscal_2026.yaml,
ou chaque valeur doit etre verifiee dans le texte officiel (article cite a cote).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "config" / "fiscal_2026.yaml"


def charger_taux(path: Path = CONFIG) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Resultat:
    montant: float
    detail: dict[str, float] = field(default_factory=dict)
    formule: str = ""


def profit_immobilier(
    prix_cession: float,
    prix_acquisition: float,
    *,
    taux: float,
    taux_minimum: float,
    frais_acquisition: float = 0.0,
    frais_cession: float = 0.0,
    coefficient_reevaluation: float = 1.0,
    exonere: bool = False,
) -> Resultat:
    """Impot sur le profit d'une cession immobiliere (TPI).

    profit = (prix_cession - frais_cession) - (prix_acquisition + frais_acquisition) * coefficient
    impot  = max(profit * taux, prix_cession * taux_minimum)
    Les cas d'exoneration (habitation principale, seuils...) sont decides en amont : exonere=True.
    """
    if prix_cession < 0 or prix_acquisition < 0:
        raise ValueError("les prix doivent etre positifs")
    if exonere:
        return Resultat(0.0, {"exonere": 1.0}, "cession exoneree")

    cout_revalue = (prix_acquisition + frais_acquisition) * coefficient_reevaluation
    profit = max(0.0, prix_cession - frais_cession - cout_revalue)
    impot_profit = profit * taux
    minimum = prix_cession * taux_minimum
    impot = max(impot_profit, minimum)
    return Resultat(
        round(impot, 2),
        {
            "cout_revalue": round(cout_revalue, 2),
            "profit": round(profit, 2),
            "impot_sur_profit": round(impot_profit, 2),
            "cotisation_minimale": round(minimum, 2),
        },
        "max(profit x taux, prix de cession x taux minimum)",
    )


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
    if prix < 0:
        raise ValueError("le prix doit etre positif")
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

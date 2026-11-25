"""Calculs fiscaux immobiliers. Le LLM appelle ces fonctions au lieu de calculer lui-meme.

Les taux ne sont pas codes en dur : ils viennent de config/fiscal_2026.yaml. charger_taux()
refuse de renvoyer une section tant qu'elle contient une valeur vide ou n'est pas marquee
verifie: true. Mieux vaut un outil qui refuse qu'un impot faux affiche avec assurance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DEFAUT = Path(__file__).resolve().parents[3] / "config" / "fiscal_2026.yaml"


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
    frais_acquisition: float = 0.0,
    investissements: float = 0.0,
    coefficient_reevaluation: float = 1.0,
    interets_pret: float = 0.0,
    frais_cession: float = 0.0,
    exonere: bool = False,
) -> Resultat:
    """Impot sur le profit d'une cession immobiliere (TPI).

    cout   = (prix_acquisition + frais_acquisition + investissements) x coefficient + interets_pret
    profit = max(0, prix_cession - frais_cession - cout)
    impot  = max(profit x taux, prix_cession x taux_minimum)

    Structure a confirmer dans le CGI (elements deductibles, regles de reevaluation, cas ou le
    minimum s'applique). Les exonerations sont decidees en amont et passees via exonere=True.
    """
    _positifs(
        prix_cession=prix_cession, prix_acquisition=prix_acquisition,
        frais_acquisition=frais_acquisition, investissements=investissements,
        interets_pret=interets_pret, frais_cession=frais_cession,
    )
    if coefficient_reevaluation < 1:
        raise ValueError("coefficient_reevaluation doit etre >= 1")
    if exonere:
        return Resultat(0.0, {"exonere": 1.0}, "cession exoneree")

    cout = (prix_acquisition + frais_acquisition + investissements) * coefficient_reevaluation
    cout += interets_pret
    profit = max(0.0, prix_cession - frais_cession - cout)
    impot_profit = profit * taux
    minimum = prix_cession * taux_minimum
    return Resultat(
        round(max(impot_profit, minimum), 2),
        {
            "cout_revalue": round(cout, 2),
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

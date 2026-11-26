"""Outils fiscaux proposes au LLM (appel de fonction). Il ne calcule jamais lui-meme un impot :
il appelle ces fonctions, qui lisent les taux verifies de config/fiscal_2026.yaml."""

from __future__ import annotations

from dataclasses import asdict

from darenclair.tools import fiscal

_NOMBRE = {"type": "number"}

OUTILS = [
    {
        "name": "impot_profit_immobilier",
        "description": (
            "Impot paye par le VENDEUR sur le profit d'une vente immobiliere (CGI art. 65, 73-II-F-6°, "
            "144-II). Applique le forfait de 15 % de frais d'acquisition et le minimum de 3 % du prix de vente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prix_cession": {**_NOMBRE, "description": "prix de vente en dirhams"},
                "prix_acquisition": {**_NOMBRE, "description": "prix d'achat en dirhams"},
                "frais_acquisition_justifies": {**_NOMBRE, "description": "frais d'achat justifies, si connus"},
                "investissements": {**_NOMBRE, "description": "travaux de construction ou d'amelioration justifies"},
                "coefficient_reevaluation": {**_NOMBRE, "description": "coefficient de l'annee d'achat, 1 si inconnu"},
                "interets_pret": {**_NOMBRE, "description": "interets d'emprunt payes pour l'achat"},
                "frais_cession": {**_NOMBRE, "description": "frais de vente justifies (courtage, annonces)"},
                "exoneration": {
                    "type": "string",
                    "enum": sorted(fiscal.EXONERATIONS),
                    "description": "article d'exoneration si les conditions sont remplies (63-II-B : habitation principale depuis 5 ans)",
                },
            },
            "required": ["prix_cession", "prix_acquisition"],
        },
    },
    {
        "name": "impot_revenus_fonciers",
        "description": (
            "Impot paye par le PROPRIETAIRE qui loue un bien, sur le total annuel brut des loyers "
            "(CGI art. 73-II-B-5° et C-4°). Loyer mensuel : multiplier par 12 avant l'appel."
        ),
        "parameters": {
            "type": "object",
            "properties": {"loyers_bruts_annuels": {**_NOMBRE, "description": "total des loyers de l'annee en dirhams"}},
            "required": ["loyers_bruts_annuels"],
        },
    },
    {
        "name": "droits_enregistrement",
        "description": (
            "Droits d'enregistrement payes par l'ACHETEUR lors de l'achat d'un bien (CGI art. 133-I-F "
            "et G) : local_construit pour un appartement, une maison ou un local, terrain_nu pour un terrain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prix": {**_NOMBRE, "description": "prix d'achat en dirhams"},
                "nature": {"type": "string", "enum": list(fiscal.NATURES_BIEN)},
            },
            "required": ["prix", "nature"],
        },
    },
]

_FONCTIONS = {
    "impot_profit_immobilier": (fiscal.profit_immobilier, "profit_immobilier"),
    "impot_revenus_fonciers": (fiscal.impot_revenus_fonciers, "revenus_fonciers"),
    "droits_enregistrement": (fiscal.droits_enregistrement, "droits_enregistrement"),
}


def executer(nom: str, arguments: dict) -> dict:
    """Appelle l'outil demande. Les erreurs sont renvoyees au LLM en clair, jamais levees."""
    if nom not in _FONCTIONS:
        return {"erreur": f"outil inconnu : {nom}"}
    fonction, section = _FONCTIONS[nom]
    try:
        taux = fiscal.charger_taux(section)
        args = {k: v for k, v in arguments.items() if v is not None}
        return {"resultat": asdict(fonction(**args, **taux))}
    except fiscal.TauxNonVerifies:
        return {"erreur": "taux non verifies dans les textes officiels : calcul indisponible"}
    except (TypeError, ValueError) as e:
        return {"erreur": f"arguments invalides : {e}"}

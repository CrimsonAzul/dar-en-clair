"""Tests de la logique de calcul avec des taux fictifs (les vrais taux sont dans config/)."""

import pytest

from darenclair.tools.fiscal import (
    TauxNonVerifies,
    charger_taux,
    frais_acquisition,
    profit_immobilier,
)

TAUX_FICTIFS = {"taux": 0.20, "taux_minimum": 0.03}


def test_impot_sur_profit_quand_profit_eleve():
    r = profit_immobilier(1_000_000, 500_000, **TAUX_FICTIFS)
    assert r.detail["profit"] == 500_000
    assert r.montant == 100_000


def test_cotisation_minimale_quand_profit_faible():
    assert profit_immobilier(1_000_000, 990_000, **TAUX_FICTIFS).montant == 30_000


def test_perte_donne_minimum():
    r = profit_immobilier(800_000, 900_000, **TAUX_FICTIFS)
    assert r.detail["profit"] == 0
    assert r.montant == 24_000


def test_couts_et_reevaluation():
    r = profit_immobilier(
        1_000_000, 500_000, frais_acquisition=40_000, investissements=10_000,
        coefficient_reevaluation=1.1, interets_pret=5_000, frais_cession=10_000, **TAUX_FICTIFS,
    )
    assert r.detail["cout_revalue"] == 610_000  # 550 000 x 1,1 + 5 000
    assert r.detail["profit"] == 380_000


def test_exoneration():
    assert profit_immobilier(1_000_000, 100_000, exonere=True, **TAUX_FICTIFS).montant == 0


def test_entrees_invalides():
    with pytest.raises(ValueError):
        profit_immobilier(-1, 0, **TAUX_FICTIFS)
    with pytest.raises(ValueError):
        profit_immobilier(1, 0, coefficient_reevaluation=0.5, **TAUX_FICTIFS)


def test_frais_acquisition_somme():
    r = frais_acquisition(
        1_000_000, taux_enregistrement=0.04, taux_conservation=0.015,
        frais_fixes_conservation=200, taux_honoraires_notaire=0.01, taux_tva_honoraires=0.10,
    )
    assert r.detail["conservation_fonciere"] == 15_200
    assert r.montant == 40_000 + 15_200 + 10_000 + 1_000


def test_config_refusee_tant_que_non_verifiee(tmp_path):
    cfg = tmp_path / "f.yaml"
    cfg.write_text("s:\n  taux: null\n  verifie: false\n", encoding="utf-8")
    with pytest.raises(TauxNonVerifies):
        charger_taux("s", cfg)
    cfg.write_text("s:\n  taux: 0.2\n  verifie: true\n", encoding="utf-8")
    assert charger_taux("s", cfg) == {"taux": 0.2}


def test_vraie_config_pas_encore_remplie():
    # Rappel volontaire : ce test echouera quand les taux seront remplis ; le supprimer alors.
    with pytest.raises(TauxNonVerifies):
        charger_taux("profit_immobilier")

"""Tests de la logique de calcul avec des taux fictifs (les vrais taux sont dans config/)."""

import pytest

from darenclair.tools.fiscal import frais_acquisition, profit_immobilier

TAUX_FICTIFS = {"taux": 0.20, "taux_minimum": 0.03}


def test_impot_sur_profit_quand_profit_eleve():
    r = profit_immobilier(1_000_000, 500_000, **TAUX_FICTIFS)
    assert r.detail["profit"] == 500_000
    assert r.montant == 100_000  # 500 000 x 20 % > 1 000 000 x 3 %


def test_cotisation_minimale_quand_profit_faible():
    r = profit_immobilier(1_000_000, 990_000, **TAUX_FICTIFS)
    assert r.montant == 30_000  # minimum = 3 % du prix de cession


def test_perte_donne_minimum():
    r = profit_immobilier(800_000, 900_000, **TAUX_FICTIFS)
    assert r.detail["profit"] == 0
    assert r.montant == 24_000


def test_frais_et_reevaluation():
    r = profit_immobilier(
        1_000_000, 500_000, frais_acquisition=50_000, frais_cession=10_000,
        coefficient_reevaluation=1.1, **TAUX_FICTIFS,
    )
    assert r.detail["cout_revalue"] == 605_000
    assert r.detail["profit"] == 385_000


def test_exoneration():
    assert profit_immobilier(1_000_000, 100_000, exonere=True, **TAUX_FICTIFS).montant == 0


def test_prix_negatif_refuse():
    with pytest.raises(ValueError):
        profit_immobilier(-1, 0, **TAUX_FICTIFS)


def test_frais_acquisition_somme():
    r = frais_acquisition(
        1_000_000, taux_enregistrement=0.04, taux_conservation=0.015,
        frais_fixes_conservation=200, taux_honoraires_notaire=0.01, taux_tva_honoraires=0.10,
    )
    assert r.detail["enregistrement"] == 40_000
    assert r.detail["conservation_fonciere"] == 15_200
    assert r.detail["tva_honoraires"] == 1_000
    assert r.montant == 40_000 + 15_200 + 10_000 + 1_000

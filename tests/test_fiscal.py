"""Tests de la logique de calcul. Taux passes en argument (memes valeurs que le CGI 2026) pour
que les tests ne dependent pas du fichier de config ; les resultats sont calcules a la main."""

import pytest

from darenclair.tools.fiscal import (
    TauxNonVerifies,
    charger_taux,
    droits_enregistrement,
    frais_acquisition,
    impot_revenus_fonciers,
    profit_immobilier,
)

TPI = {
    "taux": 0.20, "taux_minimum": 0.03, "forfait_frais_acquisition": 0.15,
    "seuil_minimum_habitation_principale": 4_000_000,
}
LOYERS = {"taux_reduit": 0.10, "taux_normal": 0.15, "seuil": 120_000}
ENREGISTREMENT = {"taux_local_construit": 0.04, "taux_terrain_nu": 0.05}


def test_impot_sur_profit_avec_forfait_de_frais():
    r = profit_immobilier(1_000_000, 500_000, **TPI)
    assert r.detail["frais_acquisition_retenus"] == 75_000  # 15 % de 500 000
    assert r.detail["profit"] == 425_000
    assert r.montant == 85_000


def test_frais_justifies_retenus_seulement_si_superieurs_au_forfait():
    assert profit_immobilier(1_000_000, 500_000, frais_acquisition_justifies=90_000, **TPI).montant == 82_000
    assert profit_immobilier(1_000_000, 500_000, frais_acquisition_justifies=50_000, **TPI).montant == 85_000


def test_cotisation_minimale_quand_profit_faible():
    # Terrain achete 900 000, vendu 1 000 000 : cout 1 035 000 avec le forfait, pas de profit.
    r = profit_immobilier(1_000_000, 900_000, **TPI)
    assert r.detail["profit"] == 0
    assert r.montant == 30_000


def test_perte_donne_minimum():
    assert profit_immobilier(800_000, 900_000, **TPI).montant == 24_000


def test_reevaluation_porte_aussi_sur_investissements_et_interets():
    r = profit_immobilier(
        1_000_000, 400_000, investissements=20_000, coefficient_reevaluation=1.1,
        interets_pret=20_000, frais_cession=10_000, **TPI,
    )
    assert r.detail["cout_revalue"] == 550_000  # (400 000 + 60 000 + 20 000 + 20 000) x 1,1
    assert r.detail["profit"] == 440_000
    assert r.montant == 88_000


def test_habitation_principale_minimum_au_dessus_de_4_millions():
    assert profit_immobilier(5_000_000, 1_000_000, exoneration="63-II-B", **TPI).montant == 30_000
    assert profit_immobilier(3_000_000, 1_000_000, exoneration="63-II-B", **TPI).montant == 0


def test_autres_exonerations():
    assert profit_immobilier(130_000, 10_000, exoneration="63-II-A", **TPI).montant == 0
    with pytest.raises(ValueError):
        profit_immobilier(130_000, 10_000, exoneration="inventee", **TPI)


def test_entrees_invalides():
    with pytest.raises(ValueError):
        profit_immobilier(-1, 0, **TPI)
    with pytest.raises(ValueError):
        profit_immobilier(1, 0, coefficient_reevaluation=0.5, **TPI)


def test_revenus_fonciers_seuil_de_120000():
    assert impot_revenus_fonciers(48_000, **LOYERS).montant == 4_800  # 4 000 DH par mois
    assert impot_revenus_fonciers(119_999, **LOYERS).montant == 11_999.9
    assert impot_revenus_fonciers(120_000, **LOYERS).montant == 18_000  # "egal ou superieur"


def test_droits_enregistrement():
    assert droits_enregistrement(900_000, "local_construit", **ENREGISTREMENT).montant == 36_000
    assert droits_enregistrement(500_000, "terrain_nu", **ENREGISTREMENT).montant == 25_000
    with pytest.raises(ValueError):
        droits_enregistrement(500_000, "villa", **ENREGISTREMENT)


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


def test_vraie_config():
    # Les sections verifiees se branchent telles quelles sur les fonctions de calcul.
    assert profit_immobilier(1_000_000, 500_000, **charger_taux("profit_immobilier")).montant == 85_000
    assert impot_revenus_fonciers(48_000, **charger_taux("revenus_fonciers")).montant == 4_800
    assert droits_enregistrement(1_000_000, "local_construit", **charger_taux("droits_enregistrement")).montant == 40_000
    # Conservation fonciere et notaire : pas de source officielle dans le corpus.
    with pytest.raises(TauxNonVerifies):
        charger_taux("frais_acquisition")

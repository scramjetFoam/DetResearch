import cantera as ct
import numpy as np
import pytest

from funcs.simulation import thermo


class TestDilutedSpeciesDict:
    def test_single_species_diluent(self):
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2",
            dil_frac
        )

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"]
            ]
        )

    def test_multi_species_diluent(self):
        mol_co2 = 5
        mol_ar = 3
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / spec_dil["AR"],
                spec_dil["CO2"] + spec_dil["AR"]
            ]
        )

    def test_single_species_diluent_plus_ox(self):
        mol_co2 = 0
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / ar_adjusted,
                spec_dil["CO2"] + ar_adjusted
            ]
        )

    def test_multi_species_diluent_plus_ox(self):
        mol_co2 = 1
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,          # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac           # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / ar_adjusted,
                spec_dil["CO2"] + ar_adjusted
            ]
        )


class TestEnforceSpeciesList:
    def test_good_inputs(self):
        test_inputs = [
            'asdf',
            ['asdf', 'ghjk'],
            'ASDF',
            ['ASDF', 'GHJK']
        ]
        good_results = [['ASDF'], ['ASDF', 'GHJK']] * 2

        checks = []
        for current_input, good in zip(test_inputs, good_results):
            # noinspection PyProtectedMember
            checks.append(
                thermo._enforce_species_list(current_input) == good
            )

        assert all(checks)

    @pytest.mark.parametrize(
        'test_input',
        [None, [None, None], 1, [0, 1, 2]]
    )
    def test_bad_input(self, test_input):
        if isinstance(test_input, list):
            current_type = [type(item) for item in test_input]
        else:
            current_type = type(test_input)
        try:
            # noinspection PyProtectedMember
            thermo._enforce_species_list(test_input)
        except TypeError as err:
            assert str(err) == 'Bad species type: %s' % current_type


def test_solution_with_inerts():
    mechanism = 'gri30.cti'
    inert = 'O'
    known_gas = thermo.ORIGINAL_SOLUTION(mechanism)
    remaining_reactions = known_gas.n_reactions - sum(
        [inert in rxn.reactants or inert in rxn.products for
         rxn in known_gas.reactions()]
    )

    test_gas = thermo.solution_with_inerts(mechanism, inert)
    # use len(forward_rate_constants) rather than n_reactions because an
    # improperly built gas object will throw an error on forward_rate_constants
    # but not n_reactions
    assert len(test_gas.forward_rate_constants) == remaining_reactions

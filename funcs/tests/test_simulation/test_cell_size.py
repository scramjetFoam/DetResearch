import numpy as np

import funcs.simulation.cell_size as cs

relTol = 1e-4
absTol = 1e-6


# noinspection PyProtectedMember
class TestAgainstDemo:
    cj_speed = 1967.8454767711942
    init_press = 100000
    init_temp = 300
    mechanism = 'Mevel2017.cti'

    def test_calculations_are_correct(self):
        c = cs.CellSize()
        c(
            base_mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            equivalence=1,
            diluent='None',
            diluent_mol_frac=0,
            cj_speed=self.cj_speed
        )
        # Test calculated values against demo script
        original_cell_sizes = cs.ModelResults(
            gavrikov=1.9316316546518768e-02,
            ng=6.5644825968914763e-03,
            westbrook=3.4852910825972942e-03,
        )
        original_induction_lengths = cs.ModelResults(
            gavrikov=0.0001137734347788197,
            ng=0.00014438224858385156,
            westbrook=0.00012018245112404462,
        )

        assert np.allclose(c.cell_size.values(), original_cell_sizes.values(), rtol=relTol, atol=absTol)
        assert np.allclose(c.induction_length.values(), original_induction_lengths.values(), rtol=relTol, atol=absTol)

    def test_build_gas_no_dilution(self):
        c = cs.CellSize()
        undiluted = {'H2': 2 / 3, 'O2': 1 / 3}

        # should not dilute with diluent=None
        c.mechanism = 'Mevel2017.cti'
        c.initial_temp = self.init_temp
        c.initial_press = self.init_press
        c.fuel = 'H2'
        c.oxidizer = 'O2'
        c.equivalence = 1
        c.diluent = None
        c.diluent_mol_frac = 0.5
        c.perturbed_reaction = -1
        test = c._build_gas_object().mole_fraction_dict()
        check_none = [
            np.isclose(undiluted[key], value) for key, value in test.items()
        ]

        # should not dilute with diluent_mol_frac=0
        c.diluent = 'AR'
        c.diluent_mol_frac = 0
        test = c._build_gas_object().mole_fraction_dict()
        check_zero = [
            np.isclose(undiluted[key], value) for key, value in test.items()
        ]
        assert all([check_none, check_zero])

    def test_build_gas_with_dilution(self):
        c = cs.CellSize()
        c.mechanism = 'Mevel2017.cti'
        c.initial_temp = self.init_temp
        c.initial_press = self.init_press
        c.fuel = 'H2'
        c.oxidizer = 'O2'
        c.equivalence = 1
        c.diluent = 'AR'
        c.diluent_mol_frac = 0.1
        c.perturbed_reaction = -1
        test = c._build_gas_object().mole_fraction_dict()
        check = [
            np.isclose(test['H2'] / test['O2'], 2),
            np.isclose(test['AR'], 0.1)
        ]
        assert all(check)

    def test_perturbed(self):
        c = cs.CellSize()
        pert = 3
        pert_frac = 0.01
        c(
            base_mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            equivalence=1,
            diluent=None,
            diluent_mol_frac=0,
            cj_speed=self.cj_speed,
            perturbed_reaction=pert,
            perturbation_fraction=pert_frac
        )
        n_rxns = c.base_gas.n_reactions
        correct_multipliers = np.ones(n_rxns)
        correct_multipliers[pert] = 1 + pert_frac
        multipliers = [c.base_gas.multiplier(i) for i in range(n_rxns)]
        assert np.allclose(multipliers, correct_multipliers)

    def test_perturbed_diluted(self):
        c = cs.CellSize()
        pert = 3
        pert_frac = 0.01
        c(
            base_mechanism='Mevel2017.cti',
            initial_temp=300,
            initial_press=101325,
            fuel='H2',
            oxidizer='O2',
            equivalence=1,
            diluent='AR',
            diluent_mol_frac=0.02,
            cj_speed=2834.9809153464994,
            perturbed_reaction=pert,
            perturbation_fraction=pert_frac
        )
        n_rxns = c.base_gas.n_reactions
        correct_multipliers = np.ones(n_rxns)
        correct_multipliers[pert] = 1 + pert_frac
        multipliers = [c.base_gas.multiplier(i) for i in range(n_rxns)]
        assert np.allclose(multipliers, correct_multipliers)


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    subprocess.check_call(
        'pytest test_cell_size.py -vv --noconftest --cov '
        '--cov-report html'
    )

import cantera as ct
import numpy as np

import funcs.simulation.cell_size as cs

relTol = 1e-4
absTol = 1e-6


# noinspection PyProtectedMember
class TestAgainstDemo:
    # Test calculated values against demo script
    original_cell_sizes = {
        'Gavrikov': 1.9316316546518768e-02,
        'Ng': 6.5644825968914763e-03,
        'Westbrook': 3.4852910825972942e-03
    }

    original_induction_lengths = {
        'Gavrikov': 0.0001137734347788197,
        'Ng': 0.00014438224858385156,
        'Westbrook': 0.00012018245112404462
    }

    cj_speed = 1967.8454767711942
    init_press = 100000
    init_temp = 300
    mechanism = 'Mevel2017.cti'

    def test_induction_lengths(self):
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
        assert (
            all([[
                abs(length - c.induction_length[correlation]) / length < relTol
                for correlation, length in
                self.original_induction_lengths.items()
            ], [
                abs(length - c.induction_length[correlation]) < absTol
                for correlation, length in
                self.original_induction_lengths.items()
            ]
            ])
        )

    def test_cell_sizes(self):
        c = cs.CellSize()
        test = c(
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
        assert (
            all([[
                abs(cell - test[correlation]) / cell < relTol
                for correlation, cell in self.original_cell_sizes.items()
            ], [
                abs(cell - test[correlation]) < absTol
                for correlation, cell in self.original_cell_sizes.items()
            ]]
            )
        )

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
    from tests.test_simulation.test_database import remove_stragglers
    try:
        subprocess.check_call(
            'pytest test_cell_size.py -vv --noconftest --cov '
            '--cov-report html'
        )
    except subprocess.CalledProcessError as e:
        # clean up in case of an unexpected error cropping up
        remove_stragglers()
        raise e

    remove_stragglers()

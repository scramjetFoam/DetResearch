"""
Shock and Detonation Toolbox
"cv" module

Calculates constant-volume explosions.
 
This module defines the following functions:

    cvsolve
    
and the following classes:
    
    CVSys

################################################################################
Theory, numerical methods and applications are described in the following report:

    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    California Institute of Technology Revised September, 2018

Please cite this report and the website if you use these routines. 

Please refer to LICENCE.txt or the above report for copyright and disclaimers.

http://shepherd.caltech.edu/EDL/PublicResources/sdt/


################################################################################ 
Updated August 2018
Tested with: 
    Python 3.5 and 3.6, Cantera 2.3 and 2.4
Under these operating systems:
    Windows 8.1, Windows 10, Linux (Debian 9)
"""
from typing import Optional

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from sdtoolbox.config import Solver
from sdtoolbox.output import SimulationDatabase, ReactionData, SpeciesData, BulkPropertiesData


class CVSys(object):
    def __init__(
        self,
        gas: ct.Solution,
    ):
        self.gas = gas
        
    def __call__(self, t, y):
        """
        Evaluates the system of ordinary differential equations for an adiabatic, 
        constant-volume, zero-dimensional reactor. 
        It assumes that the 'gas' object represents a reacting ideal gas mixture.
    
        INPUT:
            t = time
            y = solution array [temperature, species mass 1, 2, ...]
            gas = working gas object
        
        OUTPUT:
            An array containing time derivatives of:
                temperature and species mass fractions, 
            formatted in a way that the integrator in cvsolve can recognize.
            
        """
        # Set the state of the gas, based on the current solution vector.
        self.gas.TDY = y[0], self.gas.density, y[1:]
        
        # Energy/temperature equation (terms separated for clarity)  
        a = self.gas.standard_enthalpies_RT - np.ones(self.gas.n_species)
        b = self.gas.net_production_rates / (self.gas.density * self.gas.cv_mass)
        dTdt = -self.gas.T * ct.gas_constant * np.dot(a, b)
        
        # Species equations
        dYdt = self.gas.net_production_rates*self.gas.molecular_weights/self.gas.density

        return np.hstack((dTdt, dYdt))


def cvsolve(
    gas: ct.Solution,
    t_end=1e-6,
    max_step=1e-5,
    t_eval=None,
    relTol=1e-5,
    absTol=1e-8,
    rxn_indices: Optional[list[int]] = None,
    spec_indices: Optional[list[int]] = None,
    db: Optional[SimulationDatabase] = None,
    run_no: int = 1,
    method: Solver = "Radau",
):
    """
    Solves the ODE system defined in CVSys, taking the gas object input as the
    initial state.
    
    Uses scipy.integrate.solve_ivp. The 'Radau' solver is used as this is a stiff system.
    From the scipy documentation:
        
        Implicit Runge-Kutta method of the Radau IIA family of order 5. 
        The error is controlled with a third-order accurate embedded formula. 
        A cubic polynomial which satisfies the collocation conditions 
        is used for the dense output.
        
    
    FUNCTION SYNTAX:
        output = cvsolve(gas,**kwargs)
    
    INPUT:
        gas = working gas object
        
    OPTIONAL INPUT:
        t_end = end time for integration, in sec
        max_step = maximum time step for integration, in sec
        t_eval = array of time values to evaluate the solution at.
                    If left as 'None', solver will select values.
                    Sometimes these may be too sparse for good-looking plots.
        relTol = relative tolerance
        absTol = absolute tolerances
    
    OUTPUT:
        output = a dictionary containing the following results:
            time = time array
            T = temperature profile array
            P = pressure profile array
            speciesY = species mass fraction array
            speciesX = species mole fraction array
            creation_rate = species creation rate array
            concentration = species concentration array
            
            gas = working gas object
            
            exo_time = pulse width (in secs) of temperature gradient (using 1/2 max)
            ind_time = time to maximum temperature gradient
            ind_len = distance to maximum temperature gradient
            ind_time_10 = time to 10% of maximum temperature gradient
            ind_time_90 = time to 90% of maximum temperature gradient
            
    """
    r0 = gas.density
    y0 = np.hstack((gas.T, gas.Y))

    tel = [0., t_end]  # Timespan

    output = {}

    # noinspection PyTypeChecker
    out: OdeResult = solve_ivp(
        CVSys(gas),
        tel,
        y0,
        method=method,
        atol=absTol,
        rtol=relTol,
        max_step=max_step,
        t_eval=t_eval,
    )

    output['time'] = out.t
    output['T'] = out.y[0, :]
    output['speciesY'] = out.y[1:, :]
    
    # Initialize additional output matrices where needed
    b = len(output['time'])
    output['P'] = np.zeros(b)
    output['speciesX'] = np.zeros(output['speciesY'].shape)
    output['ind_time'] = 0
    output['ind_time_90'] = 0
    output['ind_time_10'] = 0
    output['exo_time'] = 0    
    temp_grad = np.zeros(b)
    
    #############################################################################
    # Extract PRESSURE and TEMPERATURE GRADIENT
    #############################################################################
    
    # Have to loop for operations involving the working gas object
    for i, T in enumerate(output['T']):
        gas.TDY = T, r0, output['speciesY'][:, i]
        wt = gas.mean_molecular_weight        
        s = 0
        for z in range(gas.n_species):
            w = gas.molecular_weights[z]
            e = ct.gas_constant*T*(gas.standard_enthalpies_RT[z]/w - 1/wt)
            s = s + e*w*gas.net_production_rates[z]
            
        temp_grad[i] = -s/(r0*gas.cv_mass)
        output['P'][i] = gas.P
        output['speciesX'][:, i] = gas.X

        if db is not None:
            db.bulk_properties.insert_or_update(
                BulkPropertiesData(
                    condition_id=db.conditions_id,
                    run_no=run_no,
                    time=output["time"][i],
                    temperature=gas.T,
                    pressure=gas.P,
                ), commit=False)
            if spec_indices is not None:
                for idx_spec in spec_indices:
                    species = gas.species(idx_spec)
                    db.species.insert_or_update(SpeciesData(
                        condition_id=db.conditions_id,
                        run_no=run_no,
                        time=output["time"][i],
                        species=species,
                        mole_frac=gas.mole_fraction_dict().get(species.name, 0),
                        concentration=gas.concentrations[idx_spec],
                        creation_rate=gas.net_production_rates[idx_spec],
                        destruction_rate=gas.destruction_rates[idx_spec],
                        net_production_rate=gas.net_production_rates[idx_spec],
                    ), commit=False)
            if rxn_indices is not None:
                for idx_rxn in rxn_indices:
                    db.reactions.insert_or_update(ReactionData(
                        condition_id=db.conditions_id,
                        run_no=run_no,
                        time=output["time"][i],
                        reaction=gas.reaction_equation(idx_rxn),
                        fwd_rate_constant=gas.forward_rate_constants[idx_rxn],
                        fwd_rate_of_progress=gas.forward_rates_of_progress[idx_rxn],
                        rev_rate_constant=gas.reverse_rate_constants[idx_rxn],
                        rev_rate_of_progress=gas.reverse_rates_of_progress[idx_rxn],
                        net_rate_of_progress=gas.net_rates_of_progress[idx_rxn],
                    ), commit=False)

    if db is not None:
        db.bulk_properties.cur.connection.commit()
        db.species.cur.connection.commit()
        db.reactions.cur.connection.commit()

    n = temp_grad.argmax()

    if n == b:
        raise ValueError(
            'Error: Maximum temperature gradient occurs at the end of the reaction zone. '
            'Your final integration length may be too short, '
            'your mixture may be too rich/lean, or something else may be wrong'
        )
        # output['ind_time'] = output['time'][b]
        # output['ind_time_10'] = output['time'][b]
        # output['ind_time_90'] = output['time'][b]
        # output['exo_time'] = 0
        # print('Induction Time: '+str(output['ind_time']))
        # print('Exothermic Pulse Time: '+str(output['exo_time']))
        # return output
    elif n == 0:
        raise ValueError(
            'Error: Maximum temperature gradient occurs at the beginning of the reaction zone '
            'Your final integration length may be too short, '
            'your mixture may be too rich/lean, or something else may be wrong'
        )
        # output['ind_time'] = output['time'][0]
        # output['ind_time_10'] = output['time'][0]
        # output['ind_time_90'] = output['time'][0]
        # output['exo_time'] = 0
        # print('Induction Time: '+str(output['ind_time']))
        # print('Exothermic Pulse Time: '+str(output['exo_time']))
        # return output
    else:
        output['ind_time'] = output['time'][n]
        
        k = 0
        MAX10 = 0.1*max(temp_grad)
        d = temp_grad[0]        
        while d < MAX10:
            k = k + 1
            d = temp_grad[k]
        output['ind_time_10'] = output['time'][k]
        
        k = 0
        MAX90 = 0.9*max(temp_grad)
        d = temp_grad[0]
        while d < MAX90:
            k = k + 1
            d = temp_grad[k]
        output['ind_time_90'] = output['time'][k]

        # find exothermic time
        half_T_flag1 = 0
        half_T_flag2 = 0
        tstep1 = 0
        tstep2 = 0
        # Go into a loop to find two times when temperature is half its maximum
        for j,tgrad in enumerate(list(temp_grad)):
            if half_T_flag1 == 0:
                if tgrad > 0.5*max(temp_grad):
                    half_T_flag1 = 1
                    tstep1 = j
                    
            elif half_T_flag2 == 0:
                if tgrad < 0.5*max(temp_grad):
                    half_T_flag2 = 1
                    tstep2 = j
                else:
                    tstep2 = 0

        # Exothermic time for CV explosion
        if tstep2 == 0:
            raise ValueError(
                'Error: No pulse in the temperature gradient '
                'Your final integration length may be too short, '
                'your mixture may be too rich/lean, or something else may be wrong'
            )
            # output['exo_time'] = 0
        else:
            output['exo_time'] = output['time'][tstep2] - output['time'][tstep1]

    output['gas'] = gas 
    return output

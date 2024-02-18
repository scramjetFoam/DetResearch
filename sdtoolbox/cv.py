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
import sqlite3
from typing import Optional

import cantera as ct
import numpy as np
from retry import retry
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from sdtoolbox.output import SimulationDatabase, ReactionData, SpeciesData, BulkPropertiesData


class CVSys(object):
    def __init__(
        self,
        gas: ct.Solution,
        rxn_indices: Optional[list[int]] = None,
        spec_indices: Optional[list[int]] = None,
        db: Optional[SimulationDatabase] = None,
        batch_threshold: int = 10_000,
        run_no: int = 1,
    ):
        self.gas = gas
        self.rxn_indices = rxn_indices
        self.spec_indices = spec_indices
        self.db = db
        self.rxn_to_store: list[ReactionData] = []
        self.spec_to_store: list[SpeciesData] = []
        self.bulk_properties_to_store: list[BulkPropertiesData] = []
        self.batch_threshold = batch_threshold
        self.run_no = run_no
        self.can_store_reaction_data = None not in (self.rxn_indices, self.db) and len(self.rxn_indices) > 0
        self.can_store_species_data = None not in (self.spec_indices, self.db) and len(self.spec_indices) > 0
        self.can_store_bulk_properties_data = self.db is not None

    def __del__(self):
        if self.can_store_reaction_data and len(self.rxn_to_store) > 0:
            self.store_all_rxn_data()

        if self.can_store_species_data and len(self.spec_to_store) > 0:
            self.store_all_spec_data()
            self.store_all_rxn_data()

        if self.can_store_bulk_properties_data and len(self.bulk_properties_to_store) > 0:
            self.store_all_bulk_properties_data()

        if self.db is not None:
            try:
                self.db.reactions.cur.connection.close()
            except:
                pass
            try:
                self.db.species.cur.connection.close()
            except:
                pass
            try:
                self.db.conditions.cur.close()
            except:
                pass
            try:
                self.db.bulk_properties.cur.close()
            except:
                pass

    @retry(tries=10, backoff=2, max_delay=2)
    def store_all_rxn_data(self):
        for data in self.rxn_to_store:
            try:
                self.db.reactions.insert_or_update(data, commit=False)
            except sqlite3.ProgrammingError:
                self.db.reconnect()
                self.db.reactions.insert_or_update(data, commit=False)
        try:
            self.db.reactions.cur.connection.commit()
        except sqlite3.ProgrammingError:
            self.db.reconnect()
            self.db.reactions.cur.connection.commit()
        self.rxn_to_store.clear()

    @retry(tries=10, backoff=2, max_delay=2)
    def store_all_spec_data(self):
        for data in self.spec_to_store:
            try:
                self.db.species.insert_or_update(data, commit=False)
            except sqlite3.ProgrammingError:
                self.db.reconnect()
                self.db.species.insert_or_update(data, commit=False)
        try:
            self.db.species.cur.connection.commit()
        except sqlite3.ProgrammingError:
            self.db.reconnect()
            self.db.species.cur.connection.commit()
        self.spec_to_store.clear()

    @retry(tries=10, backoff=2, max_delay=2)
    def store_all_bulk_properties_data(self):
        for data in self.bulk_properties_to_store:
            try:
                self.db.bulk_properties.insert_or_update(data, commit=False)
            except sqlite3.ProgrammingError:
                self.db.reconnect()
                self.db.bulk_properties.insert_or_update(data, commit=False)
        try:
            self.db.bulk_properties.cur.connection.commit()
        except sqlite3.ProgrammingError:
            self.db.reconnect()
            self.db.bulk_properties.cur.connection.commit()
        self.bulk_properties_to_store.clear()
        
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

        if self.can_store_reaction_data:
            for i in self.rxn_indices:
                data = ReactionData(
                    condition_id=self.db.conditions_id,
                    run_no=self.run_no,
                    time=t,
                    reaction=self.gas.reaction_equation(i),
                    fwd_rate_constant=self.gas.forward_rate_constants[i],
                    fwd_rate_of_progress=self.gas.forward_rates_of_progress[i],
                )
                self.rxn_to_store.append(data)

            if len(self.rxn_to_store) >= self.batch_threshold:
                self.store_all_rxn_data()

        if self.can_store_species_data:
            for j in self.spec_indices:
                species = self.gas.species(j)
                data = SpeciesData(
                    condition_id=self.db.conditions_id,
                    run_no=self.run_no,
                    time=t,
                    species=species,
                    mole_frac=self.gas.mole_fraction_dict().get(species.name, 0),
                    concentration=self.gas.concentrations[j],
                    creation_rate=self.gas.creation_rates[j],
                )
                self.spec_to_store.append(data)

            if len(self.spec_to_store) >= self.batch_threshold:
                self.store_all_spec_data()

        if self.can_store_bulk_properties_data:
            data = BulkPropertiesData(
                condition_id=self.db.conditions_id,
                run_no=self.run_no,
                time=t,
                temperature=self.gas.T,
                pressure=self.gas.P,
            )
            self.bulk_properties_to_store.append(data)

            if len(self.bulk_properties_to_store) >= self.batch_threshold:
                self.store_all_bulk_properties_data()
        
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
    batch_threshold: int = 10_000,
    run_no: int = 1,
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
        CVSys(gas, rxn_indices, spec_indices, db, batch_threshold, run_no),
        tel,
        y0,
        method='Radau',
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
    # Extract PRESSSURE and TEMPERATURE GRADIENT
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

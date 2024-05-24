# Charger assignment as a mixed integer program
# Created: 2023-10-14

# Imports
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib as plt
# from scipy import optimize ... to clunky for this application (requires DVs in one row), see scipy.optimize.milp
import pulp

# Settings
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 40
pd.options.display.max_columns = 80

# Directory for outputs
DIR = os.path.expanduser('~') + '/Desktop/Python/optimization/charger-assignment/'

# Random Number Generator
RNG = np.random.default_rng(2)

# Constants
MAX_VEH_TO_CHARGER_OPTIONS = [2, 3, 5]
NUM_VEHICLES = RNG.integers(0, 200)
CHARGING_HOURS = 12
SWAP_BUFFER = 0.5
L2_KW = 11.5
L3_KW = 50
L2_EFFICIENCY = 0.97
L3_EFFICIENCY = 0.99
L3_FRAC = 0.1
TODAY = datetime.date.today().isoformat()


# Optimization functions
def pulp_minimize_unassigned_kwh(vehicles_df, chargers_df, veh_to_charger):
    """
    Mixed-integer program that:
        - Minimizes unsatisfied charging needs
        - Assigns vehicles to chargers
        - Constrains:
            - Each charger to no more than the specified number of vehicles
            - Each vehicle to one charger

    :param vehicles_df: pd.DataFrame with vehicles and charging needs
    :param chargers_df: pd.DataFrame with chargers, charging efficiency, kW output, and energy capacity
    :param veh_to_charger: int specifying max number of vehicles per charger
    :return:
    """

    vehicles_df = vehicles_df.copy()
    chargers_df = chargers_df.copy()

    # Set ids as strings
    vehicles_df['vehicle_id'] = vehicles_df['vehicle_id'].astype(str)
    chargers_df['charger_id'] = chargers_df['charger_id'].astype(str)

    # Set up problem as minimization
    prob = pulp.LpProblem("Optimal_Vehicle_to_Charger_Assignments", pulp.LpMinimize)

    # Define unsatisfied kWh
    unsatisfied_kwh_vars = pulp.LpVariable.dicts("Unsatisfied kWh by Charger", chargers_df['charger_id'],
                                                 lowBound=0, upBound=None, cat="Continuous")

    # Minimize unsatisfied kWh (define objective function)
    prob += (pulp.lpSum(unsatisfied_kwh_vars), "Sum of Unsatisfied kWh")

    # Decided not to constrain/add upper bounds for unsatisfied kwh, since the problem is a minimization

    # Define vehicle-charger assignments

    # Create vehicle energy needs dictionary
    veh_energy_dict = pd.Series(vehicles_df['trip_energy_kwh'])
    veh_energy_dict.index = vehicles_df['vehicle_id']
    veh_energy_dict = veh_energy_dict.to_dict()
    # Create charger energy dictionary
    charger_energy_dict = pd.Series(chargers_df['kwh'])
    charger_energy_dict.index = chargers_df['charger_id']
    charger_energy_dict = charger_energy_dict.to_dict()

    # Constrain assignments to vehicle-to-charger ratio
    assignment_vars = pulp.LpVariable.dicts(
        "Assignments", (charger_energy_dict, veh_energy_dict), cat="Binary")
    for c in charger_energy_dict:
        prob += (pulp.lpSum([assignment_vars[c][v] for v in veh_energy_dict]) <= veh_to_charger,
                 f"Sum_of_Vehicles_Assigned_to_Charger_{c}")

    # Restrict vehicles to one charger
    for v in veh_energy_dict:
        prob += (pulp.lpSum([assignment_vars[c][v] for c in charger_energy_dict]) == 1,
                 f"Sum_of_Vehicle_Assignments_{v}")

    # Balance charger energy
    for c in charger_energy_dict:
        kwh_lost_per_swap = (SWAP_BUFFER * chargers_df.loc[chargers_df['charger_id'] == c, 'efficiency'] *
                             chargers_df.loc[chargers_df['charger_id'] == c, 'kw']).squeeze()
        # Vehicle energy needs must be satisfied by charger or count as unsatisfied energy
        # For every vehicle-swap, there is a half-hour penalty (does not apply to the charger's first vehicle)
        prob += ((pulp.lpDot([assignment_vars[c][v] for v in veh_energy_dict],
                             [kwh_need for kwh_need in veh_energy_dict.values()]) +
                  (pulp.lpSum([assignment_vars[c][v] for v in veh_energy_dict]) - 1) * kwh_lost_per_swap) -
                 unsatisfied_kwh_vars[c] <= charger_energy_dict[c],
                 f"Charger_{c}_Satisfies_Energy_Constraint")

    # Write formulation to file
    prob.writeLP(DIR + 'pulp/' + f'{TODAY} Charger Assignment Optimization - {len(vehicles_df)}  Vehicles '
                                 f'{len(chargers_df)} Chargers {veh_to_charger} vehicles-to-chargers.lp')

    # The problem is solved using PuLP's choice of Solver
    prob.solve()
    print("Status:", pulp.LpStatus[prob.status])

    # Extract vehicle-charger assignments
    # Add placeholder id columns in vehicles_df and chargers_df
    vehicles_df['charger_id'] = None
    chargers_df['unsatisfied_kwh'] = np.NaN
    # Create charger/vehicle matrix
    charging_matrix = pd.DataFrame(np.zeros((len(chargers_df), len(vehicles_df))))
    charging_matrix.columns = vehicles_df['vehicle_id']
    charging_matrix.index = chargers_df['charger_id']
    # Create placeholder DataFrame for vehicle energy by charger
    charger_entries_df = pd.DataFrame()
    for v in prob.variables():
        if "Assignments" in str(v):
            assignment = str(v).split('_')
            c_id = assignment[1]
            v_id = assignment[2]
            charging_matrix.loc[c_id, v_id] = v.varValue
            # Note down charger_id for vehicle
            if v.varValue:
                vehicles_df.loc[vehicles_df['vehicle_id'] == v_id, 'charger_id'] = c_id
                charger_entry = pd.DataFrame(
                    {'charger_id': c_id,
                     'vehicle_id': v_id,
                     'kwh': vehicles_df.loc[vehicles_df['vehicle_id'] == v_id, 'trip_energy_kwh'].sum()}, index=[0])
                charger_entries_df = pd.concat([charger_entries_df, charger_entry], ignore_index=True)

        # Record unsatisfied energy by charger
        elif "Unsatisfied" in str(v):
            unsatisfied_index = str(v).split('Unsatisfied_kWh_by_Charger_')[1]
            chargers_df.loc[chargers_df['charger_id'] == unsatisfied_index, 'unsatisfied_kwh'] = v.varValue
            charger_entry = pd.DataFrame(
                {'charger_id': unsatisfied_index, 'vehicle_id': 'Unsatisfied kWh', 'kwh': v.varValue}, index=[0])
            charger_entries_df = pd.concat([charger_entries_df, charger_entry], ignore_index=True)

    # Record number of vehicles for each charger
    chargers_df['num_veh'] = pd.Series(charging_matrix.sum(axis=1)).reset_index(drop=True)

    # Check for veh-to-charger ratio compliance
    assert (charging_matrix.sum(axis=1) <= veh_to_charger).all(), "Veh-to-charger ratio not met"
    # Check that all vehicles have been assigned to chargers
    assert (charging_matrix.sum(axis=0) == 1).all(), \
        "Vehicles have not been assigned properly. Each vehicle should be assigned to exactly one charger."
    assert vehicles_df['charger_id'].notna().all(), "Vehicle charger assignments have not been recorded properly"

    # Print total of unsatisfied vehicle energy requirements
    print("Total Cost of Transportation = ", pulp.value(prob.objective))

    output_dict = {'vehicles': vehicles_df, 'chargers': chargers_df, 'charger_vehicle_kwh': charger_entries_df}

    return output_dict


def cvxpy_minimize_unassigned_kwh(veh_df, chargers_df):

    vehicles_df = veh_df.copy()
    chargers_df = veh_df.copy()

    # Todo implement optimization with cvxpy

    return vehicles_df, chargers_df


# Define vehicle energy requirements
veh_energy_df = pd.DataFrame(RNG.exponential(scale=30, size=NUM_VEHICLES)).reset_index()
veh_energy_df.columns = ['vehicle_id', 'trip_energy_kwh']
veh_energy_df['trip_energy_kwh'].hist(bins=30)

# Iterate through scenarios
pulp_scenario_dict = {}
for i in MAX_VEH_TO_CHARGER_OPTIONS:

    # Define charger set
    num_chargers = np.ceil(NUM_VEHICLES / i)
    num_l3 = np.ceil(num_chargers * L3_FRAC)
    num_l2 = num_chargers - num_l3
    # Create DataFrame
    charger_df = pd.DataFrame(np.repeat(pd.DataFrame([['level_3', L3_KW]]), num_l3, axis=0))
    charger_df = pd.concat(
        [charger_df, pd.DataFrame(np.repeat(pd.DataFrame([['level_2', L2_KW]]), num_l2, axis=0))],
        ignore_index=True).reset_index()
    charger_df.columns = ['charger_id', 'level', 'kw']
    # Assign charger efficiency
    charger_df['efficiency'] = np.where(charger_df['level'] == 'level_3', L3_EFFICIENCY, L2_EFFICIENCY)
    # Add charger energy capacity
    charger_df['kwh'] = charger_df['kw'] * charger_df['efficiency'] * CHARGING_HOURS

    # Call optimization function, store outputs
    pulp_outputs_dict = pulp_minimize_unassigned_kwh(veh_energy_df, charger_df, i)
    pulp_scenario_dict.update({i: pulp_outputs_dict})


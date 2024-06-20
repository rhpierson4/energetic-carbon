# Simulation of EV load for a generic facility in Tracy, CA

# Initialize workspace
import os
from weibull_trips import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Constants
FLEET_MAKEUP = {'Vans': {'make': 'Ford', 'model': 'eTransit',
                         'battery_kwh': 89, 'range_mi': 159, 'l2_kw': 11.5,
                         'avg_mi': 60, 'max_mi': 130, 'op_days': 220, 'count': 20},
                'Pickups': {'make': 'Ford', 'model': 'F-150 Lighting XLT',
                            'battery_kwh': 131, 'range_mi': 320, 'l2_kw': 19.2,
                            'avg_mi': 80, 'max_mi': 280, 'op_days': 220, 'count': 40}}


# Classes
class ElectricVehicle(Vehicle):
    """Instance of an Electric Vehicle based on Vehicle"""
    def __init__(self, make, model, battery_kwh, range_mi, l2_kw, avg_mi, max_mi, op_days):
        super().__init__(make, model, avg_mi, max_mi, op_days)
        self.battery_kwh = battery_kwh
        self.range_mi = range_mi
        self.l2_kw = l2_kw


class Fleet:
    """Fleet of vehicle class objects"""

    def __init__(self):
        self.vehicles = {}
        self.fleet_size = 0
        self.trip_df = pd.DataFrame

    def add_vehicle(self, vehicle):
        self.vehicles[self.fleet_size + 1] = vehicle
        self.fleet_size += 1

    def aggregate_trips(self):
        self.trip_df = pd.concat([veh.trip_df for veh in self.vehicles.values()]).reset_index(drop=True)


# Build fleet
tracy_fleet = Fleet()
for veh_type in FLEET_MAKEUP.keys():
    veh_type_dict = FLEET_MAKEUP[veh_type]
    veh_count = veh_type_dict['count']
    for i in range(0, veh_count):
        new_veh = ElectricVehicle(make=veh_type_dict['make'],
                                  model=veh_type_dict['model'],
                                  battery_kwh=veh_type_dict['battery_kwh'],
                                  range_mi=veh_type_dict['range_mi'],
                                  l2_kw=veh_type_dict['l2_kw'],
                                  avg_mi=veh_type_dict['avg_mi'],
                                  max_mi=veh_type_dict['max_mi'],
                                  op_days=veh_type_dict['op_days'])
        new_veh.add_trips(start_date=pd.Timestamp('2023-01-01'), end_date=pd.Timestamp('2023-12-31'))
        new_veh.trip_df['Vehicle Type'] = veh_type
        tracy_fleet.add_vehicle(new_veh)

# Aggregate Trips
tracy_fleet.aggregate_trips()
tracy_fleet.trip_df

# QC Average Mileage
tracy_fleet.trip_df.query('miles > 0').groupby('Vehicle Type').agg('mean')

# Perform T-test
for veh_type in FLEET_MAKEUP.keys():
    t_statistic, p_value = scipy.stats.ttest_1samp(
        tracy_fleet.trip_df.query('(`Vehicle Type` == @veh_type) & (miles > 0)')['miles'],
        FLEET_MAKEUP[veh_type]['avg_mi'])
    print(f'{veh_type} p-value: {p_value}')


# Plot
sns.set(rc={'figure.figsize': (6, 3)})
sns.kdeplot(tracy_fleet.trip_df.query('miles > 0'), x='miles', hue='Vehicle Type', cut=0)
plt.savefig(os.path.expanduser('~') + '/Downloads/tracy_veh_type_kde_miles.png')


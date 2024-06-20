# Simulation of EV load for a generic facility in Tracy, CA

# Initialize workspace
import os
from weibull_trips import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Pandas view settings
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 80
pd.options.display.max_columns = 80


# Constants
FLEET_MAKEUP = {'Vans': {'make': 'Ford', 'model': 'eTransit',
                         'battery_kwh': 89, 'range_mi': 159, 'l2_kw': 11.5,
                         'avg_mi': 60, 'max_mi': 130, 'op_days': 220, 'count': 20},
                'Pickups': {'make': 'Ford', 'model': 'F-150 Lighting XLT',
                            'battery_kwh': 131, 'range_mi': 320, 'l2_kw': 19.2,
                            'avg_mi': 80, 'max_mi': 280, 'op_days': 220, 'count': 40}}

# PG&E BEV-2-S
PEAK_USD_PER_KWH = 0.41522
OFF_PEAK_USD_PER_KWH = 0.20199
SUPER_OFF_PEAK_USD_PER_KWH = 0.17872
SUBSCRIPTION_KW_PER_BLOCK = 50
SUBSCRIPTION_USD_PER_BLOCK = 95.56
SUPER_OFF_PEAK_DURATION_HRS = 5

# Charging constants
L2_EFFICIENCY = 0.95


# Classes
class ElectricVehicle(Vehicle):
    """Instance of an Electric Vehicle based on Vehicle"""
    def __init__(self, make, model, battery_kwh, range_mi, l2_kw, avg_mi, max_mi, op_days):
        super().__init__(make, model, avg_mi, max_mi, op_days)
        self.battery_kwh = battery_kwh
        self.range_mi = range_mi
        self.kwh_per_mi = battery_kwh/range_mi
        self.l2_kw = l2_kw

    def calculate_kwh_consumed(self):
        self.trip_df['kwh_consumed'] = self.trip_df['miles'] * self.kwh_per_mi

    def get_return_hr(self):
        """Assume all trips are centered around 12:30pm"""
        self.trip_df['return_hr'] = 12.5 + self.trip_df['hrs']/2

    def calculate_charging_hrs(self):
        """Assume vehicles need to be fully charged by 8am every day, or 4pm if there are no trips that day"""
        # todo: Remove hardcoded references to hours
        # assert 'return_hour' in self.trip_df.columns, "Run self.get_return_hr() before calculating charging hours"
        self.get_return_hr()

        # Set charge by time based on whether there are trips the next day or not
        next_trip_mi = self.trip_df['miles'].shift(-1).fillna(1)
        self.trip_df['charge_by_hr'] = np.where(next_trip_mi > 0, 8, 16)

        # Calculate charging hours
        self.trip_df['charging_hrs'] = self.trip_df['charge_by_hr'] + (24 - self.trip_df['return_hr'])

    def calculate_unmanaged_charging(self):
        """Calculate kWh cost and kW for each trip using an unmanaged charging approach"""

        # Assume charging at full speed
        self.trip_df['unmanaged_kw'] = self.l2_kw

        # Assign charging hours to TOU period
        hrs_df = self.trip_df.copy()
        hrs_df['unmanaged_hrs'] = (hrs_df['kwh_consumed'] / L2_EFFICIENCY) / self.l2_kw

        # Calculate pre-peak charging hours (off-peak)
        hrs_df['pre_on_peak_hrs'] = hrs_df['return_hr'].apply(lambda x: max(0, 16 - x))
        hrs_df['pre_on_peak_hrs'] = np.where(hrs_df['unmanaged_hrs'] < hrs_df['pre_on_peak_hrs'],
                                             hrs_df['unmanaged_hrs'], hrs_df['pre_on_peak_hrs'])
        hrs_df['unclaimed_hrs'] = hrs_df['unmanaged_hrs'] - hrs_df['pre_on_peak_hrs']

        # Calculate on-peak charging hours
        hrs_df['on_peak_hrs'] = hrs_df['return_hr'].apply(lambda x: max(0, 21 - max(x, 16)))
        hrs_df['on_peak_hrs'] = np.where(hrs_df['unclaimed_hrs'] > 0,
                                         hrs_df[['on_peak_hrs', 'unclaimed_hrs']].min(axis=1), 0)
        hrs_df['unclaimed_hrs'] -= hrs_df['on_peak_hrs']

        # Calculate charging cost assuming any unclaimed_hrs are off-peak
        self.trip_df['unmanaged_kwh_cost'] = hrs_df['unmanaged_kw'] * (OFF_PEAK_USD_PER_KWH * (
                    hrs_df['pre_on_peak_hrs'] + hrs_df['unclaimed_hrs']) + PEAK_USD_PER_KWH * hrs_df['on_peak_hrs'])

    def calculate_scheduled_charging(self):
        """Calculate kWh cost and kW for each trip using a scheduled charging approach"""

        # Assume off-peak charging at full speed
        self.trip_df['scheduled_kw'] = self.l2_kw
        self.trip_df['scheduled_kwh_cost'] = (self.trip_df['kwh_consumed'] / L2_EFFICIENCY) * OFF_PEAK_USD_PER_KWH

    def calculate_managed_charging(self):
        """Calculate kWh cost and kW for each trip using a managed charging approach"""

        # Assume off-peak charging at minimum speed after 9pm
        managed_hrs = (24 - self.trip_df['return_hr'].clip(lower=21)) + self.trip_df['charge_by_hr']
        self.trip_df['managed_kw'] = (self.trip_df['kwh_consumed'] / L2_EFFICIENCY) / managed_hrs
        self.trip_df['managed_kwh_cost'] = (
            np.where(self.trip_df['charge_by_hr'] == 16,
                     self.trip_df['managed_kw'] * ((SUPER_OFF_PEAK_DURATION_HRS * SUPER_OFF_PEAK_USD_PER_KWH) +
                                                   (managed_hrs - SUPER_OFF_PEAK_DURATION_HRS) * OFF_PEAK_USD_PER_KWH),
                     self.trip_df['managed_kw'] * managed_hrs * OFF_PEAK_USD_PER_KWH))


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
        new_veh.calculate_kwh_consumed()
        new_veh.calculate_charging_hrs()
        new_veh.calculate_unmanaged_charging()
        new_veh.calculate_scheduled_charging()
        new_veh.calculate_managed_charging()
        new_veh.trip_df['Vehicle Type'] = veh_type
        new_veh.trip_df['vehicle_id'] = veh_type + str(i)
        tracy_fleet.add_vehicle(new_veh)

# Aggregate Trips
tracy_fleet.aggregate_trips()
tracy_fleet.trip_df

# QC Average Mileage
tracy_fleet.trip_df.query('miles > 0').groupby('Vehicle Type')[['miles', 'hrs', 'date']].agg('mean')

# Perform T-test on mileage by vehicle type
for veh_type in FLEET_MAKEUP.keys():
    t_statistic, p_value = scipy.stats.ttest_1samp(
        tracy_fleet.trip_df.query('(`Vehicle Type` == @veh_type) & (miles > 0)')['miles'],
        FLEET_MAKEUP[veh_type]['avg_mi'])
    print(f'{veh_type} p-value: {p_value}')

# Plot mileage
sns.set(rc={'figure.figsize': (6, 3)})
sns.kdeplot(tracy_fleet.trip_df.query('miles > 0'), x='miles', hue='Vehicle Type', cut=0)
plt.savefig(os.path.expanduser('~') + '/Downloads/tracy_veh_type_kde_miles.png')

# Output charging metrics
# todo: fix number formatting
print(f'Unmanaged Charging: ${tracy_fleet.trip_df['unmanaged_kwh_cost'].sum()}, '
      f'{tracy_fleet.trip_df.groupby('date')['unmanaged_kw'].sum().max()} kW')
print(f'Scheduled Charging: ${tracy_fleet.trip_df['scheduled_kwh_cost'].sum()}, '
      f'{tracy_fleet.trip_df.groupby('date')['scheduled_kw'].sum().max()} kW')
print(f'Managed Charging: ${tracy_fleet.trip_df['managed_kwh_cost'].sum()}, '
      f'{tracy_fleet.trip_df.groupby('date')['managed_kw'].sum().max()} kW')


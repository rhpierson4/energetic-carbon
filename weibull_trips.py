# Generate synthetic trip data for vehicles

# Load packages
import pandas as pd
import numpy as np
from scipy.stats import weibull_min

# Random Number Generator
rng = np.random.default_rng(102)


# Define vehicle class
class Vehicle:

    def __init__(self, make, model, avg_mi, max_mi, op_days):
        """Initializes the Vehicle instance with average trip mileage, max trip mileage, and annual operating days"""
        self.make = make
        self.model = model
        self.avg_mi = avg_mi
        self.max_mi = max_mi
        self.op_days = op_days
        self.avg_op_hours = 8  # placeholder
        self.op_hour_sd = 1.25  # placeholder
        self.trip_df = pd.DataFrame()

    def _make_weibull_trips(self, num_days):
        """Return Series of Weibull-distributed trips for specified avg_mi, max_mi, and op_days"""

        # Generate trip lengths from weibull distribution
        c = self.max_mi / self.avg_mi  # Define shape parameter as the ratio of max mileage over avg mileage
        c = (c + 1)/2  # Bring shape parameter closer to 1
        trips = pd.Series(weibull_min.rvs(c, size=num_days, random_state=rng.integers(0, 2**31)))
        # Perform mean scaling
        trips = trips * (self.avg_mi / np.mean(trips))

        # Set some trips to zero to reflect annual operating days
        # Note: Assumes vehicles operate Mon-Sun
        trip_freq = self.op_days / 365
        op_binary = pd.Series(rng.binomial(1, trip_freq, num_days))
        trips *= op_binary

        # Cap trip mileage
        trips = trips.clip(0, self.max_mi)

        return trips

    def _calculate_trip_duration(self, num_days):
        """Return Series of normally-distributed trip durations based on operating hour parameters"""

        # Generate durations
        hours = pd.Series(rng.normal(self.avg_op_hours, self.op_hour_sd, num_days))

        # Set upper and lower bounds
        lower_bound = np.maximum(0, self.avg_op_hours - self.op_hour_sd * 3)
        upper_bound = np.minimum(self.avg_op_hours + self.op_hour_sd * 3, 18)
        hours = hours.clip(lower_bound, upper_bound)

        return hours

    def add_trips(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Add trips for the specified timeframe to the vehicle's trip DataFrame"""

        df = self.trip_df.copy()
        td_days = (end_date - start_date).days + 1

        # If trips are already present, remove any existing trips within the specified timeframe
        if not df.empty:
            df.query('(trip_date < @start_date) & (trip_date > @end_date)', inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Make new trips
        new_trips = self._make_weibull_trips(td_days)
        new_trip_df = pd.DataFrame({'miles': new_trips})

        # Add trip duration
        trip_durations = self._calculate_trip_duration(td_days)
        new_trip_df['hrs'] = trip_durations

        # Clean trip duration for zero-mileage trips
        new_trip_df.loc[new_trip_df['miles'] == 0, 'hrs'] = 0

        # Add dates
        new_trip_df['date'] = pd.date_range(start_date, end_date)

        # Combine new trips with old trips
        self.trip_df = pd.concat([df, new_trip_df]).sort_values('date').reset_index(drop=True)


"""
## Test ##
test_vehicle = Vehicle(make='Ford', 'F-150 Lightning Pro', avg_mi=50, max_mi=120, op_days=220)
test_vehicle.add_trips(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
"""

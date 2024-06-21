# Workspace for electrification and sustainability-related code

## Project 1: EVSE assignment optimization

Electric Vehicle (EV) economics improve when Electric Vehicle Supply Equipment (EVSE or chargers) is 
shared among vehicles. This project optimizes the assignment of EVs to chargers when there are more EVs 
than chargers. The problem is configured as a mixed-integer program (MIP). The main assumption is a fixed 
number of dwell hours for each charger.

The initial implementation optimizes using PuLP and an implementation with cvxpy may be added later.

File: charger_assignment_mip.py

## Project 2: Simulation of managed charging savings

Managed charging offers cost savings compared to unmanaged and scheduled charging. This project constructs 
a basic managed charging model as well as comparison models for unmanaged and scheduled charging. A 1-to-1 
vehicle-to-charger ratio is assumed. These models are run on simulated trip data using the Weibull 
distribution for mileage and the normal distribution for duration. Here, mileage and duration are assumed 
to be independently distributed. The project is set up to analyze PG&E's BEV-2-S rate as of June 2024. The 
results indicate that unmanaged charging would be twice as expensive as managed charging, but scheduled 
charging would only be 20% more expensive. Managed charging has substantially lower subscription (kW) costs 
compared to both, but similar energy (kWh) costs to scheduled charging.

The code also experiments with classes and subclasses, a secondary objective for the project. 

Files: tracy_site_modeling.py, weibull_trips.py



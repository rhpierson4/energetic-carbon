# Workspace for electrification and sustainability-related code

## Project 1: EVSE assignment optimization

Electric Vehicle (EV) economics improve when Electric Vehicle Supply Equipment (EVSE or chargers) is 
shared among vehicles. This project optimizes the assignment of EVs to chargers when there are more EVs 
than chargers. The problem is configured as a mixed-integer program (MIP). The main assumption is a fixed 
number of dwell hours for each charger.

The initial implementation optimizes using PuLP and an implementation with cvxpy may be added later.

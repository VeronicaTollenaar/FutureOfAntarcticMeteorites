Overview of scripts:

prepare_altered_temp.py - reads in climate model output and calculates temperature anomalies, prepares altered temperature files for all years and all locations, generates Figure S7
Classify_observations_TempUp.py - defines function that classifies locations as meteorite stranding zone or not, uses altered temperature files as input
predict_meteorites.py - loops over all years to predict the meteorite locations for each year given the altered temperatures
time_vs_temp.py - calculates the temperature evolution over time and relates meteorite losses directly to temperature increases, generates Figure S2, and Figures S4-S6
map_yr_lost_rev.py - analyzes meteorite losses over time, generates Figure 2, Figure S1, and Figure S3
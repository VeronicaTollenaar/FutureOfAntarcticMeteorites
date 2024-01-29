# loop over all years to predict meteorite stranding zones per year
# import packages
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd
from Classify_observations_TempUp import classifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)


#%%
# define climate model run
MAR_run = 'CESM2_ssp126' #'CESM2_ssp585'
# defin years
years = np.arange(2001,2082,1)
# loop over years
for year in years:
    print(year)
    # define savename for model outpu
    savename = f'{MAR_run}_{year}'

    # read in altered temperatures for all locations
    altered_stemp_mets = f'../data/{MAR_run}_{year}_mets.csv'
    altered_stemp_toclass = f'../data/{MAR_run}_{year}_toclass.csv'
    
    # predict meteorite stranding zones and save data (see Classify_observations_TempUp.py)
    data_classified = classifier(savename,altered_stemp_mets,altered_stemp_toclass)

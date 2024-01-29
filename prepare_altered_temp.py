# calculate temperature anomalies with respect to 2020
# prepare Figure S7
# import packages
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd
from scipy.stats import pearsonr

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)


#%%
# import MAR data
# define data directory
data_dir = '../data'
# define run of climate model MAR
MAR_run = 'CESM2_ssp585' #'CESM2_ssp126'#
var_name = 'ST2'

# Average over how many days?
MODIS_day_average=8
# Percentile:
MODIS_percentile=99
# Total time period (in days)
MODIS_total_days=6944 # From 1 January 2001 to 1 January 2020 = 6939 + 5 days to be divisble by 8: easier for later on (so approx. 19 years)

# open all .nc files in folder
data_raw = xr.open_mfdataset(f'{data_dir}/{MAR_run}/{var_name}_*.nc')
# select only surface temperature from files
surface_temperature=data_raw[f'{var_name}'][:]

# convert to datetimeindex (if not already datetimeindex)
if str(surface_temperature.TIME.values.dtype) == 'object':
    datetimeindex = surface_temperature.indexes['TIME'].to_datetimeindex()
    surface_temperature['TIME'] = datetimeindex

# average 8 days (synchronized with MODIS obs) (start date of used MODIS data is 2001/01/01, end date is 2019/12/27), every date represents the first day of the 8-day period
# MAR data runs from 1980/01/01 to 2100/12/31

# find first day of 8-day period of MAR data (so that the start date is aligned with the MODIS data)
first_modis_date = datetime.date(2001,1,1)
last_modis_date = datetime.date(2020,1,1)
first_MAR_date = datetime.date(1981,1,1) # (1980,1,1)
last_MAR_date = datetime.date(2100,12,31) # (2100,12,31)
n_8day_cycles_before = (first_modis_date - first_MAR_date)/8
n_8day_cycles_after = (last_MAR_date - first_modis_date)/8

# define start and end day of MAR data
MAR_start = np.datetime64(first_modis_date - datetime.timedelta(days=(n_8day_cycles_before.days*8)))
MAR_end = np.datetime64(first_modis_date + datetime.timedelta(days=(n_8day_cycles_after.days*8)))

# drop irrelevant days of MAR data
surface_temperature = surface_temperature.where(surface_temperature.TIME > MAR_start,drop=True)
surface_temperature = surface_temperature.where(surface_temperature.TIME < MAR_end, drop=True)

# if time is not divisible by 8 days drop last x entries
surface_temperature = surface_temperature[:(len(surface_temperature)//8)*8]

#%%
# check that the difference between dates in MAR is not too big
if (last_MAR_date - first_MAR_date).days - len(surface_temperature) > 50:
    print('check MAR dates!')

# average surface temperature over 8-days
surface_temperature_8day = surface_temperature.coarsen(TIME=8).mean()
# timestamp now indicates the middle of the interval, while the modis timestamp indicates the beginning of the 8-day period
surface_temperature_8day["modis_time"] = surface_temperature_8day.TIME - pd.Timedelta(4, unit='D')

#%%
# function extract data at meteorite and toclassify locations
def extractdata(
        data_raw,savename,variablename,file_locs_mets,file_locs_toclassify):
    # import meteorite locations (gridded)
    locs_mets = pd.read_csv('../data/'+file_locs_mets+'.csv')
    # select values at meteorite locations (change coordinates to MAR values in km)
    data_at_mets = data_raw.interp(
        X=locs_mets.x.to_xarray()/1000., 
        Y=locs_mets.y.to_xarray()/1000.)
    # export data at meteorite locations
    data_at_mets_df = data_at_mets.to_dataframe(name=variablename)[
        ['X','Y',variablename]]
    data_at_mets_df.to_csv(
        '../data/'+savename+'_at_mets.csv',
        header=True,index=False)
    
    # import locations to be classified
    locs_toclass = pd.read_csv('../data/'+file_locs_toclassify+'.csv')
    # select values at locations to be classified
    data_at_toclass = data_raw.interp(
        X=locs_toclass.x.to_xarray()/1000., 
        Y=locs_toclass.y.to_xarray()/1000.)
    # export data
    data_at_toclass_df = data_at_toclass.to_dataframe(name=variablename)[
        ['X','Y',variablename]]
    return data_at_mets_df, data_at_toclass_df

# function to calculate the 99th percentile over a 19yr period
def timemask(start_date):
    # define timemask
    timemask_left = np.datetime64(start_date - datetime.timedelta(days=1))
    timemask_right = np.datetime64(start_date + datetime.timedelta(days=19*365.25))
    timemask = ((surface_temperature_8day['modis_time'] >= 
                timemask_left) & (surface_temperature_8day['modis_time'] <= 
                                                    timemask_right))
    # select data over the period
    surface_temperature_8day_modistime = surface_temperature_8day.where(timemask,drop=True)
    
    # rechunk dataarray along time dimension
    surface_temperature_rechunked = surface_temperature_8day_modistime.chunk({'TIME': -1})
    # calculate 99th percentile
    surf_temp_99thperc = surface_temperature_rechunked.quantile(0.99,dim='TIME')
    # use function extractdata to extract data at meteorite and toclassify locations
    data_at_mets, data_at_toclass = extractdata(surf_temp_99thperc,str(start_date),'99th_MAR','stempPERC99_at_mets','stempPERC99_at_toclass')
    return(data_at_mets, data_at_toclass)

#%%
# select values on meteorite locations and on other observation locations
# read in meteorite location file
locs_mets = pd.read_csv('../data/stempPERC99_at_mets.csv')
locs_toclass = pd.read_csv('../data/stempPERC99_at_toclass.csv')
# use np.round to account for tiny differences (order 10^-9) in coordinates that are likely introduced through conversion from meters to kms
# locs_toclass.x = np.round(locs_toclass.x)
# locs_toclass.y = np.round(locs_toclass.y)

yrs = np.arange(2001,2082,1)

for yr in yrs:
    start_date = datetime.date(yr,1,1)
    print(start_date)
    MAR_at_mets, MAR_at_toclass = timemask(start_date)
    # prepare dataframe for merge
    # reset coordinates to meters
    MAR_at_mets['x'] = MAR_at_mets.X*1000.
    MAR_at_mets['y'] = MAR_at_mets.Y*1000.
    MAR_at_mets = MAR_at_mets.drop(['X','Y'],axis=1)
    # rename column according to starting year of 19 year period
    MAR_at_mets = MAR_at_mets.rename(columns={"99th_MAR":str(yr)})
    # merge year with all observations
    locs_mets = pd.merge(locs_mets,MAR_at_mets)
    # prepare dataframe for merge
    # reset coordinates to meters
    # use np.round to account for tiny differences (order 10^-9) in coordinates that are likely introduced through conversion from meters to kms
    MAR_at_toclass['x'] = np.round(MAR_at_toclass.X*1000.)
    MAR_at_toclass['y'] = np.round(MAR_at_toclass.Y*1000.)
    MAR_at_toclass = MAR_at_toclass.drop(['X','Y'],axis=1)
    # rename column according to starting year of 19 year period
    MAR_at_toclass = MAR_at_toclass.rename(columns={"99th_MAR":str(yr)})
    # merge year with all observations
    locs_toclass = pd.merge(MAR_at_toclass,locs_toclass, on=['x','y'])




#%%
# compare MODIS observations to MAR data
# merge all observations
merged_obs = pd.merge(locs_toclass,locs_mets,on=["x","y",'stemp','2001'],how="left")

# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig,ax = plt.subplots(1,1,figsize=(8.8/2.54, 8/2.54))
# plot 2d histogram
plt.hist2d(merged_obs['2001'],merged_obs['stemp'],bins=60,cmap='Oranges',cmin=0)
# set aspect equal
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plot labels and title
plt.xlabel('MAR temperature (°C)')
plt.ylabel('MODIS temperature (°C)')
plt.title('99th percentile of 19-year distribution \nof 8-daily surface temperature estimates \nover all places used for classification')
# calculate Pearson correlation  coefficient
corr, _ = pearsonr(locs_toclass['2001'],locs_toclass['stemp'])
ax.annotate(f'Pearson correlation coefficient: {np.round(corr,2)}', xy=(-24,5))
# calculate trend and offset
trend,offset = np.polyfit(locs_toclass['2001'],locs_toclass['stemp'],1)
#plt.plot(locs_toclass['2001'],np.polyval((trend,offset),locs_toclass['2001']),linewidth=0.3,color='k',label=f'{np.round(trend,2)}*$T_{{MAR}}$ + {np.round(offset,2)}')
# plot 1-1 slope
plt.plot(locs_toclass['2001'],np.polyval((1,0),locs_toclass['2001']),linewidth=0.3,color='k',label=f'1:1 slope')
# plot colorbar
plt.colorbar(label='Counts')
# plot legend
plt.legend(loc='lower right',fontsize=9)

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig(f'../figures/MAR_vs_MODIS_{MAR_run}_.pdf',bbox_inches = 'tight',
    pad_inches = 0.01,dpi=300)


#%%
# export temperature anomalies
for yr in yrs:
    # prepare dataframes for export
    # calculate the increase in temperature with respect to 2001 and add this to the observed temperatures
    locs_mets['altered_temp'] = (locs_mets[str(yr)] - locs_mets['2001']) + locs_mets['stemp'] 
    yr_mets = locs_mets[['x','y','altered_temp']].rename(columns={'altered_temp':'stemp'})

    locs_toclass['altered_temp'] = (locs_toclass[str(yr)] - locs_toclass['2001']) + locs_toclass['stemp'] 
    yr_toclass = locs_toclass[['x','y','altered_temp']].rename(columns={'altered_temp':'stemp'})

    # export dataframes as .csv
    yr_mets.to_csv(f'../data/{MAR_run}_{str(yr)}_mets.csv', index=False) # the year is the start year of 19yr interval (!)
    yr_toclass.to_csv(f'../data/{MAR_run}_{str(yr)}_toclass.csv', index=False)
    
    # delete variables
    del(yr_mets,yr_toclass)
#%%
# plot values over time
# scatter values at meteorite finding locations
plt.scatter(yrs+19,locs_mets.median()[3::],color='orange')
errorbarperc = 0.01
plt.errorbar(yrs+19,locs_mets.median()[3::],
             [locs_mets.quantile(errorbarperc)[3::]-locs_mets.median()[3::],
              locs_mets.median()[3:-1]-locs_mets.quantile(1-errorbarperc)[3::]],
             fmt='none',
             capsize=3,
             color='orange')
# scatter values at locations to classify
locs_toclass_plt = locs_toclass.drop(columns=['x','y','stemp','altered_temp'])
plt.scatter(locs_toclass_plt.columns.astype(int)+19,locs_toclass_plt.median(),color='gray')
plt.errorbar(locs_toclass_plt.columns.astype(int)+19,locs_toclass_plt.median(),
             [locs_toclass_plt.quantile(errorbarperc)-locs_toclass_plt.median(),
              locs_toclass_plt.median()-locs_toclass_plt.quantile(1-errorbarperc)],
             fmt='none',
             capsize=3,
             color='gray')
# plot labels
plt.xlabel('end of interval (yr)')
plt.ylabel('99th percentile of surface temperature')
# set limits
plt.xlim([1999,2100])
# save figure
plt.savefig('../figures/temp_evolution.png',dpi=200)
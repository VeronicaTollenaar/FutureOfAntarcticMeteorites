# compute temperature changes over time, correlate meteorite losses with temperature increase
# prepare Figure S6, Figure S4, Figure S5, and Figure S2
# import packages
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import geopandas
import rasterio
import rasterio.features
import rasterio.mask
from scipy.stats import pearsonr
import xcdat

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# prepare climate model data
# define data directory
data_dir = '../data'
# define run of climate model MAR
MAR_run = 'CESM2_ssp585'
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
#%%
# estimate the yearly average surface temperature over antarctica (exclude sea!)
# first average surface temperature over 1 year
surface_temperature_year = surface_temperature.coarsen(TIME=365).mean()
# convert numpy datetime64 to integer value indicating the year
yr_array = surface_temperature_year.TIME.values.astype('datetime64[Y]').astype(int)+1970

# mask data with coastlines
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# create union of ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union

# convert coordinates MAR to EPSG:3031
surface_temperature_year['X'] = surface_temperature_year.X*1000
surface_temperature_year['Y'] = surface_temperature_year.Y*1000
# set transform parameters
resolution = surface_temperature_year.X[1].values-surface_temperature_year.X[0].values
ll_x_main = surface_temperature_year.X.min().values
ur_y_main = surface_temperature_year.Y.max().values
surface_temperature_year.attrs['transform'] = (resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))

# calculate mask for data
ShapeMask = rasterio.features.geometry_mask([ice_boundaries],
                                      out_shape=(len(surface_temperature_year.Y),
                                                 len(surface_temperature_year.X)),
                                      transform=surface_temperature_year.transform,
                                      invert=True)
ShapeMask = xr.DataArray(ShapeMask, 
                         dims=({"Y":surface_temperature_year["Y"][::-1], "X":surface_temperature_year["X"]}))
# flip shapemask upside down
ShapeMask= ShapeMask[::-1]
# mask data
surface_temperature_masked = surface_temperature_year.where((ShapeMask == True),drop=True)
# check data with a plot
ice_boundaries_raw.boundary.plot()
surface_temperature_masked[0].plot()

#%%
# calculate yearly average surface temperature over Antarctica
T_mean = surface_temperature_masked.mean(dim=("Y","X")).values
# plot values
plt.scatter(yr_array,T_mean)
plt.xlabel('year')
plt.ylabel('temperature')
plt.title('mean surface temperature over Antarctica (ocean masked out)')

# fit 2nd order polynomial through data points
a,b,c = np.polyfit(yr_array,T_mean,2)
def Temp(yr):
    T = a*(yr**2) + b*(yr) + c
    return(T)
plt.plot(yr_array,Temp(yr_array))
plt.show()

# adjust c so that polynomial crosses zero in 2020 and replot
shift = Temp(2020)
plt.scatter(yr_array,T_mean-shift)
plt.plot(yr_array,Temp(yr_array)-shift)
plt.xlabel('year')
plt.ylabel('temperature wrt 2020')
plt.title('mean surface temperature over Antarctica (ocean masked out)')


# define inverse function (through quadratic formula)
def Yr(temp):
    yr = (-b + np.sqrt(b**2-(4*a*(c-temp-shift))))/(2*a)
    return(yr)
# calculate years when temperature increases with x degrees
t_x = np.arange(0,7.5,0.5)
yrs = np.array([float(Yr(t)) for t in t_x])
# plot fitted data points
plt.scatter(yrs,t_x,color='k')

# export data
t_x_df = pd.DataFrame(data={'t_x':t_x,'yrs':yrs})
t_x_df.to_csv('../results/time_vs_temp.csv',index=False)



#%%
# compare average temp to 99th percentile of 19-year distribution of 8-daily surface temperatures (as used in model)
# define empty list
T99 = []
# append 99th percentile of 19-year distr of 8-daily surface temperatures at places where meteorites have been found
for yr in range(2001,2082):
    T = pd.read_csv(f'../data/CESM2_ssp585_{yr}_mets.csv').mean()['stemp']
    T99.append(T)

# reformat data as dataframe
df_compare_99 = pd.DataFrame({'year':np.arange(2001+19,2082+19,1),'T_99':T99})
df_compare_av = pd.DataFrame(data={'year':yr_array,'T_average':Temp(yr_array)})
# merge dataframes
df_compare = pd.merge(df_compare_99,df_compare_av,on='year')

# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig,axs = plt.subplots(1,1,figsize=(8.8/2.54, 8/2.54))
# scatter values
plt.scatter(df_compare['T_average'],df_compare['T_99'],s=3,label='Data points')
# set axes equal (to evaluate relationship between temperatures)
axs.set_aspect('equal', 'box')

# fit 1st order polynomial through data points
trend,offset = np.polyfit(df_compare['T_average'],df_compare['T_99'],1)
# plot 1st order polynomial
plt.plot(df_compare['T_average'],np.polyval((trend,offset),df_compare['T_average']),linewidth=0.5,color='k',label='Linear fit')
# plot labels
plt.xlabel('Antarctic-wide yearly average surface temperature')
plt.ylabel('99th percentile of 19 year distribution of \n8-daily surface temperature estimates over\n places where meteorites have been found')
axs.yaxis.set_label_coords(-0.1, 0.45)
# annotate plot
axs.annotate(f'For every 1°C increase of average temperatures, \nthe 99th percentile increases with {np.round(trend,2)} °C',
            xy=(-34.2,-9.5), fontsize=7)
# calculate pearson correlation coefficient
corr, _ = pearsonr(df_compare['T_average'].values,df_compare['T_99'].values)
axs.annotate(f'Pearson correlation coefficient: {np.round(corr,3)}', xy=(-34.2,-10),fontsize=7)

# plot legend
plt.legend(loc='lower right')

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig('../figures/99thperc_vs_average.pdf',bbox_inches = 'tight',
    pad_inches = 0.01,dpi=300)


#%%
# calculate trend of extreme temperatures (stored in df_compare['T_99'])

# fit 2nd order polynomial through data points
a,b,c = np.polyfit(df_compare['year'],df_compare['T_99'],2)
def Temp(yr):
    T = a*(yr**2) + b*(yr) + c
    return(T)
plt.plot(df_compare['year'],Temp(df_compare['year']))
plt.show()

# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig,axs = plt.subplots(1,1,figsize=(8.8/2.54, 8/2.54))

# adjust c so that polynomial crosses zero in 2020 and replot
shift = Temp(2020)
# plot modelled temperatures
plt.scatter(df_compare['year'],df_compare['T_99']-shift,label='Modelled temperatures',marker='.')
# plot fit trough data
plt.plot(df_compare['year'],Temp(df_compare['year'])-shift,label='2nd order polynomial fit through temperatures')
# plot labels and title
plt.xlabel('Year')
plt.ylabel('Temperature with respect to 2020 (°C)')
plt.title('99th percentile of 19 year distribution \nof 8-daily surface temperature estimates \nover places where meteorites have been found')

# define inverse function (through quadratic formula)
def Yr(temp):
    yr = (-b + np.sqrt(b**2-(4*a*(c-temp-shift))))/(2*a)
    return(yr)
# calculate years when temperature increases with x degrees
t_x = np.arange(0,5.5,0.5)
yrs = np.array([float(Yr(t)) for t in t_x])
# plot data points
plt.scatter(yrs,t_x,color='k',zorder=3,label='Estimated years for +0.5°C intervals')
plt.legend(fontsize=8,loc='lower left')
plt.ylim([-2,5.5])

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# plot figure
fig.savefig('../figures/Temp_increase.pdf',bbox_inches = 'tight',
    pad_inches = 0.01,dpi=300)

# export data
t_x_df = pd.DataFrame(data={'t_x':t_x,'yrs':yrs})
t_x_df.to_csv('../results/time_vs_temp99.csv',index=False)
#%%
# compare temperature increase (NEAR-MAXIMUM) directly to meteorite losses 
# calculate relative temperature increase from year to year
rel_temp_incr99 = [float(df_compare[df_compare['year']==yr]['T_99'].values-df_compare[df_compare['year']==yr-10]['T_99'].values) for yr in np.arange(2030,2101,1)]
plt.scatter(np.arange(2030,2101,1),rel_temp_incr99)

# function to calucate number of meteorites
def n_mets(n_obs,
             precision,
             sensitivity,
             n_mets_per_cell):
    TP_t = precision * n_obs
    FN_t = (1-sensitivity)/sensitivity * TP_t
    n_posobs = TP_t + FN_t
    n_mets = n_posobs*n_mets_per_cell
    mets = np.round(n_mets,0)
    return(mets)
# import meteorite loss data
# define climate model run
climate_model = 'CESM2_ssp585'
# import processed data (see map_yr_lost_rev.py)
all_years_sel = pd.read_csv(f'../results/{climate_model}_all_years.csv', index_col=['x','y'])
all_years_sel = all_years_sel.reset_index()
# drop rows where meteorites have already been found
# open meteorite locations
locs_mets = pd.read_csv('../data/stempPERC99_at_mets.csv')
rows_to_drop = pd.merge(all_years_sel,locs_mets,how='inner',on=['x','y'])
all_years_merged = all_years_sel.merge(locs_mets,
                                          how='outer',
                                          on=['x','y'])
all_years_finds_excluded = all_years_merged[np.isnan(all_years_merged['stemp'])].drop(
                            ['stemp'],
                            axis=1)
# sum values per year
sum_per_year = pd.DataFrame(all_years_finds_excluded.drop(columns=['x','y','ever_meteorite']).sum(axis=0)).rename(columns={0:'n_obs_max'})
sum_per_year['n_obs_min'] = all_years_finds_excluded[all_years_finds_excluded['2001']==True].drop(columns=['x','y','ever_meteorite']).sum(axis=0)
sum_per_year = sum_per_year.reset_index().rename(columns={'index':'year'})

# correct for meteorites already found
# --> we already found 45213 meteorites in 2020. We accounted for 12,906 finds, but not for the remaining 32307 meteorites. So the lower bound and the upper bound of the interval can be diminished by this number
sum_per_year['mets_min'] = [n_mets(n_obs,0.47,0.74,5) - 32307 for n_obs in sum_per_year['n_obs_min']]
sum_per_year['mets_max'] = [n_mets(n_obs,0.81,0.48,5) - 32307 for n_obs in sum_per_year['n_obs_max']]
sum_per_year['average'] = (sum_per_year.mets_min+sum_per_year.mets_max)/2


#%%
# relative NEAR-MAXIMUM temperature increase for all different time intervals
# define empty lists
len_ints = np.arange(1,82,1)
rel_temps = []
rel_mets_min = []
rel_mets_max = []
center_yr = []
len_interval = []
# loop over all possible lengths of intervals
for len_int in len_ints:
    # compute the relative temperature increas
    rel_temp_incr99 = [float(df_compare[df_compare['year']==yr]['T_99'].values-df_compare[df_compare['year']==yr-len_int]['T_99'].values) for yr in np.arange(2020+len_int,2101,1)]
    rel_temps.extend(rel_temp_incr99)
    # compute the relative decrease of meteorites for the minimum and maximum estimate of meteorites
    rel_met_decr_min = [float((sum_per_year[sum_per_year['year']==str(int(yr))]['mets_min'].values-sum_per_year[sum_per_year['year']==str(int(yr-len_int))]['mets_min'].values)) for yr in np.arange(2020+len_int-19,2101-19,1)]
    rel_met_decr_max = [float((sum_per_year[sum_per_year['year']==str(int(yr))]['mets_max'].values-sum_per_year[sum_per_year['year']==str(int(yr-len_int))]['mets_max'].values)) for yr in np.arange(2020+len_int-19,2101-19,1)]
    # append calculated values to list
    rel_mets_min.extend(rel_met_decr_min)
    rel_mets_max.extend(rel_met_decr_max)
    # append center year of estimate to list
    center_yr.extend(list(np.arange(2020+len_int/2,2101-len_int/2,1)))
    # append length of interval to list
    len_interval.extend(list(len_int*np.ones(len(rel_temp_incr99))))
# conserve all data in dataframe
df_rel = pd.DataFrame(data={'center_year':center_yr,
                            'len_interval':len_interval,
                            'rel_temp_incr99':rel_temps,
                            'rel_met_decr_min':rel_mets_min,
                            'rel_met_decr_max':rel_mets_max})
#%%
# plot values
fig,axs = plt.subplots(1,1,figsize=(8.8/2.54, 9/2.54))
# define colors
c_min = 'navy'
c_max = 'gray'
# plot minimum estimate
plt.scatter(df_rel['rel_temp_incr99'],df_rel['rel_met_decr_min'],marker='.',s=0.2,color=c_min)
# plot maximum estimate
plt.scatter(df_rel['rel_temp_incr99'],df_rel['rel_met_decr_max'],marker='.',s=0.2,color=c_max)
# plot labels
plt.xlabel('Extreme temperature increase \n at meteorite finding locations (°C)')
plt.ylabel('Difference in number of meteorites (continent-wide)')

# caluclate correlation coefficient
corr_min, _ = pearsonr(df_rel['rel_temp_incr99'].values,df_rel['rel_met_decr_min'].values)
corr_max, _ = pearsonr(df_rel['rel_temp_incr99'].values,df_rel['rel_met_decr_max'].values)
# fit 1st order polynomial through data points
trend_min,offset = np.polyfit(df_rel['rel_temp_incr99'].values,df_rel['rel_met_decr_min'].values,1)
trend_max,offset = np.polyfit(df_rel['rel_temp_incr99'].values,df_rel['rel_met_decr_max'].values,1)

# plot trends
plt.plot(df_rel['rel_temp_incr99'],np.polyval((trend_min,offset),df_rel['rel_temp_incr99']),label='lower bound',color=c_min)
plt.plot(df_rel['rel_temp_incr99'],np.polyval((trend_max,offset),df_rel['rel_temp_incr99']),label='upper bound',color=c_max)

# annotate plot
axs.annotate(f'pearson correlation coefficient: {np.round(corr_min,3)} \ntrend: {int(np.round(trend_min,-2))} meteorites/°C', xy=(1.3,-40000),fontsize=8,color=c_min)
axs.annotate(f'pearson correlation coefficient: {np.round(corr_max,3)} \ntrend: {int(np.round(trend_max,-2))} meteorites/°C', xy=(0,-500000),fontsize=8,color=c_max)

# plot legend
plt.legend(loc='lower left')

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
# save figure
fig.savefig('../figures/correlation.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)


#%%
# compare meteorite losses to global temperature increase
# import global temperature data
glob_temp = xcdat.open_dataset('../data/year-tas-cesm2-histo-ssp585.nc')

# average global data using spatial average
glob_temp_average = glob_temp.spatial.average("tas",axis=["X","Y"])["tas"]

# glob_temp_average = glob_temp.mean(dim=['lat','lon'])['tas'] # this is wrong: surface of each gridcell is not accounted for

#%%
# convert data to dataframe
datetimes = glob_temp_average['time'].values
years = [datetime_.year+0.5 for datetime_ in datetimes]
glob_temp_df = pd.DataFrame(data={'Tas':glob_temp_average,
                            'year':years})

# drop data before 2020
glob_temp_sel = glob_temp_df[glob_temp_df['year']>2020]

# fit 2nd order polynomial through data points
a,b,c = np.polyfit(glob_temp_sel['year'],glob_temp_sel['Tas'],2)
def Temp(yr):
    T = a*(yr**2) + b*(yr) + c
    return(T)

# adjust c so that polynomial crosses 1.1 in 2020
shift = Temp(2020) - 1.1
# define inverse function (through quadratic formula)
def Yr(temp):
    yr = (-b + np.sqrt(b**2-(4*a*(c-temp-shift))))/(2*a)
    return(yr)
# calculate years when temperature increases with x degrees
tas_x = np.arange(1.5,6.5,0.5)
yrs = np.array([float(Yr(t)) for t in tas_x])

# export data
t_x_df = pd.DataFrame(data={'tas_x':tas_x,'yrs':yrs})
t_x_df.to_csv('../results/time_vs_tas.csv',index=False)

#%%
# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig,axs = plt.subplots(1,1,figsize=(8.8/2.54, 8/2.54))
# scatter modelled temperatures
plt.scatter(glob_temp_sel.year,glob_temp_sel['Tas']-shift,label='Modelled temperatures',marker='.')
# plot labels
plt.xlabel('Year')
plt.ylabel('Global air temperature with respect \nto pre-industrial values (°C)')
plt.title('Global air temperatures of CESM2 CMIP6')
# plot estimated trend
plt.plot(glob_temp_sel['year'],Temp(glob_temp_sel['year'])-shift,label='2nd order polynomial fit through temperatures')
# plot data points
plt.scatter(yrs,tas_x,color='k',label='Estimated years for +0.5°C intervals',zorder=3)
# plot legend
plt.legend(fontsize=8,loc='lower left')
# set ylimit
plt.ylim([-1,6.8])


# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figures
fig.savefig('../figures/Tas_increase_.pdf',bbox_inches = 'tight',
    pad_inches = 0.01,dpi=300)

# calculate when +2.7 and when +2.8
Yr(2.7)
Yr(2.8)
#%%
# calculate Tas (smoothed) for each year
Tasses = [Temp(yr_) - shift for yr_ in np.arange(2020,2101,1)]
# store in dataframe
Tas_per_year = pd.DataFrame({'year':np.arange(2020,2101,1),'Tas':Tasses}, columns= ['year','Tas'])
# export as .csv
Tas_per_year.to_csv('../results/Tas_per_year.csv',index=False)

#%%
# get meteorite losses WRT GLOBAL TEMPERATURE
# reset year to center of year
glob_temp_sel['year_'] = glob_temp_sel['year'] - 0.5

# relative temperature increase for all possible intervals
# define empty lists
len_ints = np.arange(1,82,1)
rel_temps = []
rel_mets_min = []
rel_mets_max = []
center_yr = []
len_interval = []
# loop over all possible lengths of intervals
for len_int in len_ints:
    # compute the relative temperature increas
    rel_temp_incrTas = [float(glob_temp_sel[glob_temp_sel['year_']==yr]['Tas'].values-glob_temp_sel[glob_temp_sel['year_']==yr-len_int]['Tas'].values) for yr in np.arange(2020+len_int,2101,1)]
    rel_temps.extend(rel_temp_incrTas)
    # compute the relative decrease of meteorites for the minimum and maximum estimate of meteorites
    rel_met_decr_min = [float((sum_per_year[sum_per_year['year']==str(int(yr))]['mets_min'].values-sum_per_year[sum_per_year['year']==str(int(yr-len_int))]['mets_min'].values)) for yr in np.arange(2020+len_int-19,2101-19,1)]
    rel_met_decr_max = [float((sum_per_year[sum_per_year['year']==str(int(yr))]['mets_max'].values-sum_per_year[sum_per_year['year']==str(int(yr-len_int))]['mets_max'].values)) for yr in np.arange(2020+len_int-19,2101-19,1)]
    # append calculated values to list
    rel_mets_min.extend(rel_met_decr_min)
    rel_mets_max.extend(rel_met_decr_max)
    # append center year of estimate to list
    center_yr.extend(list(np.arange(2020+len_int/2,2101-len_int/2,1)))
    # append length of interval to list
    len_interval.extend(list(len_int*np.ones(len(rel_temp_incrTas))))
# conserve all data in dataframe
df_rel = pd.DataFrame(data={'center_year':center_yr,
                            'len_interval':len_interval,
                            'rel_temp_incrTas':rel_temps,
                            'rel_met_decr_min':rel_mets_min,
                            'rel_met_decr_max':rel_mets_max})
#%%
# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig,axs = plt.subplots(1,1,figsize=(8.8/2.54, 9/2.54))
# define colors
c_min = 'navy'
c_max = 'gray'
# plot minimum estimate
plt.scatter(df_rel['rel_temp_incrTas'],df_rel['rel_met_decr_min'],marker='.',s=0.2,color=c_min)
# plot maximum estimate
plt.scatter(df_rel['rel_temp_incrTas'],df_rel['rel_met_decr_max'],marker='.',s=0.2,color=c_max)
# plot labels
plt.xlabel('Relative increase of global air temperature (°C)')
plt.ylabel('Difference in number of meteorites (continent-wide)')
# caluclate correlation coefficient
corr_min, _ = pearsonr(df_rel['rel_temp_incrTas'].values,df_rel['rel_met_decr_min'].values)
corr_max, _ = pearsonr(df_rel['rel_temp_incrTas'].values,df_rel['rel_met_decr_max'].values)
# fit 1st order polynomial through data points
trend_min,offset = np.polyfit(df_rel['rel_temp_incrTas'].values,df_rel['rel_met_decr_min'].values,1)
trend_max,offset = np.polyfit(df_rel['rel_temp_incrTas'].values,df_rel['rel_met_decr_max'].values,1)
# plot estimated trends
plt.plot(df_rel['rel_temp_incrTas'],np.polyval((trend_min,offset),df_rel['rel_temp_incrTas']),label='lower bound',color=c_min)
plt.plot(df_rel['rel_temp_incrTas'],np.polyval((trend_max,offset),df_rel['rel_temp_incrTas']),label='upper bound',color=c_max)
# annotate with Pearson correlation coefficients
axs.annotate(f'Pearson correlation coefficient: {np.round(corr_min,3)} \nTrend: {int(np.round(trend_min,-2))} meteorites/°C', xy=(1.8,-40000),fontsize=8,color=c_min)
axs.annotate(f'Pearson correlation coefficient: {np.round(corr_max,3)} \nTrend: {int(np.round(trend_max,-2))} meteorites/°C', xy=(0,-500000),fontsize=8,color=c_max,annotation_clip=False)

# plot legend
plt.legend(loc='lower left')

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
# save figure
fig.savefig('../figures/correlation_Tas_.png',bbox_inches = 'tight',
    pad_inches = 0.01,dpi=300)



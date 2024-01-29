# analyze and plot losses of meteorites over the years (Figure S1, Figure 2, Figure S3)

# import packages
import pandas as pd
import os
import numpy as np
import geopandas
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
# surpress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import re
import rasterio
from rasterio.plot import show
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import xarray as xr
import matplotlib.lines as mlines
# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)


#%%
# generate overview of all years SSP5-8.5(file) -- commented out if file is already generated
climate_model = 'CESM2_ssp585'
# # import classification files
# path = '../results/'
# all_files = glob.glob(os.path.join(path, f"classification_{climate_model}_****.csv"))
# # read in all files and concatenate them to one big dataframe
# all_years = pd.concat((pd.read_csv(f, index_col=['x','y'])['positive_classified'].rename(f[-8:-4]) for f in all_files), axis=1)
# # select only the observations where there has ever been a meteorite
# all_years['ever_meteorite'] = all_years.sum(axis=1)
# all_years_sel = all_years[all_years['ever_meteorite']>0]
# # sort data
# all_years_sel = all_years_sel.sort_index(axis=1)
# # export to csv
# all_years_sel.to_csv(f'../results/{climate_model}_all_years.csv')

# import overview of all years
all_years_sel = pd.read_csv(f'../results/{climate_model}_all_years.csv', index_col=['x','y'])


#%%
# generate overview of all years (file) SSP1-2.6 -- commented out if file is already generated
climate_model126 = 'CESM2_ssp126'
# import classification files
# path = '../results/'
# all_files = glob.glob(os.path.join(path, f"classification_{climate_model126}_****.csv"))
# # read in all files and concatenate them to one big dataframe
# all_years = pd.concat((pd.read_csv(f, index_col=['x','y'])['positive_classified'].rename(f[-8:-4]) for f in all_files), axis=1)
# # select only the observations where there has ever been a meteorite
# all_years['ever_meteorite'] = all_years.sum(axis=1)
# all_years_sel = all_years[all_years['ever_meteorite']>0]
# # sort data
# all_years_sel = all_years_sel.sort_index(axis=1)
# # export to csv
# all_years_sel.to_csv(f'../results/{climate_model126}_all_years.csv')

# import overview of all years
all_years_sel126 = pd.read_csv(f'../results/{climate_model126}_all_years.csv', index_col=['x','y'])


# some stats
# # are there many meteorites appearing and dissapearing multiple times?
# # number of meteorites that remain always there
# perc_always_there = len(all_years_sel[all_years_sel['ever_meteorite']==81])/all_years_sel['2001'].sum()
# print(f'{np.round(perc_always_there,3)*100} % of the meteorites remain always at the surface')
# # number of meteorites that are there in 2001 and dissapear at once
# def check_seq(array):
#     try:
#         answer = list(array) == list(range(min(array),max(array)+1))
#     except ValueError:
#         answer = False
#     return answer
# all_years_sel['dissapear_once'] = [check_seq(all_years_sel.iloc[j][all_years_sel.iloc[j]==False].index.values.astype(int)) for j in range(len(all_years_sel))]

# perc_dissapear_once = len(all_years_sel[(all_years_sel['2001']==True) & (all_years_sel['dissapear_once']==True)])/all_years_sel['2001'].sum()
# print(f'{np.round(perc_dissapear_once,3)*100} % of the meteorites dissapear at once from the surface')

# # number of meteorites that are there in 2001 and dissapear but come back and dissapear again
# perc_dissapear_mult = len(all_years_sel[(all_years_sel['2001']==True) & (all_years_sel['dissapear_once']==False) & (all_years_sel['ever_meteorite']!=81)])/all_years_sel['2001'].sum()
# print(f'{np.round(perc_dissapear_mult,3)*100} % of the meteorites dissapear but come back at the surface')

# perc of new meteorites (wrt 2001)
# perc_new = len(all_years_sel[(all_years_sel['2001']==False)])/all_years_sel['2001'].sum()
# print(f'percentage of new meteorites is {np.round(perc_new,3)*100}')
# # when do the new meteorites appear?
# all_years_sel['first_appearance'] = [int(all_years_sel.iloc[j][all_years_sel.iloc[j]==True].index.values[0]) for j in range(len(all_years_sel))]

# plt.hist(all_years_sel[all_years_sel['2001']==False]['first_appearance']+19)
# plt.show()
# # how long do they stay?
# plt.hist(all_years_sel[all_years_sel['2001']==False]['ever_meteorite'])
# # how many do reaappear more than once?
# perc_new_reappeare_more_than_once = len(all_years_sel[(all_years_sel['2001']==False) & (all_years_sel['dissapear_once']==False)])/len(all_years_sel[(all_years_sel['2001']==False)])

# all_years_sel[(all_years_sel['2001']==False)].to_csv('../results/new_meteorites.csv')


#%%
# exclude locations where meteorites have been found
# open meteorite finding locations
locs_mets = pd.read_csv('../data/stempPERC99_at_mets.csv')
# select rows to drop
rows_to_drop = pd.merge(all_years_sel,locs_mets,how='inner',on=['x','y'])
# merge all meteorite locations for all years with locations of meteorite finds
all_years_merged = all_years_sel.merge(locs_mets,
                                          how='outer',
                                          on=['x','y'])
# exclude meteorite finding locations
all_years_finds_excluded = all_years_merged[np.isnan(all_years_merged['stemp'])].drop(
                            ['stemp'],
                            axis=1)


# transform all_years_finds_excluded dataframe into geodataframe, considering the appearance of new meteorites for the max estimate and
# ignoring the appearance of new meteorites for the min estimate
# NB years are the beginning of a 19-yr interval - so 2001 is valid for the meteorites in 2020
mets_yrs_max = all_years_finds_excluded.reset_index(drop=True)
mets_yrs_min = all_years_finds_excluded[all_years_finds_excluded['2001']==True].reset_index(drop=True)
# add geometry points
mets_yrs_max['points_max'] = [Point(line[0],line[1]) for line in mets_yrs_max.values]
mets_yrs_min['points_min'] = [Point(line[0],line[1]) for line in mets_yrs_min.values]
# reformat as geopandas dataframe
mets_gdf_max = geopandas.GeoDataFrame(mets_yrs_max, geometry='points_max')
mets_gdf_min = geopandas.GeoDataFrame(mets_yrs_min, geometry='points_min')

#%% 
# exclude locations where meteorites have been found (for low emission scenario)
# open meteorite finding locations
locs_mets = pd.read_csv('../data/stempPERC99_at_mets.csv')
# select rows to drop
rows_to_drop126 = pd.merge(all_years_sel126,locs_mets,how='inner',on=['x','y'])
# merge all meteorite locations for all years with locations of meteorite finds
all_years_merged126 = all_years_sel126.merge(locs_mets,
                                          how='outer',
                                          on=['x','y'])
# exclude meteorite finding locations
all_years_finds_excluded126 = all_years_merged126[np.isnan(all_years_merged126['stemp'])].drop(
                            ['stemp'],
                            axis=1)

#%%
# function to transform n_obs to n_mets, using estimated precision and sensitivity of the classifier (see Tollenaar et al., 2022)
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

#%%
# read in newest meteorite data
meteorites = pd.read_csv('../data/mets_24NOV2022.csv')

# extract numbers from meteorite names
met_year_pre = [re.findall(r'\d+',str(s)) for s in meteorites.Abbrev]

# extract years form meteorite names
met_years = []
for s in met_year_pre:
    if len(s)>0:
        met_years.append(float(s[0][:2]))
    else:
        met_years.append(np.nan)
# assign meteorite years to column in dataframe
meteorites['met_years'] = met_years
# adjust meteorites without data manually
# print(meteorites[np.isnan(meteorites['met_years'])].Year)
# print(meteorites[np.isnan(meteorites['met_years'])].Name)

# create a list of values with which nan values can be replaced
to_replace = []
for i in range(len(meteorites[np.isnan(meteorites['met_years'])])-1):
    year_str = meteorites[np.isnan(meteorites['met_years'])].iloc[i]['Year']
    year_fl = float(year_str[-2::])
    to_replace.append(year_fl)
    #print(meteorites[np.isnan(meteorites['met_years'])].iloc[i]['met_years']).replace(year_fl)
to_replace.append(10)
# replace nan values
meteorites.loc[np.isnan(meteorites['met_years']),'met_years'] = to_replace

# include the century to the years of finding (and manually check with Year)
year_cent = []
for i in range(len(meteorites)):
    if meteorites.iloc[i]['met_years'] < 23:
        year_cent.append(meteorites.iloc[i]['met_years']+2000)
    else:
        year_cent.append(meteorites.iloc[i]['met_years']+1900)
    try:
        if (year_cent[i] - float(meteorites.iloc[i]['Year'])) > 1:
            print(meteorites.iloc[i])
    except:# != 0:
        if 'or' not in meteorites.iloc[i]['Year']:
            print(meteorites.iloc[i])
# replace year of Adelie land with 1912 (instead of 2012)
year_cent[0] = 1912
# met_years resembles the DECEMBER YEAR OF THE EXPIDITION (https://www.lpi.usra.edu/meteor/antarctnom.php)
# so +1 for the year_cent
meteorites['year_cent'] = np.array(year_cent) + 1

# group dataframe based on the year of the meteorite find
meteorites_grouped = meteorites.groupby(year_cent).count()
meteorites_grouped = meteorites_grouped[meteorites_grouped.index>1960]

#%%
# calculate loss rates

# sum the number of meteorites per year
# for maximum estimate
sum_per_year = pd.DataFrame(all_years_finds_excluded.drop(columns=['x','y','ever_meteorite']).sum(axis=0)).rename(columns={0:'n_obs_max'})
# for minimum estimate
sum_per_year['n_obs_min'] = all_years_finds_excluded[all_years_finds_excluded['2001']==True].drop(columns=['x','y','ever_meteorite']).sum(axis=0)

# correct for meteorites already found
# --> we already found 45213 meteorites in 2020. We accounted for 12,906 finds through excluding these locations,
# but not for the remaining 32,307 meteorites. So the lower bound and the upper bound of the interval can be diminished by this number
sum_per_year['mets_min'] = [n_mets(n_obs,0.47,0.74,5) - 32307 for n_obs in sum_per_year['n_obs_min']]
sum_per_year['mets_max'] = [n_mets(n_obs,0.81,0.48,5) - 32307 for n_obs in sum_per_year['n_obs_max']]
sum_per_year['average'] = (sum_per_year.mets_min+sum_per_year.mets_max)/2

# calculate trend per x-year interval
# define length of interval for loss rate
len_int_loss = 20 #yr
# change index in dataframe from string to integer
sum_per_year.index = sum_per_year.index.astype(int)
# least squares to fit a piecewise linear function (forcing the values at "breakpoints" to be equal)
# construct matrix A, observation vector y and corresponding years in vector x
A = np.zeros((len(sum_per_year)+int(len(sum_per_year)/len_int_loss-1),int(len(sum_per_year)/len_int_loss)+1))
A[0:len_int_loss+1,0] = sum_per_year.index.astype(int)[0:len_int_loss+1]
A[len_int_loss+1:,0] = 2001 + len_int_loss

y = np.zeros((len(sum_per_year)+int(len(sum_per_year)/len_int_loss-1)))
y[0:len_int_loss+1] = sum_per_year['average'][0:len_int_loss+1]

x = np.zeros((len(sum_per_year)+int(len(sum_per_year)/len_int_loss-1)))
x[0:len_int_loss+1] = sum_per_year.index.astype(int)[0:len_int_loss+1]

for i in range(1,int(len(sum_per_year)/len_int_loss)):
    A[i*(len_int_loss+1):(i+1)*(len_int_loss+1),i] = np.arange(0,len_int_loss+1,1)
    y[i*(len_int_loss+1):(i+1)*(len_int_loss+1)] = sum_per_year['average'][i*len_int_loss:(i+1)*len_int_loss+1]
    x[i*(len_int_loss+1):(i+1)*(len_int_loss+1)] = sum_per_year.index.astype(int)[i*len_int_loss:(i+1)*len_int_loss+1]
    if i < int(len(sum_per_year)/len_int_loss):
        A[(i+1)*(len_int_loss+1):,i] = len_int_loss
A[:,int(len(sum_per_year)/len_int_loss)]=1

# perform least squares
xhat = np.linalg.lstsq(A,y,rcond=None)[0]
trends = xhat[:-1]
offset = xhat[-1]
yhat = np.matmul(A,xhat.T)

# create array of center of the interval in which meteorites are lost
center_yr_loss = np.arange(0,len(sum_per_year)/len_int_loss-1)*(len_int_loss)+2001+19+(len_int_loss/2)


#%%
# calculate loss rates (low emission scenario) --> force rates to be negative or zero

# sum the number of meteorites per year
# for maximum estimate
sum_per_year126 = pd.DataFrame(all_years_finds_excluded126.drop(columns=['x','y','ever_meteorite']).sum(axis=0)).rename(columns={0:'n_obs_max'})
# for minimum estimate
sum_per_year126['n_obs_min'] = all_years_finds_excluded126[all_years_finds_excluded126['2001']==True].drop(columns=['x','y','ever_meteorite']).sum(axis=0)

# correct for meteorites already found
# --> we already found 45213 meteorites in 2020. We accounted for 12,906 finds through excluding these locations,
# but not for the remaining 32,307 meteorites. So the lower bound and the upper bound of the interval can be diminished by this number
sum_per_year126['mets_min'] = [n_mets(n_obs,0.47,0.74,5) - 32307 for n_obs in sum_per_year126['n_obs_min']]
sum_per_year126['mets_max'] = [n_mets(n_obs,0.81,0.48,5) - 32307 for n_obs in sum_per_year126['n_obs_max']]
sum_per_year126['average'] = (sum_per_year126.mets_min+sum_per_year126.mets_max)/2

# calculate trend per x-year interval
# define length of interval for loss rate
len_int_loss = 20 #yr
# change index in dataframe from string to integer
sum_per_year126.index = sum_per_year126.index.astype(int)
# least squares to fit a piecewise linear function (forcing the values at "breakpoints" to be equal)
# construct matrix A, observation vector y and corresponding years in vector x
A = np.zeros((len(sum_per_year126)+int(len(sum_per_year126)/len_int_loss-1),int(len(sum_per_year126)/len_int_loss)+1))
A[0:len_int_loss+1,0] = sum_per_year126.index.astype(int)[0:len_int_loss+1]
A[len_int_loss+1:,0] = 2001 + len_int_loss

y = np.zeros((len(sum_per_year126)+int(len(sum_per_year126)/len_int_loss-1)))
y[0:len_int_loss+1] = sum_per_year126['average'][0:len_int_loss+1]

x = np.zeros((len(sum_per_year126)+int(len(sum_per_year126)/len_int_loss-1)))
x[0:len_int_loss+1] = sum_per_year126.index.astype(int)[0:len_int_loss+1]

for i in range(1,int(len(sum_per_year126)/len_int_loss)):
    A[i*(len_int_loss+1):(i+1)*(len_int_loss+1),i] = np.arange(0,len_int_loss+1,1)
    y[i*(len_int_loss+1):(i+1)*(len_int_loss+1)] = sum_per_year126['average'][i*len_int_loss:(i+1)*len_int_loss+1]
    x[i*(len_int_loss+1):(i+1)*(len_int_loss+1)] = sum_per_year126.index.astype(int)[i*len_int_loss:(i+1)*len_int_loss+1]
    if i < int(len(sum_per_year126)/len_int_loss):
        A[(i+1)*(len_int_loss+1):,i] = len_int_loss
A[:,int(len(sum_per_year126)/len_int_loss)]=1

# force last interval to have a negative or zero trend
A = np.delete(A,3,1)

# perform least squares
xhat126 = np.linalg.lstsq(A,y,rcond=None)[0]
trends126 = xhat126[:-1]
offset126 = xhat126[-1]
yhat126 = np.matmul(A,xhat126.T)

# create array of center of the interval in which meteorites are lost
center_yr_loss126 = np.arange(0,len(sum_per_year126)/len_int_loss-1)*(len_int_loss)+2001+19+(len_int_loss/2)


#%%
# set fontsize for plot
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)

# plot figure
fig,ax = plt.subplots(1,1,figsize=(8.8/2.54, 6/2.54))
# define colors
c_trend = 'k'
c_585 = '#DF0000'
c_126 = '#00A9CF'

# plot low emission scenario
uncert_126 = ax.fill_between(sum_per_year126.index.astype(int)+19,sum_per_year126.mets_min,sum_per_year126.mets_max,color=c_126,alpha=0.05,linewidth=0)
low_126 = ax.plot(sum_per_year126.index.astype(int)+19,sum_per_year126['mets_min'],color=c_126,linewidth=0.5)
high_126 = ax.plot(sum_per_year126.index.astype(int)+19,sum_per_year126['mets_max'],color=c_126,linewidth=0.5)
mean_126 = ax.plot(sum_per_year126.index.astype(int)+19,sum_per_year126['average'],color=c_126,linewidth=2.5,label='SSP1-2.6\nand lower/upper bound')

# plot high emission scenario
uncert_585 = ax.fill_between(sum_per_year.index.astype(int)+19,sum_per_year.mets_min,sum_per_year.mets_max,color=c_585,alpha=0.05,linewidth=0)
low_585 = ax.plot(sum_per_year.index.astype(int)+19,sum_per_year['mets_min'],color=c_585,linewidth=0.5)
high_585 = ax.plot(sum_per_year.index.astype(int)+19,sum_per_year['mets_max'],color=c_585,linewidth=0.5)
mean_585 = ax.plot(sum_per_year.index.astype(int)+19,sum_per_year['average'],color=c_585,linewidth=2.5,label='SSP5-8.5\nand lower/upper bound')

# plot estimated linear trend
ax.plot(x+19,yhat,color=c_trend,linestyle='--',linewidth=1.2)
lin_trend = ax.plot(x+19,yhat126,color=c_trend,linestyle='--',linewidth=1.2,label='Linear fit')

# set labels
ax.set_ylabel('Meteorites at \n ice sheet surface')
ax.set_xlabel('Year')
ax.set_yticks(np.linspace(0,800000,5))

# set limits
ax.set_ylim(0,930000)
ax.set_xlim(2019,2101)

# format yticks
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))

# plot legend (manually)
p_uncert_126 = matplotlib.patches.Patch(facecolor="none",edgecolor=c_126,linewidth=0.5,alpha=1)
f_uncert_126 = matplotlib.patches.Patch(facecolor=c_126,edgecolor="none",alpha=0.05)
l_126 = mlines.Line2D([], [], color=c_126, linewidth=2.5,solid_capstyle='butt')
p_uncert_585 = matplotlib.patches.Patch(facecolor="none",edgecolor=c_585,linewidth=0.5,alpha=1)
f_uncert_585 = matplotlib.patches.Patch(facecolor=c_585,edgecolor="none",alpha=0.05)
l_585 = mlines.Line2D([], [], color=c_585, linewidth=2.5,solid_capstyle='butt')
l_trend = mlines.Line2D([], [], color=c_trend,linestyle='--', linewidth=1.2)
ax.legend([(p_uncert_126,f_uncert_126,l_126),(p_uncert_585,f_uncert_585,l_585),(l_trend)],['SSP1-2.6 (mean and\nlower/upper bound)','SSP5-8.5 (mean and\nlower/upper bound)', 'Linear fit'])


# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig('../figures/Fig_emissionscenarios.pdf',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#%%
# append zero as last trend
trends126 = np.append(trends126,0)

#%%
# calculate year in which difference between the emission scenarios starts to be larger than 2%
difference_in_percent = 100*(sum_per_year126['average']-sum_per_year['average'])/sum_per_year126['average']
print('scenarios deviate in ', sum_per_year.index.astype(int)[difference_in_percent>2][0]+19)


#%%
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# create union of ice boundaries
ice_boundaries_all = geopandas.GeoSeries(ice_boundaries_raw.unary_union)

#%%
# read in time vs temperature estimates
# 99th percentile of surface temperatures in anatarctica
ticks_temp = pd.read_csv('../results/time_vs_temp99.csv')
# global air temperature 
ticks_tas = pd.read_csv('../results/time_vs_tas.csv')
#%%
# calculate meteorite losses for pie diagrams per region
# import regions
regions = geopandas.read_file('../data/regions_edited.shp').set_geometry('geometry')

# count number of meteorites per region in 2001 (for max and min)
# max
# define year
year = '2001'
# join points and polygons
dfsjoin = geopandas.sjoin(regions, mets_gdf_max)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = regions.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'})

# min
# join points and polygons
dfsjoin = geopandas.sjoin(regions, mets_gdf_min)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}).drop(columns=False)

# calculate number of mets 
mets_region[f'n_min_{year}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
mets_region[f'n_max_{year}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
mets_region[f'average_{year}'] = (mets_region[f'n_max_{year}']+mets_region[f'n_min_{year}'])/2


# loop to calculate number of observations with respect to the previous interval per region per degree of warming
# define the years in which +1.5, +2, +2.5 etc. of global temp wrt pre-industrial is reached
year_of_plusdegree = np.round(ticks_tas['yrs'][:8]) - 19#ticks_tas['yrs'][:6])-19
# predefine warming of first year
warming_previous = 'first'

for i in range(len(year_of_plusdegree)):
    # redefine year as string
    year = str(int(year_of_plusdegree.iloc[i]))
    warming = str(ticks_tas['tas_x'].iloc[i])
    # maximum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions, mets_gdf_max) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'}).drop(columns=False)
    
    #minimum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions, mets_gdf_min) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}).drop(columns=False)
    
    # estimate number of meteorites
    mets_region[f'n_min_plus{warming}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
    mets_region[f'n_max_plus{warming}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
    mets_region[f'average_plus{warming}'] = (mets_region[f'n_max_plus{warming}']+mets_region[f'n_min_plus{warming}'])/2
    
    # estimate meteorites with respect to previous interval
    if warming_previous == 'first':
        mets_region[f'wrt_previous_plus{warming}'] = mets_region[f'average_2001']-mets_region[f'average_plus{warming}']
    else:
        mets_region[f'wrt_previous_plus{warming}'] = mets_region[f'average_plus{warming_previous}']-mets_region[f'average_plus{warming}']
    # save previous year
    warming_previous = warming

    # drop unused columns of mets_region
    mets_region = mets_region.drop(columns=[f'n_obs_min_{year}',
                                            f'n_min_plus{warming}',
                                            f'n_obs_max_{year}',
                                            f'n_max_plus{warming}'])

#%%
# account for warming levels in which the number of meteorites increases (i.e., resulting in negative wedgesizes in the pie diagram)
# we do this by saying that these increased meteorites reduce the loss of meteorites in the previous warming level (so it sketches a too positive scenario)
warming_prev = '1.5'
for warming_level in range(len(year_of_plusdegree)-1):
    warming = str(ticks_tas['tas_x'].iloc[warming_level+1])
    # select the column with respect to the previous estimate of meteorites
    numbers_wrt_previous = mets_region[f'wrt_previous_plus{warming}']
    # loop over all numbers_wrt_previous to see if there are values that are negative and adjust them,
    # skip first column that concerns areas outside the defined regions
    for j in range(1,len(numbers_wrt_previous)-1):
        if numbers_wrt_previous[j]<0:
            if mets_region.loc[j,f'wrt_previous_plus{warming_prev}'] + numbers_wrt_previous[j] > 0:
                mets_region.loc[j,f'wrt_previous_plus{warming_prev}'] = mets_region.loc[j,f'wrt_previous_plus{warming_prev}'] + numbers_wrt_previous[j]
                mets_region.loc[j,f'wrt_previous_plus{warming}'] = 0
            else:
                mets_region.loc[j,f'wrt_previous_plus{str(float(warming_prev) - 0.5)}'] = mets_region.loc[j,f'wrt_previous_plus{str(float(warming_prev) - 0.5)}'] + mets_region.loc[j,f'wrt_previous_plus{warming_prev}'] + numbers_wrt_previous[j]
                mets_region.loc[j,f'wrt_previous_plus{warming_prev}'] = 0
                numbers_wrt_previous[j] = 0
    warming_prev = warming
    
#%%
# change index in dataframe back from integer to string
sum_per_year.index = sum_per_year.index.astype(str)
# calculate percentage per year
perc_per_year = [100*(1-sum_per_year['average'].iloc[i+1]/sum_per_year['average'].iloc[i]) for i in range(len(sum_per_year)-1)]
# repeat for the estimated trends
lineartrend = np.sort(np.array(list(set(yhat))))[::-1]
# calculate the percentage of loss per year with respect to the previous year
perc_per_year_lt = [100*(1-lineartrend[i+1]/lineartrend[i]) for i in range(len(lineartrend)-1)]



#%%
# calculate percentages with respect to 2001 (=2020)
percs = sum_per_year['average']/sum_per_year['average'].loc['2001']

# calculate year of loss
yr_25percloss = int(percs[percs<0.75].index.min())+19
yr_50percloss = int(percs[percs<0.50].index.min())+19
yr_75percloss = int(percs[percs<0.25].index.min())+19
perc_2100 = 100 - percs['2081']*100
perc_2050 = 100 - percs['2031']*100

#%%
# same for the low emission scenario:
# change index in dataframe back from integer to string
sum_per_year126.index = sum_per_year126.index.astype(str)

# calculate percentages SSP1-2.6 with respect to 2001 (=2020)
percs126 = sum_per_year126['average']/sum_per_year126['average'].loc['2001']

# calculate year of loss
yr_25percloss_126 = int(percs126[percs126<0.75].index.min())+19
perc126_2100 = 100 - percs126['2081']*100
perc126_2050 = 100 - percs126['2031']*100

#%%
# calculate percentages wrt temperatures (years calculated in time_vs_temp.py)
# +2.6 in 2048.99955090175
print(f'+2.6 is {100*(1-percs[str(int(np.round(2052.5031968850963,0))-19)])}% loss')
# +2.7 in 2050.5743615040037
print(f'+2.7 is {100*(1-percs[str(int(np.round(2054.244860661909,0))-19)])}% loss')
# +2.8 in 2052.119922797426
print(f'+2.8 is {100*(1-percs[str(int(np.round(2055.952764060757,0))-19)])}% loss')
# copy dataframe of ticks per air temperature
ticks_tas_percs = ticks_tas
# append the percentage lost per degree warming to dataframe
ticks_tas_percs['perc_lost'] = [100*(1-percs[str(int(np.round(ticks_tas['yrs'][i],0))-19)]) for i in range(len(ticks_tas))]
# plot data
plt.scatter(ticks_tas_percs['tas_x'],ticks_tas_percs['perc_lost'])

#%%
# read in MSZs2020 (processed in QGIS, see Tollenaar et al., 2022)
MSZs2020 = geopandas.read_file('../data/613MSZs.shp').set_geometry('geometry')
# open MSZs in 2050 (processed in QGIS, see Tollenaar et al., 2022)
MSZs2050 = geopandas.read_file('../results/MSZs_2050_larger4km2.shp').set_geometry('geometry')
# open MSZs in 2069: this is the year when the percentage of meteorites is reduced to 50% (processed in QGIS, see Tollenaar et al., 2022)
MSZs2069 = geopandas.read_file('../results/mets_2069_larger4km2.shp').set_geometry('geometry')
# open MSZs in 2100 (processed in QGIS, see Tollenaar et al., 2022)
MSZs2100 = geopandas.read_file('../results/mets_2100_larger4km2.shp').set_geometry('geometry')

#%%
# function to open Landsat images given coordinates, to plot as background 
def openJPGgivenbounds(xmin,ymin):
    # define arrays referring to names of LIMA subtiles
    cirref_xs = np.arange(-2700000, 2700000, 150000)
    cirref_ys = np.arange(-2700000, 2700000, 150000)
    # select names of LIMA subtiles corresponding to given coordinates
    cirref_x = cirref_xs[cirref_xs<xmin][-1]
    cirref_y = cirref_ys[cirref_ys<ymin][-1]
    # define list of zeros to make sure names include zeros
    name_zeros = list('0000000')
    # ensure a minus is included in namestring when given coordinates are negative
    if cirref_x < 0:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '-'+name_x_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '+'+name_x_abs
        name_zeros = list('0000000')
    if cirref_y < 0:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '-'+name_y_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '+'+name_y_abs
        name_zeros = list('0000000')
    # define full name of LIMA subtile
    name = 'CIRREF_'+'x'+name_x+'y'+name_y
    # try open LIMA subtile (for high latitudes no data exists)
    try:
        img = rasterio.open('../data/LIMA/'+name+'.jpg')
    except:
        print('no high resolution background image')
        img = 0
    # return image and lowerleft coordinates of image
    return(img,cirref_x,cirref_y)

#%%
# function to calculate closest point on exterior of polygon to a given point (to plot connection between polygon and pie chart)
def annotpoint(polygon,point_x,point_y):
    xx,yy = polygon.exterior.coords.xy
    dists = [np.sqrt((xx[i]-point_x)**2 + (yy[i]-point_y)**2) for i in range(len(xx))]
    x_annot = xx[pd.Series(dists).idxmin()]
    y_annot = yy[pd.Series(dists).idxmin()]
    return(x_annot,y_annot)



#%%
# reset index of dataframes
if sum_per_year.index[0]=='2001':
    sum_per_year = sum_per_year.reset_index(names='year')
    sum_per_year126 = sum_per_year126.reset_index(names='year')
#%%
# relate number of meteorites at the ice sheet surface directly to global temperature increase 
# read in smoothed data of Tas per year
Tas_per_year = pd.read_csv('../results/Tas_per_year.csv')
# merge to number of meteorites per year
sum_per_year['Tas'] = Tas_per_year['Tas']

#%%
# Plot Figure 2
# parameter to plot background data or not (to speed up iterations)
plt_lima = True
# font settings
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)
# plot figure
fig = plt.figure(figsize=(18/2.54, 16/2.54))
# define subaxes
gs = fig.add_gridspec(30, 30)
# map
ax1 = fig.add_subplot(gs[12:,:18])# rows, columns
# petermann ranges
ax5 = fig.add_subplot(gs[11:21,19:])
# allan hills
ax4 = fig.add_subplot(gs[20:,19:])
# trend
ax2 = fig.add_subplot(gs[0:9,17:29])
# barplot
ax_hist = fig.add_subplot(gs[0:9,2:13])

# define colors
c_MSZ2020 =  '#FF7376'
c_MSZ2100 = '#980002'

# plot ice boundaries
ice_boundaries_all.plot(ax=ax1, color='#bfcbe3',zorder=0)
# open graticule data
longs_path = r'../data/60dg_longitude_clipped.shx'
longs_raw = geopandas.read_file(longs_path,encoding='utf-8')
lats_path = r'../data/10dg_latitude_clipped.shx'
lats_raw = geopandas.read_file(lats_path,encoding='utf-8')
# plot graticules
longs_raw.plot(ax=ax1,color='k',linewidth=0.1,zorder=1)
lats_raw.plot(ax=ax1,color='k',linewidth=0.1,zorder=1)
# plot annotations of graticules
longs_annot = longs_raw.boundary.explode()[1::2]
for x, y, label in zip(longs_annot.geometry.x, longs_annot.geometry.y, longs_raw.Longitude):
    if (label[:2]!='30') and (label!='150°E'):
        ax1.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=6, textcoords='offset points', va='center',ha='center',
                path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])

ax1.annotate(lats_raw.iloc[0].Latitude,xy=(0,2.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
ax1.annotate(lats_raw.iloc[1].Latitude,xy=(0,1.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])


# plot boundaries of regions
mets_region[mets_region['name']!='rest'].boundary.plot(ax=ax1,zorder=1,color='dimgray',linewidth=1.1)
# plot meteorites in 2020
MSZs2020.buffer(10000).plot(ax=ax1,linewidth=0,
              facecolor=c_MSZ2020,edgecolor=c_MSZ2020)
# plot meteorites in 2100
MSZs2100.buffer(10000).plot(ax=ax1,linewidth=0,
              facecolor=c_MSZ2100,edgecolor=c_MSZ2100)

# define colors and normalization of colors for pie plot
norm = matplotlib.colors.Normalize(1.5, 5)
colors = matplotlib.colors.LinearSegmentedColormap.from_list('colors',colors=['khaki','orangered'],N=len(ticks_tas['yrs'][:8]))
colors=colors(norm(ticks_tas['tas_x'][:8].values))
colors=np.append(colors,[[0,0,0,0]],axis=0)

# loop over all regions to plot pies
for region in range(len(regions)):
    # plot all regions except "rest"
    if mets_region['name'].iloc[region]!='rest':
        # select numbers with respect to previous interval
        n_mets_rel = mets_region[[column for column in mets_region.columns.values if column.startswith('wrt_previous')]].iloc[region] #[0::10]
        # append number of remaining meteorites
        n_mets_rel = np.append(n_mets_rel,mets_region['average_plus5.0'].iloc[region])
        # define size of pie
        size = np.sqrt((mets_region['average_2001'].iloc[region]))/600
        # add axes to plot pie
        center_region = mets_region.centroid[region].coords[0]
        if mets_region.iloc[region]['name']=='South Victoria Land':
            center_pie_x = center_region[0]+1000000
            center_pie_y = center_region[1]-1000000
        elif mets_region.iloc[region]['name']=='West Antarctica':
            center_pie_x = center_region[0]*1.6
            center_pie_y = center_region[1]*1.3
        elif mets_region.iloc[region]['name']=='Pensacola Mountains':
            center_pie_x = center_region[0]*2.15
            center_pie_y = center_region[1]*2.8
        elif mets_region.iloc[region]['name']=='Queen Maud Mountains':
            center_pie_x = center_region[0]+900000
            center_pie_y = center_region[1]
        elif mets_region.iloc[region]['name']=='North Victoria Land':
            center_pie_x = center_region[0]*1.8
            center_pie_y = center_region[1]*1.35
        elif mets_region.iloc[region]['name']=='Prince Charles Mountains':
            center_pie_x = center_region[0]+700000
            center_pie_y = center_region[1]
        elif mets_region.iloc[region]['name']=='SÃ¸r Rondane Mountains':
            center_pie_x = center_region[0]
            center_pie_y = center_region[1]+620000
        elif mets_region.iloc[region]['name']=='rest':
            center_pie_x = 2.25e6
            center_pie_y = -1.75e6
        else:
            center_pie_x = center_region[0]*1.2
            center_pie_y = center_region[1]+620000
        ax_sub = inset_axes(ax1, width=size, height=size, 
                            loc='center',bbox_to_anchor=(center_pie_x,center_pie_y),
                            bbox_transform=ax1.transData)
        ax_sub.pie(n_mets_rel,colors=colors,startangle=90,counterclock=False,
                   wedgeprops = {'linewidth': 0.2,'edgecolor':'k'})
        # add connecting line
        annot_x, annot_y = annotpoint(mets_region.iloc[region].geometry,center_pie_x,center_pie_y)
        arrowprops=dict(arrowstyle='-', linestyle='--', color='dimgray', linewidth=0.8)
        ax1.annotate('',xy=(annot_x,annot_y),
                 xytext=(center_pie_x,center_pie_y),
                 ha='center',fontsize=7,
                 bbox=dict(facecolor='none', edgecolor='k',
                           linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
                 arrowprops=arrowprops) #f'-25%\n{yr_25percloss}'
# annotate region Enderby Land
coords_centroid = mets_region[mets_region.name=='Enderby Land'].geometry.centroid
ax1.annotate('ENDERBY LAND',
                 xy=(coords_centroid.x*0.7,coords_centroid.y*0.76),ha='center',va='bottom',fontsize=7,color='dimgray')

#create list of colors
color_list = colors

#add patches of circles as legend (representing number of meteorites)
for i in range(1,4):
    # define size of pie
    size = np.sqrt(-30000+(50000*i))/600
    # define label of pie
    label = '{:,}'.format(int(-30000+(50000*i)))
    # add axes to plot pie
    ax_sub = inset_axes(ax1, width=size, height=size, 
                        loc='center left',bbox_to_anchor=((-2.45e6-size*240000,-0.5e6-i*size*700000)),
                        bbox_transform=ax1.transData)
    # plot pie
    ax_sub.pie([1,1],colors=['dimgray'],wedgeprops = {'linewidth': 0.5,'edgecolor':'dimgray'})
    # annotate pie
    ax_sub.annotate(label,xy=(1.1-size*0.1,0),ha='left',va='center',fontsize=7)
# annotate legend
ax_sub.annotate('Present-day estimated number \n of meteorites per region',
                 xy=(-1.6,-1.5),ha='center',va='bottom',fontsize=7,rotation=90,
                 annotation_clip=False)

# add patches as colorbar (representing air temperature increase)
# define spacing
vspacing = -2.42e6
thickness = 115000
len_sect = 350e3
hspacing = -2.3e6
# loop over colors to plot patches
for j in range(len(color_list[:-1])):
    # plot patch
    ax1.add_patch(matplotlib.patches.Rectangle((hspacing+(len_sect*j),vspacing-thickness/2),len_sect*0.94,thickness,
                                              facecolor=color_list[j],edgecolor='k',linewidth=0.2,clip_on=False))
    # annotate patch
    ax1.annotate(f'+{ticks_tas["tas_x"].iloc[j]}',xy=((hspacing+(len_sect*j),vspacing-thickness*2)),rotation=0,fontsize=7,annotation_clip=False)
# plot last patch (arrow)
ax1.add_patch(matplotlib.patches.FancyArrow(hspacing+(len_sect*(len(color_list)-1)),vspacing,
                                            len_sect*0.6,0,facecolor=color_list[-1],width=thickness,head_width=thickness,
                                            head_length=len_sect/4.5,linewidth=0.2,clip_on=False))
# annotate last patch
ax1.annotate(f'+>{ticks_tas["tas_x"].iloc[7]}',xy=((hspacing+(len_sect*(len(color_list)-1)),vspacing-thickness*2)),rotation=0,fontsize=7,annotation_clip=False) 
# annotate legend
ax1.annotate('Meteorites lost under global air \n temperature increase vs. pre-industrial (°C)',
            xy=(hspacing-190e3+(len_sect*4.75),vspacing+thickness*1.5),
            ha='center',rotation=0,fontsize=7,annotation_clip=False)

# add legend for MSZs
# add axis for lengend
ax_legend = inset_axes(ax1, width=size*0.4, height=size*0.7,
                        loc='center',bbox_to_anchor=((1.3e6,-1.4e6)),
                        bbox_transform=ax1.transData)
# add patches in legend
ax_legend.add_patch(matplotlib.patches.Rectangle((0,0.5),0.4,0.4,
                                                  facecolor=c_MSZ2020,edgecolor='k',linewidth=0))
ax_legend.add_patch(matplotlib.patches.Rectangle((0,0),0.4,0.4,
                                                  facecolor=c_MSZ2100,edgecolor='k',linewidth=0))
# annotate patches
ax_legend.annotate('MSZs in 2020',xy=(0.5,0.7),annotation_clip=False,va='center',fontsize=7)
ax_legend.annotate('MSZs in 2100 \nunder SSP5-8.5',xy=(0.5,0.2),annotation_clip=False,va='center',fontsize=7,linespacing=0.95)
# switch of axis
ax_legend.axis('off')

# plot settings (visibility of axes, limits)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlim([-2.6e6,2.7e6])
ax1.set_ylim([-2.4e6,2.25e6])


# plot trend
# define color
c_total = 'k'
# plot uncertainty band, lower and upper bound
ax2.fill_between(sum_per_year.Tas,sum_per_year.mets_min,sum_per_year.mets_max,color=c_total,alpha=0.05,linewidth=0)
ax2.plot(sum_per_year.Tas,sum_per_year['mets_min'],color=c_total,linewidth=0.5)
ax2.plot(sum_per_year.Tas,sum_per_year['mets_max'],color=c_total,linewidth=0.5)
# plot mean
ax2.plot(sum_per_year.Tas,sum_per_year['average'],color=c_total,linewidth=2)

# set labels and ticks
ax2.set_ylabel('Meteorites at \n ice sheet surface')
ax2.set_yticks(np.linspace(0,800000,5))
ticks_tas_sel = [ticks_tas['tas_x'][i] for i in [0,1,2,3,5,7,9]]
ticks_tas_sel[1] = ticks_tas_sel[1].astype(int)
ticks_tas_sel[3:] = [val.astype(int) for val in ticks_tas_sel[3:]]
ax2.set_xticks(ticks_tas_sel,list(map(str,ticks_tas_sel)))
ax2.set_xlabel('Global air temperature vs. pre-industrial (°C)')#average Antarctic surface temperature increase with respect to 2020

# plot legend (manually)
l_upperlower = mlines.Line2D([], [], color=c_total, linewidth=0.5,solid_capstyle='butt')
l_mean = mlines.Line2D([], [], color=c_total, linewidth=2,solid_capstyle='butt')
leg_ax2 = ax2.legend([l_upperlower,l_mean,l_upperlower],['Upper bound','Mean','Lower bound'],
           frameon=False,loc='upper right')
# plot background of legend (manually)
ax2.add_patch(matplotlib.patches.Rectangle((4.7,699000),0.33,143000,
                                               facecolor=c_total,alpha=0.05,clip_on=False))

# format yticks
ax2.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))

# annotate years when x% is lost
# define placement of annotations
vert_place = 120000
# define arrow properties of annotations
arrowprops=dict(arrowstyle='-', linestyle='--', color='k', linewidth=0.8)
# annotate trend
ax2.annotate(f'-25%',xy=(sum_per_year[sum_per_year['year']==str(yr_25percloss-19)]['Tas'],0),
             xytext=(sum_per_year[sum_per_year['year']==str(yr_25percloss-19)]['Tas'],
                     sum_per_year[sum_per_year['year']==str(yr_25percloss-19)]['mets_max']+vert_place*0.88),
                                          ha='center',fontsize=7,
             bbox=dict(facecolor='none', edgecolor='black',
                       linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
             arrowprops=arrowprops) #f'-25%\n{yr_25percloss}'
ax2.annotate(f'-50%',xy=(sum_per_year[sum_per_year['year']==str(yr_50percloss-19)]['Tas'],0),
             xytext=(sum_per_year[sum_per_year['year']==str(yr_50percloss-19)]['Tas'],
                     sum_per_year[sum_per_year['year']==str(yr_50percloss-19)]['mets_max']+vert_place),
             ha='center',fontsize=7,
             bbox=dict(facecolor='none', edgecolor='black',
                       linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
             arrowprops=arrowprops) 
ax2.annotate(f'-75%',xy=(sum_per_year[sum_per_year['year']==str(yr_75percloss-19)]['Tas'],0),
             xytext=(sum_per_year[sum_per_year['year']==str(yr_75percloss-19)]['Tas'],
                     sum_per_year[sum_per_year['year']==str(yr_75percloss-19)]['mets_max']+vert_place),
             ha='center',fontsize=7,
             bbox=dict(facecolor='none', edgecolor='black',
                       linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
             arrowprops=arrowprops) 
    
# set visibility of axes and limits
ax2.spines['right'].set_visible(False)
ax2.set_xlim([1.1,6.25])
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_bounds((1.1,6.1))
ax2.set_ylim([0,900000])
ax2.spines['left'].set_bounds([0,900000])

# plot insets of selected regions

# REGION AX4
region_ax4 = 'South Victoria Land'
# plot MSZs in 2020
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = c_MSZ2020
MSZs2020.plot(ax=ax4,linewidth=0.8,
              facecolor="none",edgecolor=c_MSZ2020,hatch='//////////')
# plot MSZs in 2100
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = c_MSZ2100
MSZs2100.plot(ax=ax4,linewidth=0.8,
              facecolor="none",edgecolor=c_MSZ2100,hatch='\\\\\\\\\\\\\\\\\\')

# set boundaries (manually)
min_x4 = 460000
max_x4 = 655000
min_y4 = -1.425e6
max_y4 = -1.27e6
ax4.set_xlim([min_x4,max_x4])
ax4.set_ylim([min_y4,max_y4])

# load and plot background image
# caluculate how many background images to open (is slow)
if plt_lima == True:
    img_open_x = np.arange(min_x4,max_x4+0.14e6,0.15e6)
    img_open_y = np.arange(min_y4,max_y4+0.14e6,0.15e6)
    for xs in img_open_x:
        for ys in img_open_y:
            backgr,_x,_y = openJPGgivenbounds(xs,ys)
            show(backgr.read(),ax=ax4,transform=backgr.transform)
# hide axes
ax4.xaxis.set_visible(False)
ax4.yaxis.set_visible(False)
# plot scalebar
scalebar = AnchoredSizeBar(ax4.transData,
                           30000, '30 km', 
                           loc='lower left', 
                           pad=0.5, #0.0005
                           color='black',
                           frameon=False, 
                           size_vertical=(max_x4-min_x4)/90,
                           fontproperties=fm.FontProperties(size=9),
                           label_top=True,
                           sep=1)
ax4.add_artist(scalebar)
# plot inset on main map
ins_ax4 = Polygon([[min_x4,min_y4],[max_x4,min_y4],[max_x4,max_y4],[min_x4,max_y4]])
ax1.plot(*ins_ax4.exterior.xy,color='k')

# REGION AX5
region_ax5 = 'Fimbulheimen'
# plot MSZs in 2020
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = c_MSZ2020
MSZs2020.plot(ax=ax5,linewidth=0.8,
              facecolor="none",edgecolor=c_MSZ2020,hatch='//////////')
# plot MSZs in 2100
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = c_MSZ2100
MSZs2100.plot(ax=ax5,linewidth=0.8,
              facecolor="none",edgecolor=c_MSZ2100,hatch='\\\\\\\\\\\\\\\\\\')

# set boundaries (manually)
min_x5 = 300000
max_x5 = 500000
min_y5 = 1.85e6
max_y5 = 2e6
ax5.set_xlim([min_x5,max_x5])
ax5.set_ylim([min_y5,max_y5])

# load and plot background image
# caluculate how many background images to open (is slow)
if plt_lima == True:
    img_open_x = np.arange(min_x5,max_x5+0.14e6,0.15e6)
    img_open_y = np.arange(min_y5,max_y5+0.14e6,0.15e6)
    for xs in img_open_x:
        for ys in img_open_y:
            backgr,_x,_y = openJPGgivenbounds(xs,ys)
            show(backgr.read(),ax=ax5,transform=backgr.transform)

# hide axes
ax5.xaxis.set_visible(False)
ax5.yaxis.set_visible(False)
# plot scalebar
scalebar = AnchoredSizeBar(ax5.transData,
                           30000, '30 km', 
                           loc='lower left', 
                           pad=0.5, #0.0005
                           color='black',
                           frameon=False,
                           size_vertical=(max_x5-min_x5)/90,
                           fontproperties=fm.FontProperties(size=9),
                           label_top=True,
                           sep=1)
ax5.add_artist(scalebar)

# plot inset on main map
ins_ax5 = Polygon([[min_x5,min_y5],[max_x5,min_y5],[max_x5,max_y5],[min_x5,max_y5]])
ax1.plot(*ins_ax5.exterior.xy,color='k')

# adjust spacing of plot
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0)
plt.margins(0,0)

# plot barplot
# define colors
c_finds = 'dimgray' 
c_losses = color_list[3]
c_losses585 = '#980002'
c_losses126 = '#003466'
c_annotations = 'dimgray'
# plot meteorite finds (all data and average data)
# calculate average loss for interval of len_interval
len_int_find = 5
intervals = np.arange(1960,2020,len_int_find)
average_finds = []
center_yr_find = []
for year in intervals:
    average_finds.append(np.mean(meteorites_grouped[(meteorites_grouped.index>=year)&
                                        (meteorites_grouped.index<year+len_int_find)]['year_cent']))
    center_yr_find.append(year + len_int_find/2)
# plot finding rates
ax_hist.bar(center_yr_find,np.array(average_finds),width=3,alpha=1,color=c_finds)

# plot first loss rate
first_loss_rate = (trends[0]+trends126[0])/-2
internal_variability = first_loss_rate + trends[0]
ax_hist.bar(center_yr_loss[0],first_loss_rate, yerr=internal_variability,width=(len_int_loss)*0.9,alpha=1,color=c_finds)

# plot following loss rates
ax_hist.bar(center_yr_loss[1:],np.array(trends[1:])*-1,width=(len_int_loss)*0.9,alpha=1,color=c_losses585) #color_list[:-1]
ax_hist.bar(center_yr_loss[1:],np.array(trends126[1:])*-1,width=(len_int_loss)*0.9,alpha=1,color=c_losses126)

# annotate bars
ax_hist.annotate('Scenario independent', xy=(center_yr_loss[0],550),
                 color='white',
                 fontsize=7,
                 rotation=90,
                 horizontalalignment='center')
ax_hist.annotate('SSP1-2.6', xy=(center_yr_loss[1],220),
                 color='white',
                 fontsize=7,
                 rotation=90,
                 horizontalalignment='center')
ax_hist.annotate('SSP5-8.5', xy=(center_yr_loss[1],3000),
                 color='white',
                 fontsize=7,
                 rotation=90,
                 horizontalalignment='center')

# set ticks
xticks = np.linspace(1980,2120,8)
ax_hist.set_xticks(ticks=xticks[:-1],labels=xticks[:-1],fontsize=7) 
xticklabels = list(xticks.astype(int).astype(str))
ax_hist.set_xticklabels(xticklabels[:-1],rotation=0,fontsize=7)
yticks = np.linspace(0,8000,9)
yticklabels = list(yticks.astype(int).astype(str))
ax_hist.set_yticks(ticks=yticks,labels=yticklabels,fontsize=7)

# set x and y label and limits
ax_hist.set_xlabel('Year')
ax_hist.set_ylabel('Meteorites per year',fontsize=7)
ax_hist.set_xlim([1968,2113])
ax_hist.set_ylim([0,6500]) #0,6500
ax_hist.spines['left'].set_bounds((0,6500))
ax_hist.spines['bottom'].set_position(('data',0))
ax_hist.spines['bottom'].set_bounds((1968,2105))
ax_hist.yaxis.set_label_coords(-0.19,0.5)
ax_hist.xaxis.set_label_coords(0.5,-0.135)
# set background transparency to 0
ax_hist.patch.set_alpha(0)

# annotate bars
ax_hist.annotate('ca. 1000/year', xy=(1995, 2500), xytext=(1995, 2700), 
            fontsize=7, ha='center', va='bottom', color=c_annotations,
            arrowprops=dict(arrowstyle='-[, widthB=3.55, lengthB=1', lw=0.8,color=c_annotations)) #3.6
ax_hist.annotate('Finds,', xy=(1995, 2700), xytext=(1995, 3300), 
            fontsize=7, ha='center', va='bottom', color=c_annotations)

ax_hist.annotate('Losses', xy=(2060, 6200), xytext=(2060, 6400), 
            fontsize=7, ha='center', va='bottom', color=c_annotations,
            arrowprops=dict(arrowstyle='-[, widthB=6.15, lengthB=1', lw=0.8,color=c_annotations))
# plot settings
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)


# annotate panels
# barplot
ax_hist.annotate('A',xy=(0.03,1),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
# map
ax1.annotate('C',xy=(0.03,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
# trend
ax2.annotate('B',xy=(0.03,1),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")],
             annotation_clip=False)
# inset Allan Hills
ax4.annotate('E',xy=(0.03,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax4.annotate('ALLAN HILLS',xy=(0.24,0.625),
                 xycoords='axes fraction',fontsize=7,weight='bold',path_effects=[pe.withStroke(linewidth=1, foreground="white")])
ax4.annotate('ELEPHANT MORAINE',xy=(0.46,0.42),
                 xycoords='axes fraction',fontsize=7,weight='bold',path_effects=[pe.withStroke(linewidth=1, foreground="white")])
ax4.annotate('RECKLING MORAINE',xy=(0.25,0.06),
                 xycoords='axes fraction',fontsize=7,weight='bold',path_effects=[pe.withStroke(linewidth=1, foreground="white")])
# inset Petermann ranges
ax5.annotate('D',xy=(0.03,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax5.annotate('PETERMANN RANGES',xy=(0.13,0.91),xycoords='axes fraction',fontsize=7,weight='bold',path_effects=[pe.withStroke(linewidth=1, foreground="white")])

# move axes position
# align plots to the right
right_align = ax5.get_position().x1
pos_x2 = ax2.get_position()
pos_x2.x1 = right_align
ax2.set_position(pos_x2)

# align plots to the bottom
pos_x4 = ax4.get_position()
pos_x4.y0 = -0.05
ax4.set_position(pos_x4)

# save figure
fig.savefig('../figures/Fig2_rev.pdf',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#%%
# calculate in which year 50% of meteorites is lost in each region
# delete variables
del(mets_region,dfpolynew)

# FOR SSP5-8.5
# DO NOT exclude locations where meteorites have been found

# transform all_years_sel dataframe into geodataframe, considering the appearance of new meteorites for the max estimate and
# ignoring the appearance of new meteorites for the min estimate
mets_yrs_max_finds = all_years_sel.reset_index(drop=True)
mets_yrs_min_finds = all_years_sel[all_years_sel['2001']==True].reset_index(drop=True)


mets_yrs_max_finds['points_max'] = [Point(line[0],line[1]) for line in mets_yrs_max_finds.values]

mets_yrs_min_finds['points_min'] = [Point(line[0],line[1]) for line in mets_yrs_min_finds.values]

mets_gdf_max_finds = geopandas.GeoDataFrame(mets_yrs_max_finds, geometry='points_max')
mets_gdf_min_finds = geopandas.GeoDataFrame(mets_yrs_min_finds, geometry='points_min')

#%%
# calculate in which year 50% of meteorites is lost in each dense meteorite collection site region
#del(mets_region,dfpolynew)
# import regions
regions_coll = geopandas.read_file('../data/collection_regions_3031.shp').set_geometry('geometry')
regions_coll['id'] = np.linspace(0,len(regions_coll)-1,len(regions_coll)).astype('int')
regions_coll['area'] = regions_coll.area * 1e-6
#%%
# count number of meteorites per region in 2001 (for max and min)
# max
# define year
year = '2001'
# join points and polygons
dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_max_finds)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = regions_coll.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'})

# min
# join points and polygons
dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_min_finds)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}).drop(columns=False)

# calculate number of mets 
mets_region[f'n_min_{year}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
mets_region[f'n_max_{year}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
mets_region[f'average_{year}'] = (mets_region[f'n_max_{year}']+mets_region[f'n_min_{year}'])/2
#%%
# calculate n_meteorites per year for the regions
for yr in range(2002,2082,1):
    # redefine year as string
    year = str(yr)
    # maximum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_max_finds) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'}).drop(columns=False)
    
    #minimum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_min_finds) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}).drop(columns=False)
    
    # estimate number of meteorites
    mets_region[f'n_min_{year}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
    mets_region[f'n_max_{year}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
    mets_region[f'average_{year}'] = (mets_region[f'n_max_{year}']+mets_region[f'n_min_{year}'])/2
    
    # drop unused columns of mets_region
    mets_region = mets_region.drop(columns=[f'n_obs_min_{year}',
                                            f'n_min_{year}',
                                            f'n_obs_max_{year}',
                                            f'n_max_{year}'])
#%%
# calculate percentage with respect to 2020 for every region for every year
for yr in range(2002,2082,1):
    #print(yr)
    mets_region[f'perc_{yr}'] = mets_region[f'average_{yr}']/mets_region['average_2001']

# define function that calculates the first time the percentage of observations drops below PERC precent for a given square
def getfirstyear(mets_region,n_region,PERC):
    # try to get the first year the percentage is below PERC
    try:
        yr = int(mets_region.iloc[n_region][-81:][(mets_region.iloc[n_region][-81:]<PERC)].index[0][-4:])
    # if this year does not occur assign the value of np.nan
    except IndexError:
        yr = np.nan
    return yr
# loop over all squares to get the year that 50% is lost in this square
mets_region['yr_loss'] = [getfirstyear(mets_region,n_region,0.5)+19 for n_region in range(len(mets_region))]
mets_region['mets_per_area'] = mets_region['average_2001']/mets_region['area']

#%%
# export geopandas dataframe as table
sel_cols = mets_region[['Name','area','mets_per_area','yr_loss']]
sel_cols = sel_cols.round({'area':1,'mets_per_area':1})
sel_cols = sel_cols.sort_values('yr_loss')
sel_cols.to_csv('../results/DCAs_loss.csv',index=False)

#%%
# select densest meteorite regions
mets_region_densest = mets_region[mets_region['mets_per_area']>5]
# sort areas
mets_region_sorted = mets_region_densest.sort_values('yr_loss',ascending=True)

#%%
# calculate in which year 50% of meteorites is lost in each dense meteorite collection site region
# FOR LOW EMISSION SCENARIO
# DO NOT exclude locations where meteorites have been found

# transform all_years_sel dataframe into geodataframe, considering the appearance of new meteorites for the max estimate and
# ignoring the appearance of new meteorites for the min estimate
all_years_sel126 = all_years_sel126.reset_index()
mets_yrs_max_finds = all_years_sel126.reset_index(drop=True)
mets_yrs_min_finds = all_years_sel126[all_years_sel126['2001']==True].reset_index(drop=True)


mets_yrs_max_finds['points_max'] = [Point(line[0],line[1]) for line in mets_yrs_max_finds.values]

mets_yrs_min_finds['points_min'] = [Point(line[0],line[1]) for line in mets_yrs_min_finds.values]

mets_gdf_max_finds = geopandas.GeoDataFrame(mets_yrs_max_finds, geometry='points_max')
mets_gdf_min_finds = geopandas.GeoDataFrame(mets_yrs_min_finds, geometry='points_min')

#%%
# import regions
regions_coll = geopandas.read_file('../data/collection_regions_3031.shp').set_geometry('geometry')
regions_coll['id'] = np.linspace(0,len(regions_coll)-1,len(regions_coll)).astype('int')
regions_coll['area'] = regions_coll.area * 1e-6
#%%
# count number of meteorites per region in 2001 (for max and min)
# max
# define year
year = '2001'
# join points and polygons
dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_max_finds)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = regions_coll.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'}).drop(columns=False)


# min
# join points and polygons
dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_min_finds)
# count number of points in polygons
dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
# drop unused columns
dfpivot.columns = dfpivot.columns.droplevel()
# merge counting data with original sqs data
dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
# rename columns 
mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}) #.drop(columns=False)

#%%
# calculate number of mets 
mets_region[f'n_min_{year}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
mets_region[f'n_max_{year}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
mets_region[f'average_{year}'] = (mets_region[f'n_max_{year}']+mets_region[f'n_min_{year}'])/2
#%%
# calculate n_meteorites per year for the regions
for yr in range(2002,2082,1):
    # redefine year as string
    year = str(yr)
    # maximum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_max_finds) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_max_{year}'}).drop(columns=False)
    
    #minimum estimate
    # join points and polygons
    dfsjoin = geopandas.sjoin(regions_coll, mets_gdf_min_finds) #Spatial join Points to polygons
    # count number of points in polygons
    dfpivot = pd.pivot_table(dfsjoin,index='id',columns=year,aggfunc={year:len})
    # drop unused columns
    dfpivot.columns = dfpivot.columns.droplevel()
    # merge counting data with original sqs data
    dfpolynew = mets_region.merge(dfpivot, how='left', on='id')
    # rename columns 
    mets_region = dfpolynew.rename(columns={True:f'n_obs_min_{year}'}).drop(columns=False)
    
    # estimate number of meteorites
    mets_region[f'n_min_{year}'] = [n_mets(n_obs,0.47,0.74,5) for n_obs in mets_region[f'n_obs_min_{year}']]
    mets_region[f'n_max_{year}'] = [n_mets(n_obs,0.81,0.48,5) for n_obs in mets_region[f'n_obs_max_{year}']]
    mets_region[f'average_{year}'] = (mets_region[f'n_max_{year}']+mets_region[f'n_min_{year}'])/2
    
    # drop unused columns of mets_region
    mets_region = mets_region.drop(columns=[f'n_obs_min_{year}',
                                            f'n_min_{year}',
                                            f'n_obs_max_{year}',
                                            f'n_max_{year}'])
#%%
# calculate percentage with respect to 2020 for every region for every year
for yr in range(2002,2082,1):
    #print(yr)
    mets_region[f'perc_{yr}'] = mets_region[f'average_{yr}']/mets_region['average_2001']

# define function that calculates the first time the percentage of observations drops below PERC precent for a given square
def getfirstyear(mets_region,n_region,PERC):
    # try to get the first year the percentage is below PERC
    try:
        yr = int(mets_region.iloc[n_region][-81:][(mets_region.iloc[n_region][-81:]<PERC)].index[0][-4:])
    # if this year does not occur assign the value of np.nan
    except IndexError:
        yr = np.nan
    return yr
# loop over all squares to get the year that 50% is lost in this square
mets_region['yr_loss'] = [getfirstyear(mets_region,n_region,0.5)+19 for n_region in range(len(mets_region))]
mets_region['mets_per_area'] = mets_region['average_2001']/mets_region['area']
#%%
# export geopandas dataframe as table
sel_cols = mets_region[['Name','area','mets_per_area','yr_loss']]
sel_cols = sel_cols.round({'area':1,'mets_per_area':1})
sel_cols = sel_cols.sort_values('yr_loss')
sel_cols.to_csv('../results/DCAs_loss_SSP126.csv',index=False)

#%%
# elevation of predicted meteorites
# open TDX DEM
TDX_ortho_path = r'../data/TDM_merged_ortho.nc'
TDX_ortho_raw = xr.open_rasterio(TDX_ortho_path)

# convert DataArray to DataSet
TDX_ds = TDX_ortho_raw.drop('band')[0].to_dataset(name='DEM')

del(TDX_ortho_raw)

#%%
# here we use different values: mets_gdf_min, mets_gdf_max
dataset_sel = mets_gdf_min
# extract values at meteorite locations in 2020
mets_elevation_2020_min = TDX_ds.interp(x=dataset_sel[dataset_sel['2001']==True].geometry.x.to_xarray(),
                   y=dataset_sel[dataset_sel['2001']==True].geometry.y.to_xarray())

# extract values at meteorite locations in 2100
mets_elevation_2100_min = TDX_ds.interp(x=dataset_sel[dataset_sel['2081']==True].geometry.x.to_xarray(),
                   y=dataset_sel[dataset_sel['2081']==True].geometry.y.to_xarray())

mets_elevation_2020_min_df = mets_elevation_2020_min.to_dataframe()[['x','y','DEM']]
mets_elevation_2100_min_df = mets_elevation_2100_min.to_dataframe()[['x','y','DEM']]


dataset_sel = mets_gdf_max
# extract values at meteorite locations in 2020
mets_elevation_2020_max = TDX_ds.interp(x=dataset_sel[dataset_sel['2001']==True].geometry.x.to_xarray(),
                   y=dataset_sel[dataset_sel['2001']==True].geometry.y.to_xarray())

# extract values at meteorite locations in 2100
mets_elevation_2100_max = TDX_ds.interp(x=dataset_sel[dataset_sel['2081']==True].geometry.x.to_xarray(),
                   y=dataset_sel[dataset_sel['2081']==True].geometry.y.to_xarray())

mets_elevation_2020_max_df = mets_elevation_2020_max.to_dataframe()[['x','y','DEM']]
mets_elevation_2100_max_df = mets_elevation_2100_max.to_dataframe()[['x','y','DEM']]

#%%
# plot histogram
bins = np.linspace(0,4000,21)
# calculate average bins for lower and upper bound
(n_hist_2100_min, bins_2100, patches) = plt.hist(mets_elevation_2100_min_df.DEM, bins=bins, alpha=0.7, label='2100',color=c_MSZ2100,edgecolor='k')
(n_hist_2100_max, bins_2100, patches) = plt.hist(mets_elevation_2100_max_df.DEM, bins=bins, alpha=0.7, label='2100',color=c_MSZ2100,edgecolor='k')
n_hist_2100_average = (n_hist_2100_min + n_hist_2100_max)/2
plt.show()

# font settings
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)

# plot figure
fig = plt.figure(figsize=(8.8/2.54, 8/2.54))
# plot histogram for 2020 (min and max do not differ)
(n_hist_2020_average, bins_2020, patches) = plt.hist(mets_elevation_2020_min_df.DEM, bins=bins, label='2020', color=c_MSZ2020, alpha=0.7,edgecolor='k')
# plot histogram for 2100 (min and max do differ, plot average)
plt.bar(bins[:-1]+100,n_hist_2100_average,200, label='2100',color=c_MSZ2100,edgecolor='k',alpha=0.7)

# plot labels and legend
plt.xlabel('Elevation (m)')
plt.ylabel('Positive classified observations')
plt.legend()

# print values
# print(mets_elevation_2020_df['DEM'].median())
# print(mets_elevation_2100_df['DEM'].median())
# print(mets_elevation_2020_df['DEM'].mean())
# print(mets_elevation_2100_df['DEM'].mean())
# print(mets_elevation_2020_df['DEM'].quantile(0.1))
# print(mets_elevation_2100_df['DEM'].quantile(0.1))

# calculate percentile of meteorites below 1500 m (for min and max estimate in 2100)
perc_min = len(mets_elevation_2020_min_df[mets_elevation_2020_min_df['DEM']<1500])/len(mets_elevation_2020_min_df)
new_threshold_min = mets_elevation_2100_min_df['DEM'].quantile(perc_min)
print(new_threshold_min)

perc_max = len(mets_elevation_2020_max_df[mets_elevation_2020_max_df['DEM']<1500])/len(mets_elevation_2020_max_df)
new_threshold_max = mets_elevation_2100_max_df['DEM'].quantile(perc_max)
print(new_threshold_max)

# take average of new_thresholds
new_threshold_average = (new_threshold_min + new_threshold_max)/2


# plot thresholds
vert_place = 20000
arrowprops=dict(arrowstyle='-', linestyle='--', color='k', linewidth=0.8,alpha=0)
plt.annotate(f'1500 meter corresponds \n to 8th perc. of predicted \n meteorite locations in 2020',
             xy=(1500,vert_place),
             xytext=(800,vert_place),
                                          ha='center',fontsize=7,
             bbox=dict(facecolor=c_MSZ2020, edgecolor='black',
                       linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
             arrowprops=arrowprops) #f'-25%\n{yr_25percloss}'

plt.vlines(1500,0,vert_place-500,linestyles='--',color='k',linewidth=0.8)

vert_place2 = 16000
plt.annotate(f'8th perc. in 2100 \n shifts to ca. 1700 m',
             xy=(1700,vert_place2),
              xytext=(2900,vert_place2),
              ha='center',fontsize=7,
              bbox=dict(facecolor=c_MSZ2100, edgecolor='black',
                        linewidth=0.6,boxstyle='round,pad=0.3,rounding_size=0.8'),
              arrowprops=arrowprops)
plt.vlines(new_threshold_average,0,vert_place2+780,linestyles='--',color='k',linewidth=0.8)
plt.hlines(vert_place2+780,new_threshold_average,2300,linestyles='--',color='k',linewidth=0.8)
plt.ylim([0,23000])

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig('../figures/Elevation_histogram.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)

#%%
# calculate percentage per bin
percentages_per_bin = n_hist_2100_average/n_hist_2020_average
# create dataframe with percentages per bin
perc_per_bin_df = pd.DataFrame(data={'perc_per_bin':percentages_per_bin,
                            'center_bin':bins[:-1]+100})













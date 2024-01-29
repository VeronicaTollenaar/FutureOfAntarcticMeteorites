# ML algorithm that classifies obsrevations as meteorite stranding zone or not (adapted from Tollenaar et al., 2022)
# takes the altered temperature of meteorite finding locations and locations to classify as input

# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

def classifier(savename,altered_stemp_mets,altered_stemp_toclass):
  #%%
  # set threshold/select operating point on ROC curve by maximizing the F-1 score

  # open cross validation data obtained in feature selection
  version = '../data/ROC_average_4pcs4fs0134'
  version_tosave = f'{savename}'
  ROC_average = pd.read_csv(f'{version}.csv') 

  # calculate number of True Positives (TP), True Negatives (TN), 
  # False Positives (FP) and False Negatives (FN)
  n_pos_obs_CV = 2541 # number of positive observations used in cross validation
  n_neg_obs_CV = 8726 # number of negative observations used in cross validation
  TP = ROC_average.TruePositive_average*n_pos_obs_CV
  TN = (1-ROC_average.FalsePositive_average)*n_neg_obs_CV
  FP = ROC_average.FalsePositive_average*n_neg_obs_CV
  FN = (1-ROC_average.TruePositive_average)*n_pos_obs_CV

  # calculate Accuracy (ACC), Precision (PREC), Recall (REC) and F1-score (F1)
  ACC = (TP+TN)/(TP+TN+FP+FN)
  PREC = TP/(TP+FP)
  REC = TP/(TP+FN)
  F1 = 2*TP/(2*TP + FP + FN)
  SPEC = TN/(TN+FP)

  # plot performance measures versus cost
  cost = ROC_average.cost
  imax=480
  plt.plot(cost[:imax],ACC[:imax],label='Accuracy')
  plt.plot(cost[:imax],PREC[:imax],label='Precision')
  plt.plot(cost[:imax],REC[:imax],label='Recall')
  plt.plot(cost[:imax],SPEC[:imax],label='Specificity')
  plt.plot(cost[:imax],F1[:imax],label='F1 Score',color='k')
  # scatter maximum of F1 value
  plt.scatter(cost[F1==F1.max()],F1.max(),color='k',marker='x')
  # plot labels, title, legend
  plt.xlabel('cost')
  plt.ylabel('fraction')
  plt.title('Precision, Recall, and F1 Score')
  plt.legend()
  #plt.savefig('../Figures/PerformanceMeasures_vs_cost.png',dpi=200)
  plt.show()

  cost_sel = ROC_average[F1==F1.max()].cost.values
  cost_idx = F1.idxmax(axis=0)
  #cost_sel=80

  print('accuracy:', ACC[cost_idx])
  print('precision:', PREC[cost_idx])
  print('F1 score:', F1[cost_idx])
  print('recall', REC[cost_idx])
  print('specificity', SPEC[cost_idx])
  print(TP[cost_idx],FN[cost_idx],FP[cost_idx],TN[cost_idx])

  #%%
  # open labelled data
  f1_radar_mets = pd.read_csv('../data/radarbackscatter_at_mets.csv') 
  f2_speed_mets = pd.read_csv('../data/velocities_at_mets.csv') 
  f4_slope_mets = pd.read_csv('../data/slope2km_at_mets.csv') 
  f5_stemp_mets = pd.read_csv('../data/stempPERC99_at_mets.csv') 


  # merge data
  data_mets = f1_radar_mets.merge(
              f2_speed_mets).merge(
              f4_slope_mets).merge(
              f5_stemp_mets)            

  #%%
  # open unlabelled data
  f1_radar_toclass = pd.read_csv('../data/radarbackscatter_at_toclass.csv') 
  f2_speed_toclass = pd.read_csv('../data/velocities_at_toclass.csv')
  f4_slope_toclass = pd.read_csv('../data/slope2km_at_toclass.csv')
  f5_stemp_toclass = pd.read_csv('../data/stempPERC99_at_toclass.csv') 

  # merge data
  data_toclass = f1_radar_toclass.merge(
                  f2_speed_toclass).merge(
                  f4_slope_toclass).merge(
                  f5_stemp_toclass)
                     
  #%%
  # transform features
  data_mets_transf = data_mets.copy()
  data_toclass_transf = data_toclass.copy()

  data_mets_transf['radar']    = data_mets.radar + np.random.RandomState(
                                  3).normal(0,0.25,len(data_mets.radar))
  data_toclass_transf['radar'] = data_toclass.radar + np.random.RandomState(
                                  6).normal(0,0.25,len(data_toclass.radar))

  data_mets_transf['speed']    = np.log10(data_mets.speed.values)
  data_toclass_transf['speed'] = np.log10(data_toclass.speed.values)

  data_mets_transf['slope_max']    = np.log10(data_mets.slope_max.values)
  data_toclass_transf['slope_max'] = np.log10(data_toclass.slope_max.values)

  data_mets_transf['stemp']    = data_mets.stemp + np.random.RandomState(
                                 5).normal(0,0.04,len(data_mets.stemp))
  data_toclass_transf['stemp'] = data_toclass.stemp + np.random.RandomState(
                                 8).normal(0,0.04,len(data_toclass.stemp)) 
  #%%
  # reorganize the data into groups of:
  # - labelled training data
  # - unlabelled training data
  # - all data

  # define positve labelled training data
  train_lab = data_mets_transf[:]


  # define unlabelled training data
  # exclude positive labeled data from unlabeled data
  lab_to_exclude = data_mets_transf.copy()[['x','y']]
  lab_to_exclude['label'] = 1
  lab_to_exclude_merged = data_toclass_transf.merge(lab_to_exclude,
                                                    how='outer',
                                                    on =['x','y'])
  train_unlab = lab_to_exclude_merged[
                np.isnan(lab_to_exclude_merged.label)].drop(
                ['label'],
                axis=1)[:]

  # combine all training data into one array
  train_all = pd.concat([train_lab,train_unlab])

  del(data_mets_transf,
      lab_to_exclude,
      lab_to_exclude_merged,
      data_toclass_transf)
  #%%
  # standardize features
  # use index [:,2:] as the first two columns are the x and y coordinates
  scaler_train = preprocessing.StandardScaler().fit(train_all.iloc[:,2:].values)
  train_lab_st = scaler_train.transform(train_lab.iloc[:,2:].values)
  train_unlab_st = scaler_train.transform(train_unlab.iloc[:,2:].values)
  train_all_st = scaler_train.transform(train_all.iloc[:,2:].values)

  del(train_lab,train_unlab)

  #%%
  # compute principal components
  pca = PCA()
  pca.fit(train_all_st)

  # transform standardized features to principal components
  pcs_train_lab = pca.transform(train_lab_st)
  pcs_train_unlab = pca.transform(train_unlab_st)
  pcs_train_all = pca.transform(train_all_st)

  del(train_lab_st,train_unlab_st,train_all_st)

  #%%
  # kernel density estimation labelled data
  pcmax = 4 # number of principal components

  # reorganize data
  xy_train_lab  = np.array(pcs_train_lab[:,0:pcmax].tolist()).squeeze()

  # define possible bandwiths
  bandwidths = np.linspace(0.3,0.35,30)

  # perform gridsearch (cross-validation) to estimate optimal bandwith
  grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                      {'bandwidth': bandwidths},
                      cv = 10)
  grid.fit(xy_train_lab);

  # print bandwidth to check it is within the limits of the bandwiths
  print(grid.best_params_)
  bw_lab = grid.best_params_['bandwidth']

  # estimate density using the findal bandwidth
  kde_lab = KernelDensity(bandwidth=bw_lab).fit(xy_train_lab)

  #%%
  # kernel density estimation unlabelled data
  # select a random sample of 10.000 from the unlabelled data
  pcs_train_unlab_df = pd.DataFrame(pcs_train_unlab)
  randsample_pcs_train_unlab = pcs_train_unlab_df.sample(
                               10000,random_state=5).values
  # reorganize data
  xy_train_unlab  = np.array(randsample_pcs_train_unlab[:,0:pcmax].tolist()).squeeze()

  # define possible bandwiths
  bandwidths = np.linspace(0.2,0.3,30)

  # perform gridsearch (cross-validation) to estimate optimal bandwith
  grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                      {'bandwidth': bandwidths},
                      cv = 10)
  grid.fit(xy_train_unlab);

  # print bandwidth to check it is within the limits of the bandwiths
  print(grid.best_params_)
  bw_unlab = grid.best_params_['bandwidth']

  # estimate density using the findal bandwidth
  kde_unlab = KernelDensity(bandwidth=bw_unlab).fit(xy_train_unlab)

  #DEFINE ALTERED DATA
  # read in new surface temperature data
  del(data_mets,data_toclass)
  f5_stemp_mets_ALTERED = pd.read_csv(f'{altered_stemp_mets}')

  # merge data
  data_mets = f1_radar_mets.merge(
              f2_speed_mets).merge(
              f4_slope_mets).merge(
              f5_stemp_mets_ALTERED)            

  # open unlabelled data
  f5_stemp_toclass_ALTERED = pd.read_csv(f'{altered_stemp_toclass}')

  # merge data
  data_toclass = f1_radar_toclass.merge(
                  f2_speed_toclass).merge(
                  f4_slope_toclass).merge(
                  f5_stemp_toclass_ALTERED)
                      
  # transform features
  data_mets_transf = data_mets.copy()
  data_toclass_transf = data_toclass.copy()

  data_mets_transf['radar']    = data_mets.radar + np.random.RandomState(
                                  3).normal(0,0.25,len(data_mets.radar))
  data_toclass_transf['radar'] = data_toclass.radar + np.random.RandomState(
                                  6).normal(0,0.25,len(data_toclass.radar))

  data_mets_transf['speed']    = np.log10(data_mets.speed.values)
  data_toclass_transf['speed'] = np.log10(data_toclass.speed.values)

  data_mets_transf['slope_max']    = np.log10(data_mets.slope_max.values)
  data_toclass_transf['slope_max'] = np.log10(data_toclass.slope_max.values)

  data_mets_transf['stemp']    = data_mets.stemp + np.random.RandomState(
                                 5).normal(0,0.04,len(data_mets.stemp))
  data_toclass_transf['stemp'] = data_toclass.stemp + np.random.RandomState(
                                 8).normal(0,0.04,len(data_toclass.stemp))


  # reorganize the data into groups of:
  # - labelled training data
  # - unlabelled training data
  # - all data

  # define positve labelled training data
  classify_lab = data_mets_transf[:]


  # define unlabelled training data
  # exclude positive labeled data from unlabeled data
  lab_to_exclude = data_mets_transf.copy()[['x','y']]
  lab_to_exclude['label'] = 1
  lab_to_exclude_merged = data_toclass_transf.merge(lab_to_exclude,
                                                    how='outer',
                                                    on =['x','y'])
  classify_unlab = lab_to_exclude_merged[
                np.isnan(lab_to_exclude_merged.label)].drop(
                ['label'],
                axis=1)[:]

  # combine all training data into one array
  classify_all = pd.concat([classify_lab,classify_unlab])

  del(data_mets_transf,
      lab_to_exclude,
      lab_to_exclude_merged,
      data_toclass_transf)

  # standardize features
  # use index [:,2:] as the first two columns are the x and y coordinates
  classify_all_st = scaler_train.transform(classify_all.iloc[:,2:].values)


  # transform standardized features to principal components
  pcs_classify_all = pca.transform(classify_all_st)

  del(classify_all_st)

  #%%
  # score all data using kde_lab

  # reorganize data
  xy_toclass = np.array(pcs_classify_all[:,0:pcmax].tolist()).squeeze()
  # score data
  scores_lab = np.exp(kde_lab.score_samples(xy_toclass))

  # score all data using kde_unlab
  # score data
  scores_unlab = np.exp(kde_unlab.score_samples(xy_toclass))


  #%%
  # estimate following probabilities:
      
  # p(x|s=1) = scored_lab
  # p(x|s=0) = scored_unlab 
  # p(s=1)
  # p(s=0)
  # p(s=1|x)
  # p(s=0|x)

  ps1 = len(pcs_train_lab)*cost_sel/(len(pcs_train_lab)*cost_sel + 
                                     len(pcs_train_unlab))
  ps0 = len(pcs_train_unlab)/(len(pcs_train_lab)*cost_sel + 
                                     len(pcs_train_unlab))
  def probs(inp_scores_lab,inp_scores_unlab):
      ps1gx = (inp_scores_lab*ps1)/(inp_scores_lab*ps1 + inp_scores_unlab*ps0)
      ps0gx = (inp_scores_unlab*ps0)/(inp_scores_lab*ps1 + inp_scores_unlab*ps0)
      return ps1gx, ps0gx
  ps1gx_class, ps0gx_class = probs(scores_lab,scores_unlab)

  #%%
  # check how many observations are classified as positive
  positive_classified = ps1gx_class[(ps1gx_class)>0.5]
  print(len(positive_classified))


  #%%
  # save data
  data_classified = classify_all.copy()
  print(len(classify_all))
  print(len(data_classified))
  print(len(((ps1gx_class)>0.5)))
  data_classified['positive_classified']=((ps1gx_class)>0.5)
  data_classified['classification_value'] = (ps1gx_class)
  data_classified.to_csv(f'../results/classification_{savename}.csv')
  return(data_classified)
 







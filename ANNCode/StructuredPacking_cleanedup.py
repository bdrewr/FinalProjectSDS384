# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:21:55 2023

@author: bendr
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:45:17 2023

@author: bendr
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


from sklearn.preprocessing import MinMaxScaler

data_unscaled = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\AE_UTSRP_Filled_2.csv')
data_unscaled = data_unscaled.drop(labels=['Unnamed: 0', 'void fraction', 'Glycerol', '[alk]'], axis='columns')
scaler = MinMaxScaler()
unscaled_columns = ['Packing Type', 'Who?', 'Fractional Area']
columns_to_scale = [i for i in data_unscaled.columns if i not in unscaled_columns]
data = pd.DataFrame(data=scaler.fit_transform(data_unscaled[columns_to_scale]), columns = columns_to_scale)
data[unscaled_columns] = data_unscaled[unscaled_columns]
drop_labels = [
    "CO2 in",
    "CO2 out",
    "k OH E-3",
    "[OH-]",
    "HCO2 E-5",
    'packing element height',
    'DelP',
    'ReL',
    #'WeL E4',
    #'FrL',
]
data_clean_ish = data_unscaled.drop(labels=drop_labels, axis='columns')
randoms = data_clean_ish["Packing Type"].str.startswith("RSR")
data_struct = data_clean_ish.loc[~randoms]
data_clean_struct = data_struct[['specific area','Corrugation angle','L','uG','T Corr','DCO2 E9','Fractional Area']]
columns_to_scale = ['specific area','Corrugation angle','L','uG','T Corr','DCO2 E9','Fractional Area']
data = pd.DataFrame(data=scaler.fit_transform(data_clean_struct[columns_to_scale]), columns = columns_to_scale)
data_inverted = pd.DataFrame(data=scaler.inverse_transform(data[columns_to_scale]), columns=columns_to_scale)


df = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\scaled_data_NoRandomPacking.csv')
df.head()

X_base = df[['specific area','Corrugation angle','S, Channel Side','B, Channel Base','h, Crimp height','L','uG','T Corr','k OH E-3','DCO2 E9','HCO2 E-5','[OH-]']]
Y = df['Fractional Area'].values


X_simple = X_base[['specific area','Corrugation angle','S, Channel Side','B, Channel Base','h, Crimp height','L','uG','T Corr']].values

X_2 = X_base[['specific area','Corrugation angle','S, Channel Side','B, Channel Base','h, Crimp height','L','uG','T Corr','DCO2 E9','HCO2 E-5']].values

X_3 = X_base[['specific area','Corrugation angle','L','uG','T Corr','DCO2 E9']]

X_train, X_test, Y_train, Y_test = train_test_split(X_3, Y, test_size=0.2)

X_b_tr,X_b_te,Y_b_tr,Y_b_te = train_test_split(X_3,Y,test_size = 0.2)


X_reduced2 = X_base[['specific area','L','uG','T Corr']].values
X_r1_tr,X_r1_te,Y_r1_tr,Y_r1_te = train_test_split(X_reduced2,Y,test_size = 0.2)


#%% Defining Our ANN Models here
name = "Neural Network"
clf = MLPRegressor(
                solver="lbfgs",
                activation = 'logistic',
                hidden_layer_sizes=[700, 700],
                alpha=1e-05,
                random_state=1,
                max_iter=10000,
                learning_rate_init = 0.00018
            )



clf_b = MLPRegressor(
                solver="lbfgs",
                activation = 'logistic',
                hidden_layer_sizes=[700, 700],
                alpha=1e-05,
                random_state=1,
                max_iter=10000,
                learning_rate_init = 0.00018
            )

clf_r1 = MLPRegressor(
                solver="lbfgs",
                activation = 'logistic',
                hidden_layer_sizes=[700, 700],
                alpha=1e-05,
                random_state=1,
                max_iter=10000,
                learning_rate_init = 0.00018
            )

#%%SECTION FOR HYPERPARAMETER TUNING

# Hyperparameter tuning
param_grid = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [1e-05, 1e-04, 1e-03],
              'learning_rate_init': [0.001, 0.01, 0.1]}

grid_search = GridSearchCV(MLPRegressor(solver="lbfgs", random_state=1, max_iter=10000), param_grid, cv=5)
grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_

Y_predict_test = best_model.predict(X_test)
Y_predict_train = best_model.predict(X_train)

print("")
print("HYPERPARAMETER SECTION")
print("")
print("R2 for test comparisons is: ", best_model.score(X_test,Y_test))
print("")
print("R2 for training comparisons is: ", best_model.score(X_train,Y_train))

#%% Storage Initialization for metrics, predicted variables, statistical metrics.

optimized_model_array_train = np.zeros((6,50))
r1_array_train = np.zeros((6,50))
base_array_train = np.zeros((6,50))

optimized_model_array_test= np.zeros((6,50))
r1_array_test = np.zeros((6,50))
base_array_test = np.zeros((6,50))

#predicted variable array initalization
Y_te_arr_best_p = np.zeros((len(Y_test),50))
Y_tr_arr_best_p = np.zeros((len(Y_train),50))
Y_te_arr_base_p = np.zeros((len(Y_test),50))
Y_tr_arr_base_p = np.zeros((len(Y_train),50))
Y_te_arr_re_p = np.zeros((len(Y_test),50))
Y_tr_arr_re_p = np.zeros((len(Y_train),50))

#non-predicted array initialization
Y_te_arr_best = np.zeros((len(Y_test),50))
Y_tr_arr_best = np.zeros((len(Y_train),50))
Y_te_arr_base = np.zeros((len(Y_test),50))
Y_tr_arr_base = np.zeros((len(Y_train),50))
Y_te_arr_re = np.zeros((len(Y_test),50))
Y_tr_arr_re = np.zeros((len(Y_train),50))

#%% Section for looping
#This is where I perform multiple test/train splits (n = 50) to see what mean/stdev looks like for the runs.
for i in range(50):
    X_b_tr,X_b_te,Y_b_tr,Y_b_te = train_test_split(X_3, Y, test_size = 0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_3, Y, test_size=0.2)
    X_r1_tr,X_r1_te,Y_r1_tr,Y_r1_te = train_test_split(X_reduced2, Y, test_size = 0.2)

    clf_r1.fit(X_r1_tr, Y_r1_tr)
    clf_b.fit(X_b_tr, Y_b_tr)
    best_model.fit(X_train,Y_train)
    
    Y_predict_test_r1 = clf_r1.predict(X_r1_te)
    Y_predict_train_r1 = clf_r1.predict(X_r1_tr)
    Y_predict_test_b = clf_b.predict(X_b_te)
    Y_predict_train_b = clf_b.predict(X_b_tr)
    Y_predict_test = best_model.predict(X_test)
    Y_predict_train = best_model.predict(X_train)
    
# Training Dataset Metrics
    
    r2_train_b = r2_score(Y_b_tr, Y_predict_train_b)
    mae_train_b = mean_absolute_error(Y_b_tr, Y_predict_train_b)
    rmse_train_b = np.sqrt(mean_squared_error(Y_b_tr, Y_predict_train_b))
    mse_train_b = mean_squared_error(Y_b_tr, Y_predict_train_b)
    explained_variance_train_b = explained_variance_score(Y_b_tr, Y_predict_train_b)
    score_tr_b = clf_b.score(X_b_tr,Y_b_tr)

    r2_train_r1 = r2_score(Y_r1_tr, Y_predict_train_r1)
    mae_train_r1 = mean_absolute_error(Y_r1_tr, Y_predict_train_r1)
    rmse_train_r1 = np.sqrt(mean_squared_error(Y_r1_tr, Y_predict_train_r1))
    mse_train_r1 = mean_squared_error(Y_r1_tr, Y_predict_train_r1)
    explained_variance_train_r1 = explained_variance_score(Y_r1_tr, Y_predict_train_r1)
    score_tr_r1 = clf_r1.score(X_r1_tr,Y_r1_tr)

    r2_train_best = r2_score(Y_train, Y_predict_train)
    mae_train_best = mean_absolute_error(Y_train, Y_predict_train)
    rmse_train_best = np.sqrt(mean_squared_error(Y_train, Y_predict_train))
    mse_train_best = mean_squared_error(Y_train, Y_predict_train)
    explained_variance_train_best = explained_variance_score(Y_train, Y_predict_train)
    score_tr_best = best_model.score(X_train,Y_train)

#Test Dataset Metrics
    r2_test_b = r2_score(Y_b_te, Y_predict_test_b)
    mae_test_b = mean_absolute_error(Y_b_te, Y_predict_test_b)
    rmse_test_b = np.sqrt(mean_squared_error(Y_b_te, Y_predict_test_b))
    mse_test_b = mean_squared_error(Y_b_te, Y_predict_test_b)
    explained_variance_test_b = explained_variance_score(Y_b_te, Y_predict_test_b)
    score_te_b = clf_b.score(X_b_te,Y_b_te)
    
    r2_test_r1 = r2_score(Y_r1_te, Y_predict_test_r1)
    mae_test_r1 = mean_absolute_error(Y_r1_te, Y_predict_test_r1)
    rmse_test_r1 = np.sqrt(mean_squared_error(Y_r1_te, Y_predict_test_r1))
    mse_test_r1 = mean_squared_error(Y_r1_te, Y_predict_test_r1)
    explained_variance_test_r1 = explained_variance_score(Y_r1_te, Y_predict_test_r1)
    score_te_r1 = clf_r1.score(X_r1_te,Y_r1_te)

    r2_test_best = r2_score(Y_test, Y_predict_test)
    mae_test_best = mean_absolute_error(Y_test, Y_predict_test)
    rmse_test_best = np.sqrt(mean_squared_error(Y_test, Y_predict_test))
    mse_test_best = mean_squared_error(Y_test, Y_predict_test)
    explained_variance_test_best = explained_variance_score(Y_test, Y_predict_test)
    score_te_best = best_model.score(X_test,Y_test)
    
#Metric Storage
    optimized_model_array_train[:,i] = [r2_train_best, mae_train_best, rmse_train_best, mse_train_best, explained_variance_train_best, score_tr_best]
    optimized_model_array_test[:,i] = [r2_test_best, mae_test_best, rmse_test_best, mse_test_best, explained_variance_test_best, score_te_best]
    r1_array_train[:,i] = [r2_train_r1, mae_train_r1, rmse_train_r1, mse_train_r1, explained_variance_train_r1, score_tr_r1]
    r1_array_test[:,i] = [r2_test_r1, mae_test_r1, rmse_test_r1, mse_test_r1, explained_variance_test_r1, score_te_r1]
    base_array_train[:,i] = [r2_train_b, mae_train_b, rmse_train_b, mse_train_b, explained_variance_train_b, score_tr_b]
    base_array_test[:,i] = [r2_test_b, mae_test_b, rmse_test_b, mse_test_b, explained_variance_test_b, score_te_b]

#Predicted and Used Y Storage    
    Y_te_arr_best_p[:,i] = Y_predict_test
    Y_tr_arr_best_p[:,i] = Y_predict_train
    Y_te_arr_base_p[:,i] = Y_predict_test_b
    Y_tr_arr_base_p[:,i] = Y_predict_train_b
    Y_te_arr_re_p[:,i] = Y_predict_test_r1
    Y_tr_arr_re_p[:,i] = Y_predict_train_r1

    Y_te_arr_best[:,i] = Y_test
    Y_tr_arr_best[:,i] = Y_train
    Y_te_arr_base[:,i] = Y_b_te
    Y_tr_arr_base[:,i] = Y_b_tr
    Y_te_arr_re[:,i] = Y_r1_te
    Y_tr_arr_re[:,i] = Y_r1_tr
    
#%% Print the results
print("BEST MODEL")
print("")
print(optimized_model_array_train)
print("")
print(optimized_model_array_test)

print("REDUCED MODEL")
print("")
print(r1_array_train)
print("")
print(r1_array_test)

print("BASE MODEL")
print("")
print(base_array_train)
print("")
print(base_array_test)
#print(
#    f"Classification report for classifier {clf}:\n"
#    f"{metrics.classification_report(y_test, predicted)}\n"
#)
be_tr = {
    'r2': optimized_model_array_train[0],
    'mae': optimized_model_array_train[1],
    'rmse': optimized_model_array_train[2],
    'mse': optimized_model_array_train[3],
    'explained_variance': optimized_model_array_train[4],
    'score': optimized_model_array_train[5]
}
be_te = {
    'r2': optimized_model_array_test[0],
    'mae': optimized_model_array_test[1],
    'rmse': optimized_model_array_test[2],
    'mse': optimized_model_array_test[3],
    'explained_variance': optimized_model_array_test[4],
    'score': optimized_model_array_test[5]
}
re_tr = {
    'r2': r1_array_train[0],
    'mae': r1_array_train[1],
    'rmse': r1_array_train[2],
    'mse': r1_array_train[3],
    'explained_variance': r1_array_train[4],
    'score': r1_array_train[5]
}
re_te = {
    'r2': r1_array_test[0],
    'mae': r1_array_test[1],
    'rmse': r1_array_test[2],
    'mse': r1_array_test[3],
    'explained_variance': r1_array_test[4],
    'score': r1_array_test[5]
}
ba_tr = {
    'r2': base_array_train[0],
    'mae': base_array_train[1],
    'rmse': base_array_train[2],
    'mse': base_array_train[3],
    'explained_variance': base_array_train[4],
    'score': base_array_train[5]
}
ba_te = {
    'r2': base_array_test[0],
    'mae': base_array_test[1],
    'rmse': base_array_test[2],
    'mse': base_array_test[3],
    'explained_variance': base_array_test[4],
    'score': base_array_test[5]
}


df_be_tr = pd.DataFrame(be_tr)
df_be_te = pd.DataFrame(be_te)
df_re_tr = pd.DataFrame(re_tr)
df_re_te = pd.DataFrame(re_te)
df_ba_tr = pd.DataFrame(ba_tr)
df_ba_te = pd.DataFrame(ba_te)

mean_be_tr = df_be_tr.mean()
mean_be_te = df_be_te.mean()
mean_re_tr = df_re_tr.mean()
mean_re_te = df_re_te.mean()
mean_ba_tr = df_ba_tr.mean()
mean_ba_te = df_ba_te.mean()

std_be_tr = df_be_tr.std()
std_be_te = df_be_te.std()
std_re_tr = df_re_tr.std()
std_re_te = df_re_te.std()
std_ba_tr = df_ba_tr.std()
std_ba_te = df_ba_te.std()

column_names = ['r2','mae','rmse','mse','explained_variance','score']


results_be_tr_1 = np.array([mean_be_tr,std_be_tr])
results_be_te_1 = np.array([mean_be_te,std_be_te])
results_re_tr_1 = np.array([mean_re_tr,std_re_tr])
results_re_te_1 = np.array([mean_re_te,std_re_te])
results_ba_tr_1 = np.array([mean_ba_tr,std_ba_tr])
results_ba_te_1 = np.array([mean_ba_te,std_ba_te])

results_be_tr = pd.DataFrame(data = results_be_tr_1, index=['mean','std'], columns = column_names)
results_be_te = pd.DataFrame(data = results_be_te_1, index=['mean','std'], columns = column_names)
results_re_tr = pd.DataFrame(data = results_re_tr_1, index=['mean','std'], columns = column_names)
results_re_te = pd.DataFrame(data = results_re_te_1, index=['mean','std'], columns = column_names)
results_ba_tr = pd.DataFrame(data = results_ba_tr_1, index=['mean','std'], columns = column_names)
results_ba_te = pd.DataFrame(data = results_ba_te_1, index=['mean','std'], columns = column_names)

#%%Mean Output Variable Calculation Section
Y_te_arr_base_mean = np.mean(Y_te_arr_base, axis=1)
Y_tr_arr_base_mean = np.mean(Y_tr_arr_base, axis=1)
Y_te_arr_best_mean = np.mean(Y_te_arr_best, axis=1)
Y_tr_arr_best_mean = np.mean(Y_tr_arr_best, axis=1)
Y_te_arr_re_mean = np.mean(Y_te_arr_re, axis=1)
Y_tr_arr_re_mean = np.mean(Y_tr_arr_re, axis=1)

Y_te_arr_base_p_mean = np.mean(Y_te_arr_base_p, axis=1)
Y_tr_arr_base_p_mean = np.mean(Y_tr_arr_base_p, axis=1)
Y_te_arr_best_p_mean = np.mean(Y_te_arr_best_p, axis=1)
Y_tr_arr_best_p_mean = np.mean(Y_tr_arr_best_p, axis=1)
Y_te_arr_re_p_mean = np.mean(Y_te_arr_re_p, axis=1)
Y_tr_arr_re_p_mean = np.mean(Y_tr_arr_re_p, axis=1)

from scipy.stats import linregress
slope_test, intercept_test, r_value_test, p_value_test, std_err_test = linregress(Y_te_arr_base_mean, Y_te_arr_base_p_mean)
slope_train, intercept_train, r_value_train, p_value_train, std_err_train = linregress(Y_tr_arr_base_mean, Y_tr_arr_base_p_mean)
slope_test_best, intercept_test_best, r_value_test_best, p_value_test_best, std_err_test_best = linregress(Y_te_arr_best_mean, Y_te_arr_best_p_mean)
slope_train_best, intercept_train_best, r_value_train_best, p_value_train_best, std_err_train_best = linregress(Y_tr_arr_best_mean, Y_tr_arr_best_p_mean)
slope_test_re, intercept_test_re, r_value_test_re, p_value_test_re, std_err_test_re = linregress(Y_te_arr_re_mean, Y_te_arr_re_p_mean)
slope_train_re, intercept_train_re, r_value_train_re, p_value_train_re, std_err_train_re = linregress(Y_tr_arr_re_mean, Y_tr_arr_re_p_mean)


#%% Real Parity Plots

error_margin = 0.2

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_te_arr_best,Y_te_arr_best_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_te_arr_best), 100), np.linspace(0, np.max(Y_te_arr_best), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_te_arr_best), 100), np.linspace(0, np.max(Y_te_arr_best), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Test Dataset Parity Plot, Best Model')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.text(0.1, 0.94, 'r^2 = 0.99', horizontalalignment='center', verticalalignment='center')

plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_tr_arr_best,Y_tr_arr_best_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_best), 100), np.linspace(0, np.max(Y_tr_arr_best), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_best), 100), np.linspace(0, np.max(Y_tr_arr_best), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Train Dataset Parity Plot, Best Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_te_arr_base,Y_te_arr_base_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_te_arr_base), 100), np.linspace(0, np.max(Y_te_arr_base), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_te_arr_base), 100), np.linspace(0, np.max(Y_te_arr_base), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Test Dataset Parity Plot, Base Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_tr_arr_base,Y_tr_arr_base_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_base), 100), np.linspace(0, np.max(Y_tr_arr_base), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_base), 100), np.linspace(0, np.max(Y_tr_arr_base), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Train Dataset Parity Plot, Base Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_te_arr_re,Y_te_arr_re_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_te_arr_re), 100), np.linspace(0, np.max(Y_te_arr_re), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_te_arr_re), 100), np.linspace(0, np.max(Y_te_arr_re), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Test Dataset Parity Plot, Reduced Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_tr_arr_re,Y_tr_arr_re_p)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_re), 100), np.linspace(0, np.max(Y_tr_arr_re), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_tr_arr_re), 100), np.linspace(0, np.max(Y_tr_arr_re), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Structured Packing Model: Train Dataset Parity Plot, Reduced Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#%%Graphs for comparison of current models to Computed Data (Jorge's Method)
#For Reference on how I've defined the Input Variables to the model......
#X_3 = X_base[['specific area','Corrugation angle','L','uG','T Corr','DCO2 E9']].values

#Fit using best feature combination
#Redo scaling so it can be undone later

best_combo = ['specific area', 'Corrugation angle','L', 'uG', 'T Corr', 'DCO2 E9']
X_best = data_unscaled.loc[~randoms]
X_best = X_best[best_combo]
best_scaler = MinMaxScaler()
X_best = pd.DataFrame(data=best_scaler.fit_transform(X_best), columns=best_combo)


g = 9.8 #m/s^2
rho_L = 1000 #kg/m^3 (water)
sigma = 72 #mN/m (water)
eta = 1 #value for structured packing, stainless steel, pre-loading zone
specific_area = 250 #m^2/m^3
T = 25 #degC
corrugation_angle = 45 #degrees
uG = 1 #m/s
DCO2 = 2.07 #m^2/s (typical value for 25 degC)
params = f'Fixed parameters:\ng={g}\nrho={rho_L}\nsigma={sigma}\neta={eta}\nT={T}\ncorrugation angle={corrugation_angle}\nuG={uG}\nDCO2={DCO2}'

def Wang(a_p, rho_L, sigma, g, u_L):
  a_e = a_p * 1.34 * (rho_L/sigma*g**(1/3)*(u_L/a_p)**(4/3))**0.116
  fractional_area = a_e/a_p
  return fractional_area

def Song(a_p, rho_L, sigma, g, u_L, eta):
  a_e = a_p * 1.16 * (rho_L/sigma*g**(1/2)*u_L*a_p**(-3/2))**0.138
  fractional_area = a_e/a_p
  return fractional_area

u_L = np.linspace(1, 100, 100)
frac_Wang_250 = Wang(specific_area, rho_L, sigma, g, u_L)
frac_Song_250 = Song(specific_area, rho_L, sigma, g, u_L, eta)

X_pred = np.zeros((100,6))
for i in range(100):
  X_pred[i,:] = [specific_area,corrugation_angle, u_L[i], uG, T, DCO2]
X_pred = pd.DataFrame(data=X_pred, columns=best_combo)
frac_ANN_250 = best_model.predict(best_scaler.transform(X_pred))

specific_area = 125
X_pred['specific area'] = specific_area
frac_ANN_125 = best_model.predict(best_scaler.transform(X_pred))
frac_Wang_125 = Wang(specific_area, rho_L, sigma, g, u_L)
frac_Song_125 = Song(specific_area, rho_L, sigma, g, u_L, eta)

T = 27 #degC
specific_area = 500
X_pred['specific area'] = specific_area
X_pred['T Corr'] = T
frac_ANN_500 = best_model.predict(best_scaler.transform(X_pred))
frac_Wang_500 = Wang(specific_area, rho_L, sigma, g, u_L)
frac_Song_500 = Song(specific_area, rho_L, sigma, g, u_L, eta)

fig, axs = plt.subplots(1, 3, dpi=100, figsize=(24,8), sharey=True)


# plot parity plot with line of best fit for training data
axs[1].plot(u_L, frac_Wang_250, '-.', color='C0', label=r'Wang, $a_p=250$')
axs[1].plot(u_L, frac_Song_250, '--', color='C0', label=r'Song, $a_p=250$')
axs[1].plot(u_L, frac_ANN_250, '-', color='C0', label=r'SVM Best Fit, $a_p=250$')
axs[2].plot(u_L, frac_Wang_500, '-.', color='C1', label=r'Wang, $a_p=500$')
axs[2].plot(u_L, frac_Song_500, '--', color='C1', label=r'Song, $a_p=500$')
axs[2].plot(u_L, frac_ANN_500, '-', color='C1', label=r'SVM Best Fit, $a_p=500$')
axs[0].plot(u_L, frac_Wang_125, '-.', color='C2', label=r'Wang, $a_p=125$')
axs[0].plot(u_L, frac_Song_125, '--', color='C2', label=r'Song, $a_p=125$')
axs[0].plot(u_L, frac_ANN_125, '-', color='C2', label=r'SVM Best Fit, $a_p=125$')

plot_250 = data_inverted[data_inverted['specific area']==250.0]
plot_250 = plot_250[plot_250['uG']==1.0]
plot_250 = plot_250[plot_250['Corrugation angle']==45.0]
plot_250 = plot_250[plot_250['DCO2 E9']>2]
plot_250 = plot_250[plot_250['DCO2 E9']<2.2]
axs[1].scatter(plot_250['L'], plot_250['Fractional Area'], color='C0', label='Experimental data, $a_p=250$')

plot_500 = data_inverted[data_inverted['specific area']==500.0]
plot_500 = plot_500[plot_500['uG']==1.0]
plot_500 = plot_500[plot_500['Corrugation angle']==45.0]
plot_500 = plot_500[plot_500['DCO2 E9']>2]
plot_500 = plot_500[plot_500['DCO2 E9']<2.2]
plot_500 = plot_500[plot_500['T Corr']>24]
plot_500 = plot_500[plot_500['T Corr']<30]
axs[2].scatter(plot_500['L'], plot_500['Fractional Area'], color='C1', label='Experimental data, $a_p=500$')

plot_125 = data_inverted[data_inverted['specific area']==125.0]
plot_125 = plot_125[plot_125['uG']==1.0]
plot_125 = plot_125[plot_125['Corrugation angle']==45.0]
plot_125 = plot_125[plot_125['DCO2 E9']>2]
plot_125 = plot_125[plot_125['DCO2 E9']<2.2]
axs[0].scatter(plot_125['L'], plot_125['Fractional Area'], color='C2', label='Experimental data, $a_p=125$')

for i, ax in enumerate(axs):
  axs[i].set_box_aspect(1)
  axs[i].set_xscale('log')
  axs[i].set_yscale('log')
  axs[i].set_ylabel(r'log($\Phi$)')
  axs[i].set_xlabel(r'log$(u_L)$ [$m^3/m^2h$]')
  axs[i].set_xlim([1,100])
  #axs.set_ylim([0,2])
  axs[i].legend()

axs[1].text(x=1.2,y=1,s=params)
axs[2].text(x=1.2,y=1,s='NOTE: T=27')
plt.suptitle(r'Fractional area ($\Phi$) vs liquid flux ($u_L$)')

plt.tight_layout()
plt.show()
#%%Aggregate Section where I predict the Other Data based on UTSRP Model

df_agg = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\UTSRP_ST_S_scaled.csv')
#df_agg_2 = df_agg.drop(df_agg[df_agg['Who?'] == 'Tsai' or df_agg['Who?'] == 'Wang' or df_agg['Who?'] == 'Song'])
df_agg = df_agg[~df_agg['Who?'].isin(['Tsai', 'Wang', 'Song'])]
#train on L, uG, and Speciifc Area for Song etc and see if we can predict this data.

X_agg_UTSRP = X_base[['specific area','L','uG']].values


X_agg_NotUT = df_agg[['specific area','L','uG']].values
Y_agg_NotUT = df_agg['Fractional Area'].values

#test/train split for UTSRP data
X_tr_aggU, X_te_aggU, Y_tr_aggU, Y_te_aggU = train_test_split(X_agg_UTSRP, Y, test_size=0.2)

grid_search_agg = GridSearchCV(MLPRegressor(solver="lbfgs", random_state=1, max_iter=10000), param_grid, cv=5)
grid_search_agg.fit(X_tr_aggU, Y_tr_aggU)

best_model_agg = grid_search_agg.best_estimator_

Y_te_aggU_predicted = best_model_agg.predict(X_te_aggU)
Y_tr_aggU_predicted = best_model_agg.predict(X_tr_aggU)


#%%
print("HYPERPARAMETER SECTION for UTSRP - NonUTSRP data....")
print("")
print("R2 for test comparisons on UTSRP is: ", best_model_agg.score(X_te_aggU,Y_te_aggU))
print("")
print("R2 for training comparisons on UTSRP is: ", best_model_agg.score(X_tr_aggU,Y_tr_aggU))


#%%
Y_NotUT_predicted = best_model_agg.predict(X_agg_NotUT)
#r2_NonUT = r2_score(Y_agg_NotUT, Y_NotUT_predicted)
r2_NonUT = best_model_agg.score(X_agg_NotUT,Y_agg_NotUT)
error_margin = 0.2

r2_NonUT = '%.3f'%r2_NonUT
plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_agg_NotUT,Y_NotUT_predicted)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
plt.plot(np.linspace(0, np.max(Y_agg_NotUT), 100), np.linspace(0, np.max(Y_agg_NotUT), 100)*(1+error_margin), 'r--')
plt.plot(np.linspace(0, np.max(Y_agg_NotUT), 100), np.linspace(0, np.max(Y_agg_NotUT), 100)*(1-error_margin), 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction of Non-UTSRP Data using "Best" Model Trained on UTSRP Data')
plt.xlim(0,1)
plt.ylim(0,1)
plt.text(0.1, 0.94, 'r^2 = %s' %r2_NonUT, horizontalalignment='center', verticalalignment='center')

#%% PDP Section
from sklearn.inspection import PartialDependenceDisplay
#For the Structured Packing, Best Model....
features_1 = [0,1,2]
features_1_2 = [3,4,5]
features_2way_1 = [(0,1),(0,2)]
features_2way_2 = [(0,3),(2,3)]

PDP_Single_1 = PartialDependenceDisplay.from_estimator(best_model,X_3,features_1,kind = 'both',ice_lines_kw = {"color": "orange"},line_kw = {"color" : "black"})
PDP_Single_2 = PartialDependenceDisplay.from_estimator(best_model,X_3,features_1_2, kind = 'both',ice_lines_kw = {"color": "orange"},line_kw = {"color" : "black"})

PDP_Double_1 = PartialDependenceDisplay.from_estimator(best_model,X_3,features_2way_1)
PDP_Double_2 = PartialDependenceDisplay.from_estimator(best_model,X_3,features_2way_2)

fig1, (ax1,ax2) = plt.subplots(2,1,figsize=(8,8))
PDP_Single_1.plot(ax=ax1,ice_lines_kw = {"color": "orange"},line_kw = {"color" : "black"})
PDP_Single_2.plot(ax=ax2,ice_lines_kw = {"color": "orange"},line_kw = {"color" : "black"})
fig2, (ax3,ax4) = plt.subplots(2,1,figsize=(8,8))
PDP_Double_1.plot(ax=ax3)
PDP_Double_2.plot(ax=ax4)
plt.show()
#%%SHAP SECTION
features =  ['specific area', 'Corrugation angle','L', 'uG', 'T Corr', 'DCO2 E9']
import shap
explainer = shap.KernelExplainer(best_model.predict,X_train)
shap_values = explainer.shap_values(X_test)
# summarize the effects of all the features
shap.summary_plot(shap_values,X_test,feature_names =features)
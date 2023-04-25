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


df = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\UTSRP_ST_S_scaled.csv')
df.head()

X_base = df[['Height','specific area','L','uG']].values
Y = df['Fractional Area'].values

X_train, X_test, Y_train, Y_test = train_test_split(X_base, Y, test_size=0.2)

#name = "Neural Network"
base_model = MLPRegressor(
                solver="lbfgs",
                activation = 'logistic',
                hidden_layer_sizes=[700, 700],
                alpha=1e-05,
                random_state=1,
                max_iter=10000,
                learning_rate_init = 0.00018
            )


base_model.fit(X_train, Y_train)
Y_predict_test = base_model.predict(X_test)
Y_predict_train = base_model.predict(X_train)

print("R2 for test comparisons is: ", base_model.score(X_test,Y_test))
print("")
print("R2 for training comparisons is: ", base_model.score(X_train,Y_train))


#SECTION FOR HYPERPARAMETER TUNING

# Hyperparameter tuning
param_grid = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'alpha': [1e-05, 1e-04, 1e-03],
              'learning_rate_init': [0.001, 0.01, 0.1]}

grid_search = GridSearchCV(MLPRegressor(solver="lbfgs", random_state=1, max_iter=10000), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
#
best_model = grid_search.best_estimator_
#
Y_predict_test = best_model.predict(X_test)
Y_predict_train = best_model.predict(X_train)
#
print("")
print("HYPERPARAMETER SECTION")
print("")
print("R2 for test comparisons is: ", best_model.score(X_test,Y_test))
print("")
print("R2 for training comparisons is: ", best_model.score(X_train,Y_train))
#
#This is where I perform multiple test/train splits (n = 50) to see what mean/stdev looks like for the runs.

base_array_train = np.zeros((6,50))
base_array_test = np.zeros((6,50))
optimized_model_array_train = np.zeros((6,50))
optimized_model_array_test= np.zeros((6,50))
Y_te_arr_best_p = np.zeros((len(Y_test),50))
Y_tr_arr_best_p = np.zeros((len(Y_train),50))
Y_te_arr_base_p = np.zeros((len(Y_test),50))
Y_tr_arr_base_p = np.zeros((len(Y_train),50))
Y_te_arr_best = np.zeros((len(Y_test),50))
Y_tr_arr_best = np.zeros((len(Y_train),50))
Y_te_arr_base = np.zeros((len(Y_test),50))
Y_tr_arr_base = np.zeros((len(Y_train),50))

for i in range(50):
    X_train, X_test, Y_train, Y_test = train_test_split(X_base, Y, test_size=0.2)

    base_model.fit(X_train,Y_train)
    best_model.fit(X_train,Y_train)
    
    Y_predict_test_base = base_model.predict(X_test)
    Y_predict_train_base = base_model.predict(X_train)
    Y_predict_test = best_model.predict(X_test)
    Y_predict_train = best_model.predict(X_train)    
# Training Dataset Metrics
    

    r2_train_base = r2_score(Y_train, Y_predict_train_base)
    mae_train_base = mean_absolute_error(Y_train, Y_predict_train_base)
    rmse_train_base = np.sqrt(mean_squared_error(Y_train, Y_predict_train_base))
    mse_train_base = mean_squared_error(Y_train, Y_predict_train_base)
    explained_variance_train_base = explained_variance_score(Y_train, Y_predict_train_base)
    score_tr_base = base_model.score(X_train,Y_train)

    r2_train_best = r2_score(Y_train, Y_predict_train)
    mae_train_best = mean_absolute_error(Y_train, Y_predict_train)
    rmse_train_best = np.sqrt(mean_squared_error(Y_train, Y_predict_train))
    mse_train_best = mean_squared_error(Y_train, Y_predict_train)
    explained_variance_train_best = explained_variance_score(Y_train, Y_predict_train)
    score_tr_best = best_model.score(X_train,Y_train)
#Test Dataset Metrics

    r2_test_base = r2_score(Y_test, Y_predict_test_base)
    mae_test_base = mean_absolute_error(Y_test, Y_predict_test_base)
    rmse_test_base = np.sqrt(mean_squared_error(Y_test, Y_predict_test_base))
    mse_test_base = mean_squared_error(Y_test, Y_predict_test_base)
    explained_variance_test_base = explained_variance_score(Y_test, Y_predict_test_base)
    score_te_base = base_model.score(X_test,Y_test)

    r2_test_best = r2_score(Y_test, Y_predict_test)
    mae_test_best = mean_absolute_error(Y_test, Y_predict_test)
    rmse_test_best = np.sqrt(mean_squared_error(Y_test, Y_predict_test))
    mse_test_best = mean_squared_error(Y_test, Y_predict_test)
    explained_variance_test_best = explained_variance_score(Y_test, Y_predict_test)
    score_te_best = best_model.score(X_test,Y_test)    
#Metric Storage
    base_array_train[:,i] = [r2_train_base, mae_train_base, rmse_train_base, mse_train_base, explained_variance_train_base, score_tr_base]
    base_array_test[:,i] = [r2_test_base, mae_test_base, rmse_test_base, mse_test_base, explained_variance_test_base, score_te_base]
    
    optimized_model_array_train[:,i] = [r2_train_best, mae_train_best, rmse_train_best, mse_train_best, explained_variance_train_best, score_tr_best]
    optimized_model_array_test[:,i] = [r2_test_best, mae_test_best, rmse_test_best, mse_test_best, explained_variance_test_best, score_te_best]

#Predicted and Used Y Storage    
    Y_te_arr_best_p[:,i] = Y_predict_test
    Y_tr_arr_best_p[:,i] = Y_predict_train
    Y_te_arr_base_p[:,i] = Y_predict_test_base
    Y_tr_arr_base_p[:,i] = Y_predict_train_base

    Y_te_arr_best[:,i] = Y_test
    Y_tr_arr_best[:,i] = Y_train
    Y_te_arr_base[:,i] = Y_test
    Y_tr_arr_base[:,i] = Y_train    

print("BASE MODEL")
print("")
print(base_array_train)
print("")
print(base_array_test)
#print(
#    f"Classification report for classifier {clf}:\n"
#    f"{metrics.classification_report(y_test, predicted)}\n"
#)


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

df_ba_tr = pd.DataFrame(ba_tr)
df_ba_te = pd.DataFrame(ba_te)
df_be_tr = pd.DataFrame(be_tr)
df_be_te = pd.DataFrame(be_te)

mean_ba_tr = df_ba_tr.mean()
mean_ba_te = df_ba_te.mean()
mean_be_tr = df_be_tr.mean()
mean_be_te = df_be_te.mean()

std_ba_tr = df_ba_tr.std()
std_ba_te = df_ba_te.std()
std_be_tr = df_be_tr.std()
std_be_te = df_be_te.std()

column_names = ['r2','mae','rmse','mse','explained_variance','score']


results_ba_tr_1 = np.array([mean_ba_tr,std_ba_tr])
results_ba_te_1 = np.array([mean_ba_te,std_ba_te])
results_be_tr_1 = np.array([mean_be_tr,std_be_tr])
results_be_te_1 = np.array([mean_be_te,std_be_te])

results_ba_tr = pd.DataFrame(data = results_ba_tr_1, index=['mean','std'], columns = column_names)
results_ba_te = pd.DataFrame(data = results_ba_te_1, index=['mean','std'], columns = column_names)
results_be_tr = pd.DataFrame(data = results_be_tr_1, index=['mean','std'], columns = column_names)
results_be_te = pd.DataFrame(data = results_be_te_1, index=['mean','std'], columns = column_names)

#parity plot section
Y_te_arr_base_mean = np.mean(Y_te_arr_base, axis=1)
Y_tr_arr_base_mean = np.mean(Y_tr_arr_base, axis=1)
Y_te_arr_best_mean = np.mean(Y_te_arr_best, axis=1)
Y_tr_arr_best_mean = np.mean(Y_tr_arr_best, axis=1)

Y_te_arr_base_p_mean = np.mean(Y_te_arr_base_p, axis=1)
Y_tr_arr_base_p_mean = np.mean(Y_tr_arr_base_p, axis=1)
Y_te_arr_best_p_mean = np.mean(Y_te_arr_best_p, axis=1)
Y_tr_arr_best_p_mean = np.mean(Y_tr_arr_best_p, axis=1)

from scipy.stats import linregress
slope_test, intercept_test, r_value_test, p_value_test, std_err_test = linregress(Y_te_arr_base_mean, Y_te_arr_base_p_mean)
slope_train, intercept_train, r_value_train, p_value_train, std_err_train = linregress(Y_tr_arr_base_mean, Y_tr_arr_base_p_mean)
slope_test_best, intercept_test_best, r_value_test_best, p_value_test_best, std_err_test_best = linregress(Y_te_arr_best_mean, Y_te_arr_best_p_mean)
slope_train_best, intercept_train_best, r_value_train_best, p_value_train_best, std_err_train_best = linregress(Y_tr_arr_best_mean, Y_tr_arr_best_p_mean)

#plotting of parity plots
plt.scatter(Y_te_arr_base_mean,Y_te_arr_base_p_mean)
plt.plot(Y_te_arr_base_mean,slope_test*Y_te_arr_base_mean + intercept_test, 'black')
plt.plot(Y_te_arr_base_mean,slope_test*Y_te_arr_base_mean + intercept_test + 1.96*std_err_test,color = 'black', linestyle = (0, (5, 10)))
plt.plot(Y_te_arr_base_mean,slope_test*Y_te_arr_base_mean + intercept_test - 1.96*std_err_test,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Structured Packing Model: Test Dataset Parity Plot, Base Model')
plt.show()

plt.scatter(Y_tr_arr_base_mean,Y_tr_arr_base_p_mean)
plt.plot(Y_tr_arr_base_mean,slope_train*Y_tr_arr_base_mean + intercept_train, 'black')
plt.plot(Y_tr_arr_base_mean,slope_train*Y_tr_arr_base_mean + intercept_train + 1.96*std_err_train,color = 'black', linestyle = (0, (5, 10)))
plt.plot(Y_tr_arr_base_mean,slope_train*Y_tr_arr_base_mean + intercept_train - 1.96*std_err_train,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Structured Packing Model: Train Dataset Parity Plot, Base Model')
plt.show()

####
plt.scatter(Y_te_arr_best_mean,Y_te_arr_best_p_mean)
plt.plot(Y_te_arr_best_mean,slope_test_best*Y_te_arr_best_mean + intercept_test_best, 'black')
plt.plot(Y_te_arr_best_mean,slope_test_best*Y_te_arr_best_mean + intercept_test_best + 1.96*std_err_test_best,color = 'black', linestyle = (0, (5, 10)))
plt.plot(Y_te_arr_best_mean,slope_test_best*Y_te_arr_best_mean + intercept_test_best - 1.96*std_err_test_best,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Structured Packing Model: Test Dataset Parity Plot, Best Model')
plt.show()

plt.scatter(Y_tr_arr_best_mean,Y_tr_arr_best_p_mean)
plt.plot(Y_tr_arr_best_mean,slope_train_best*Y_tr_arr_best_mean + intercept_train_best, 'black')
plt.plot(Y_tr_arr_best_mean,slope_train_best*Y_tr_arr_best_mean + intercept_train_best + 1.96*std_err_train_best,color = 'black', linestyle = (0, (5, 10)))
plt.plot(Y_tr_arr_best_mean,slope_train_best*Y_tr_arr_best_mean + intercept_train_best - 1.96*std_err_train_best,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Structured Packing Model: Train Dataset Parity Plot, Best Model')
plt.show()

#%%
#feature importance section
from sklearn.inspection import permutation_importance

result_base = permutation_importance(base_model, X_base, Y, n_repeats=30, random_state=0)
importance_scores = result_base.importances_mean
print("Base Model Feature Importance")
for i, score in enumerate(importance_scores):
    print(f"Feature {i}: {score}")

result_best = permutation_importance(best_model, X_base, Y, n_repeats=30, random_state=0)
importance_scores_best = result_best.importances_mean
print("Best Model Feature Importance")
for i, score in enumerate(importance_scores_best):
    print(f"Feature {i}: {score}")

#Thing at end that I forgot about previosuly
print("")
print("Characteristics of Best Model")
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
print("Best model:", best_model)

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

#%%Proper Graphs for everyting else
slope_store_best_test = np.zeros(50)
int_store_best_test = np.zeros(50)
std_err_store_best_test = np.zeros(50)

slope_store_best_train = np.zeros(50)
int_store_best_train = np.zeros(50)
std_err_store_best_train = np.zeros(50)

slope_store_base_test = np.zeros(50)
int_store_base_test = np.zeros(50)
std_err_store_base_test = np.zeros(50)

slope_store_base_train = np.zeros(50)
int_store_base_train = np.zeros(50)
std_err_store_base_train = np.zeros(50)

for iteration in range(50):
  slope_store_best_test_i, int_store_best_test_i, r_value_best_test_iteration, p_value_best_test_iteration, std_err_store_best_test_i = linregress(Y_te_arr_best[:,iteration], Y_te_arr_best_p[:,iteration])
  slope_store_best_train_i, int_store_best_train_i, r_value_best_train_iteration, p_value_best_train_iteration, std_err_store_best_train_i = linregress(Y_tr_arr_best[:,iteration], Y_tr_arr_best_p[:,iteration])
  slope_store_base_test_i, int_store_base_test_i, r_value_base_test_iteration, p_value_base_test_iteration, std_err_store_base_test_i = linregress(Y_te_arr_base[:,iteration], Y_te_arr_base_p[:,iteration])
  slope_store_base_train_i, int_store_base_train_i, r_value_base_train_iteration, p_value_base_train_iteration, std_err_store_base_train_i = linregress(Y_tr_arr_base[:,iteration], Y_tr_arr_base_p[:,iteration])

  slope_store_best_test[iteration] = slope_store_best_test_i
  int_store_best_test[iteration] = int_store_best_test_i
  std_err_store_best_test[iteration] = std_err_store_best_test_i
  
  slope_store_best_train[iteration] = slope_store_best_train_i
  int_store_best_train[iteration] = int_store_best_train_i
  std_err_store_best_train[iteration] = std_err_store_best_train_i
  
  slope_store_base_test[iteration] = slope_store_base_test_i
  int_store_base_test[iteration] = int_store_base_train_i
  std_err_store_base_test[iteration] = std_err_store_base_test_i
  
  slope_store_base_train[iteration] = slope_store_base_train_i
  int_store_base_train[iteration] = int_store_base_train_i
  std_err_store_base_test[iteration] = std_err_store_base_train_i

slope_mean_be_te = np.mean(slope_store_best_test)
int_mean_be_te = np.mean(int_store_best_test)
std_err_mean_be_te = np.mean(std_err_store_best_test)

slope_mean_be_tr = np.mean(slope_store_best_train)
int_mean_be_tr = np.mean(int_store_best_train)
std_err_mean_be_tr = np.mean(std_err_store_best_train)

slope_mean_ba_te = np.mean(slope_store_base_test)
int_mean_ba_te = np.mean(int_store_base_test)
std_err_mean_ba_te = np.mean(std_err_store_base_test)

slope_mean_ba_tr = np.mean(slope_store_base_train)
int_mean_ba_tr = np.mean(int_store_base_train)
std_err_mean_ba_tr = np.mean(std_err_store_base_test)

error_margin = 0.2

#plotting time :^)

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_te_arr_best,Y_te_arr_best_p)
plt.plot(Y_te_arr_best,slope_mean_be_te*Y_te_arr_best + int_mean_be_te, 'black')
plt.plot(np.linspace(0, np.max(Y_te_arr_best), 100),slope_mean_be_te*np.linspace(0, np.max(Y_te_arr_best), 100) + int_mean_be_te + 1.96*std_err_mean_be_te,color = 'black', linestyle = (0, (5, 10)))
plt.plot(np.linspace(0, np.max(Y_te_arr_best), 100),slope_mean_be_te*np.linspace(0, np.max(Y_te_arr_best), 100) + int_mean_be_te - 1.96*std_err_mean_be_te,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Aggregate Packing Model: Test Dataset Parity Plot, Best Model')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.text(0.1, 0.94, 'r^2 = 0.99', horizontalalignment='center', verticalalignment='center')

plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_tr_arr_best,Y_tr_arr_best_p)
plt.plot(Y_tr_arr_best,slope_mean_be_tr*Y_tr_arr_best + int_mean_be_tr, 'black')
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean + 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean - 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#alt lines with the linspace function
plt.plot(np.linspace(0, np.max(Y_tr_arr_best), 100),slope_mean_be_tr*np.linspace(0, np.max(Y_tr_arr_best), 100) + int_mean_be_tr + 1.96*std_err_mean_be_tr,color = 'black', linestyle = (0, (5, 10)))
plt.plot(np.linspace(0, np.max(Y_tr_arr_best), 100),slope_mean_be_tr*np.linspace(0, np.max(Y_tr_arr_best), 100) + int_mean_be_tr - 1.96*std_err_mean_be_tr,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Aggregate Packing Model: Train Dataset Parity Plot, Best Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_te_arr_base,Y_te_arr_base_p)
plt.plot(Y_te_arr_base,slope_mean_ba_te*Y_te_arr_base + int_mean_ba_te, 'black')
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean + 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean - 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#alt lines with the linspace function
plt.plot(np.linspace(0, np.max(Y_te_arr_base), 100),slope_mean_ba_te*np.linspace(0, np.max(Y_te_arr_base), 100) + int_mean_ba_te + 1.96*std_err_mean_ba_te,color = 'black', linestyle = (0, (5, 10)))
plt.plot(np.linspace(0, np.max(Y_te_arr_base), 100),slope_mean_ba_te*np.linspace(0, np.max(Y_te_arr_base), 100) + int_mean_ba_te - 1.96*std_err_mean_ba_te,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Aggregate Packing Model: Test Dataset Parity Plot, Base Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (6,4),dpi = 400)
plt.scatter(Y_tr_arr_base,Y_tr_arr_base_p)
plt.plot(Y_tr_arr_base,slope_mean_ba_tr*Y_tr_arr_base + int_mean_ba_tr, 'black')
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean + 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#plt.plot(Y_te_arr_base,slope_mean*Y_te_arr_base + int_mean - 1.96*std_err_mean,color = 'black', linestyle = (0, (5, 10)))
#alt lines with the linspace function
plt.plot(np.linspace(0, np.max(Y_tr_arr_base), 100),slope_mean_ba_tr*np.linspace(0, np.max(Y_tr_arr_base), 100) + int_mean_ba_tr + 1.96*std_err_mean_ba_tr,color = 'black', linestyle = (0, (5, 10)))
plt.plot(np.linspace(0, np.max(Y_tr_arr_base), 100),slope_mean_ba_tr*np.linspace(0, np.max(Y_tr_arr_base), 100) + int_mean_ba_tr - 1.96*std_err_mean_ba_tr,color = 'black', linestyle = (0, (5, 10)))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.title('Aggregate Packing Model: Train Dataset Parity Plot, Base Model')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

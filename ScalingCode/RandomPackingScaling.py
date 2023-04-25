# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:23:50 2023

@author: bendr
"""

import pandas as pd
import sklearn as skl


DF = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\AE_UTSRP_RandomOnly.csv',skip_blank_lines=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
columns_to_scale = ['Height','specific area','void fraction','L','uG','T Corr','DelP','k OH E-3','DCO2 E9','HCO2 E-5','[OH-]','CO2 in','CO2 out','Fractional Area','ReL','WeL E4','FrL']
scaled_data = DF
scaled_data[columns_to_scale]=scaler.fit_transform(scaled_data[columns_to_scale])
scaled_data.head()
scaled_data.to_csv('scaled_data_RandomOnly.csv')
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:55:14 2023

@author: Benjamin Drewry
"""

import pandas as pd
import sklearn as skl


DF = pd.read_csv(r'C:\Users\bendr\Box\SDS 384 Project\UTSRP_Database_Ae_formatting.csv',skip_blank_lines=True)
#header = [0,1]
print(DF)
#NOTE: B,S, h, are all going to be in units of meters.

for index, row in DF.iterrows():
    if row['Packing Type'] == 'M250Y':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.16
        DF.at[index, 'B, Channel Base'] = 0.024
        DF.at[index, 'h, Crimp height'] = 0.011
        DF.at[index, 'packing element height'] = 242.4242   
        DF.at[index, 'LP/A'] = 242424.24
    elif row['Packing Type'] == 'M500Y':
        DF.at[index, 'specific area'] = 500.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 8.1
        DF.at[index, 'B, Channel Base'] = 9.6
        DF.at[index, 'h, Crimp height'] = 6.53
        DF.at[index, 'packing element height'] = 516.8453
        DF.at[index, 'LP/A'] = 516.85
    elif row['Packing Type'] == 'M250X':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 60.0
        DF.at[index, 'S, Channel Side'] = 0.017
        DF.at[index, 'B, Channel Base'] = 0.0241
        DF.at[index, 'h, Crimp height'] = 0.0119
        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 237.11
    elif row['Packing Type'] == 'M252Y':
        DF.at[index, 'specific area'] = 252.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .017
        DF.at[index, 'B, Channel Base'] = .0241
        DF.at[index, 'h, Crimp height'] = .0119
        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 237.11       
    elif row['Packing Type'] == 'M250YS':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .0170
        DF.at[index, 'B, Channel Base'] = .0241
        DF.at[index, 'h, Crimp height'] = .0119
        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 237.11             
    elif row['Packing Type'] == 'M125Y':
        DF.at[index, 'specific area'] = 125.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .0370
        DF.at[index, 'B, Channel Base'] = .0550
        DF.at[index, 'h, Crimp height'] = .0248
#        DF.at[index, 'packing element height'] = 0
        DF.at[index, 'LP/A'] = 108.5   
    elif row['Packing Type'] == 'M2Y':
        DF.at[index, 'specific area'] = 205.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .0215
        DF.at[index, 'B, Channel Base'] = .033
        DF.at[index, 'h, Crimp height'] = .0138
#        DF.at[index, 'packing element height'] = 
        DF.at[index, 'LP/A'] = 188.845
    elif row['Packing Type'] == 'F1Y':
        DF.at[index, 'specific area'] = 410.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .009
        DF.at[index, 'B, Channel Base'] = .0127
        DF.at[index, 'h, Crimp height'] = .064
#        DF.at[index, 'packing element height'] = 442.9134
        DF.at[index, 'LP/A'] = 442.91
    elif row['Packing Type'] == 'P500':
        DF.at[index, 'specific area'] = 500.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .0081
        DF.at[index, 'B, Channel Base'] = .0096
        DF.at[index, 'h, Crimp height'] = .00653
#        DF.at[index, 'packing element height'] = 516.8453
        DF.at[index, 'LP/A'] = 516.85
    elif row['Packing Type'] == 'M2X':
        DF.at[index, 'specific area'] = 205.0
        DF.at[index, 'Corrugation angle'] = 60.0
        DF.at[index, 'S, Channel Side'] = .0190
        DF.at[index, 'B, Channel Base'] = 0.026
        DF.at[index, 'h, Crimp height'] = 0.0014
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 208.791
    elif row['Packing Type'] == 'RSP250Y':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.013
        DF.at[index, 'B, Channel Base'] = 0.019
        DF.at[index, 'h, Crimp height'] = 0.008
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 342.105
    elif row['Packing Type'] == 'GTC350Z':
        DF.at[index, 'specific area'] = 350.0
        DF.at[index, 'Corrugation angle'] = 70.0
        DF.at[index, 'S, Channel Side'] = 0.011
        DF.at[index, 'B, Channel Base'] = 0.016
        DF.at[index, 'h, Crimp height'] = 0.01
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 343.75
    elif row['Packing Type'] == 'A350Y':
        DF.at[index, 'specific area'] = 350.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = .013
        DF.at[index, 'B, Channel Base'] = .016
        DF.at[index, 'h, Crimp height'] = .01
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 325
    elif row['Packing Type'] == 'B350X':
        DF.at[index, 'specific area'] = 350.0
        DF.at[index, 'Corrugation angle'] = 60.0
        DF.at[index, 'S, Channel Side'] = .011
        DF.at[index, 'B, Channel Base'] = .016
        DF.at[index, 'h, Crimp height'] = .009
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 305.556
    elif row['Packing Type'] == 'GTC350Y':
        DF.at[index, 'specific area'] = 350.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.013
        DF.at[index, 'B, Channel Base'] = 0.016
        DF.at[index, 'h, Crimp height'] = 0.01
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 325
    elif row['Packing Type'] == 'GTC 500Y':
        DF.at[index, 'specific area'] = 500.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.008
        DF.at[index, 'B, Channel Base'] = 0.01
        DF.at[index, 'h, Crimp height'] = 0.006
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 533.33
    elif row['Packing Type'] == 'RSP 200X':
        DF.at[index, 'specific area'] = 200.0
        DF.at[index, 'Corrugation angle'] = 60.0
        DF.at[index, 'S, Channel Side'] = 0.015
        DF.at[index, 'B, Channel Base'] = 0.027
        DF.at[index, 'h, Crimp height'] = 0.005
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 444.444
    elif row['Packing Type'] == 'RSR 0.5':
        DF.at[index, 'specific area'] = 250.0
#        DF.at[index, 'Corrugation angle'] = 60.0
#        DF.at[index, 'S, Channel Side'] = 0.011
#        DF.at[index, 'B, Channel Base'] = 0.0175
#        DF.at[index, 'h, Crimp height'] = 0.0079
#        DF.at[index, 'packing element height'] = 237.1073
#        DF.at[index, 'LP/A'] = 237.11
    elif row['Packing Type'] == 'RSR 0.7':
        DF.at[index, 'specific area'] = 180.0
#        DF.at[index, 'Corrugation angle'] = 60.0
#        DF.at[index, 'S, Channel Side'] = 0.011
#        DF.at[index, 'B, Channel Base'] = 0.0175
#        DF.at[index, 'h, Crimp height'] = 0.0079
#        DF.at[index, 'packing element height'] = 237.1073
#        DF.at[index, 'LP/A'] = 237.11
    elif row['Packing Type'] == 'RSR 0.3':
        DF.at[index, 'specific area'] = 307.0
#        DF.at[index, 'Corrugation angle'] = 60.0
#        DF.at[index, 'S, Channel Side'] = 0.011
#        DF.at[index, 'B, Channel Base'] = 0.0175
#        DF.at[index, 'h, Crimp height'] = 0.0079
#        DF.at[index, 'packing element height'] = 237.1073
#        DF.at[index, 'LP/A'] = 237.11
    elif row['Packing Type'] == 'MG 64Y':
        DF.at[index, 'specific area'] = 64.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.069
        DF.at[index, 'B, Channel Base'] = 0.09
        DF.at[index, 'h, Crimp height'] = 0.053
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 57.86164
    elif row['Packing Type'] == 'RSR 1.5':
        DF.at[index, 'specific area'] = 120.0
#        DF.at[index, 'Corrugation angle'] = 60.0
#        DF.at[index, 'S, Channel Side'] = 0.011
#        DF.at[index, 'B, Channel Base'] = 0.0175
#        DF.at[index, 'h, Crimp height'] = 0.0079
#        DF.at[index, 'packing element height'] = 237.1073
#        DF.at[index, 'LP/A'] = 237.11
    elif row['Packing Type'] == 'B1 250 MN':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.017
        DF.at[index, 'B, Channel Base'] = 0.022
        DF.at[index, 'h, Crimp height'] = 0.011
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 280.9917
    elif row['Packing Type'] == 'HFP 2':
        DF.at[index, 'specific area'] = 100.0
        DF.at[index, 'Corrugation angle'] = 60.0
        DF.at[index, 'S, Channel Side'] = 0.025
        DF.at[index, 'B, Channel Base'] = 0.041
        DF.at[index, 'h, Crimp height'] = 0.015
#        DF.at[index, 'packing element height'] = 237.1073
        DF.at[index, 'LP/A'] = 162.6016
    elif row['Packing Type'] == 'GTO 250Y':
        DF.at[index, 'specific area'] = 250.0
        DF.at[index, 'Corrugation angle'] = 45.0
        DF.at[index, 'S, Channel Side'] = 0.016
        DF.at[index, 'B, Channel Base'] = 0.027
        DF.at[index, 'h, Crimp height'] = 0.01
#        DF.at[index, 'packing element height'] = 237.1073
#        DF.at[index, 'LP/A'] = 162.6016
print(DF)
DF.to_csv(r'C:\Users\bendr\Box\SDS 384 Project\AE_UTSRP_Filled.csv',index=True)
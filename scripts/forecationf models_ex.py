#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:23:50 2019

@author: chiara
"""


import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import itertools
import warnings


from statsmodels.tsa.arima_model import ARMA

ts1=list(range(0,500,2))
len(ts1)

model=ARMA(ts1,order=(0,1))
#model.information()
fit=model.fit(disp=5)
fit.summary()
#                             ARMA Model Results                              
#==============================================================================
#Dep. Variable:                      y   No. Observations:                  250
#Model:                     ARMA(0, 1)   Log Likelihood               -1428.744
#Method:                       css-mle   S.D. of innovations             72.604
#Date:                Thu, 17 Oct 2019   AIC                           2863.489
#Time:                        10:57:35   BIC                           2874.053
#Sample:                             0   HQIC                          2867.740
#                                                                              
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#const        249.0083      9.165     27.169      0.000     231.045     266.972
#ma.L1.y        0.9999      0.010    101.243      0.000       0.981       1.019
#                                    Roots                                    
#=============================================================================
#                  Real          Imaginary           Modulus         Frequency
#-----------------------------------------------------------------------------
#MA.1           -1.0001           +0.0000j            1.0001            0.5000
#-----------------------------------------------------------------------------

# o) P>\z\ is the p-val
# o) AIC (Akaike Information Criterion) value measures how well a model fits 
# the data while taking into account the overall complexity of the model. 
# A model that fits the data very well while using lots of features will be 
# assigned a larger AIC score than a model that uses fewer features to achieve 
# the same goodness-of-fit. Therefore, we are interested in finding the model 
# that yields the lowest AIC value.


pred=fit.predict(len(ts1),len(ts1)) #374.49
pred

from statsmodels.tsa.vector_ar.var_model import VAR
#from statsmodels.tsa.statespace.varmax import VARMAX

ts2=list(range(500,1000,2))
ts=pd.DataFrame({"ts1":ts1,"ts2":ts2})

model=VAR(ts) #,order=(0,1)
#model.information()
fit=model.fit()
fit.summary()

#  Summary of Regression Results   
#==================================
#Model:                         VAR
#Method:                        OLS
#Date:           Thu, 17, Oct, 2019
#Time:                     16:00:22
#--------------------------------------------------------------------
#No. of Equations:         2.00000    BIC:                   -116.125
#Nobs:                     249.000    HQIC:                  -116.175
#Log likelihood:           13767.4    FPE:                3.39553e-51
#AIC:                     -116.209    Det(Omega_mle):     3.31516e-51
#--------------------------------------------------------------------
#Results for equation ts1
#=========================================================================
#            coefficient       std. error           t-stat            prob
#-------------------------------------------------------------------------
#const         -0.001984              NAN              NAN             NAN
#L1.ts1         0.995996              NAN              NAN             NAN
#L1.ts2         0.004004              NAN              NAN             NAN
#=========================================================================
#
#Results for equation ts2
#=========================================================================
#            coefficient       std. error           t-stat            prob
#-------------------------------------------------------------------------
#const          0.002016              NAN              NAN             NAN
#L1.ts1        -0.003996              NAN              NAN             NAN
#L1.ts2         1.003996              NAN              NAN             NAN
#=========================================================================
#
#Correlation matrix of residuals
#            ts1       ts2
#ts1    1.000000  0.951165
#ts2    0.951165  1.000000

pred=fit.forecast(fit.y,steps=1) #array([[ 500., 1000.]])
pred

pred=fit.forecast(fit.y,steps=3) 
pred #array([[ 500., 1000.],
     #       [ 502., 1002.],
     #       [ 504., 1004.]])

##################################### SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create parameters     
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in pdq]#list(itertools.product(p, d, q))


warnings.filterwarnings("ignore") # specify to ignore warning messages

param=pdq[0]
param_seasonal=seasonal_pdq[0]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(ts1, order=param,seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#ARIMA(0, 0, 0)x(0, 0, 0, 52)12 - AIC:3529.4532640333523
#ARIMA(0, 0, 0)x(0, 0, 1, 52)12 - AIC:8524.710121490572  
#ARIMA(0, 0, 0)x(0, 1, 0, 52)12 - AIC:2390.951838473629
#ARIMA(0, 0, 0)x(0, 1, 1, 52)12 - AIC:6109.756521634717
#ARIMA(0, 0, 0)x(1, 0, 0, 52)12 - AIC:2132.090287303192
#ARIMA(0, 0, 0)x(1, 0, 1, 52)12 - AIC:2034.1091306333342
#ARIMA(0, 0, 0)x(1, 1, 0, 52)12 - AIC:-3089.4441840755426 
#ARIMA(0, 0, 0)x(1, 1, 1, 52)12 - AIC:nan
#ARIMA(0, 0, 1)x(0, 0, 0, 52)12 - AIC:8827.74964853632
#ARIMA(0, 0, 1)x(0, 0, 1, 52)12 - AIC:nan
#ARIMA(0, 0, 1)x(0, 1, 0, 52)12 - AIC:8529.012165403003
#ARIMA(0, 0, 1)x(0, 1, 1, 52)12 - AIC:16764.04877539664
#ARIMA(0, 0, 1)x(1, 0, 0, 52)12 - AIC:9566.733370582071 
#ARIMA(0, 0, 1)x(1, 0, 1, 52)12 - AIC:8295.369705647365
#ARIMA(0, 0, 1)x(1, 1, 0, 52)12 - AIC:6356.26416402472
#ARIMA(0, 0, 1)x(1, 1, 1, 52)12 - AIC:6271.2742439695485
#ARIMA(0, 1, 0)x(0, 0, 0, 52)12 - AIC:1049.5945140272559
#ARIMA(0, 1, 0)x(0, 0, 1, 52)12 - AIC:9789.103372012913   
#ARIMA(0, 1, 0)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(0, 1, 0)x(0, 1, 1, 52)12 - AIC:nan
#ARIMA(0, 1, 0)x(1, 0, 0, 52)12 - AIC:-4170.033637108996 
#ARIMA(0, 1, 0)x(1, 0, 1, 52)12 - AIC:-4153.431343153703
#ARIMA(0, 1, 0)x(1, 1, 0, 52)12 - AIC:-3013.1187268516032
#ARIMA(0, 1, 0)x(1, 1, 1, 52)12 - AIC:-3202.583612185782
#ARIMA(0, 1, 1)x(0, 0, 0, 52)12 - AIC:10707.71402921827
#ARIMA(0, 1, 1)x(0, 0, 1, 52)12 - AIC:20986.03629024016  worst
#ARIMA(0, 1, 1)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(0, 1, 1)x(0, 1, 1, 52)12 - AIC:nan
#ARIMA(0, 1, 1)x(1, 0, 0, 52)12 - AIC:8542.970298607246
#ARIMA(0, 1, 1)x(1, 0, 1, 52)12 - AIC:8458.300549382868
#ARIMA(0, 1, 1)x(1, 1, 0, 52)12 - AIC:-3011.1187268516032
#ARIMA(0, 1, 1)x(1, 1, 1, 52)12 - AIC:-3018.8321417660136
#ARIMA(1, 0, 0)x(0, 0, 0, 52)12 - AIC:712.1298895449919
#ARIMA(1, 0, 0)x(0, 0, 1, 52)12 - AIC:10620.112972204352
#ARIMA(1, 0, 0)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 0, 0)x(0, 1, 1, 52)12 - AIC:6111.756521634712
#ARIMA(1, 0, 0)x(1, 0, 0, 52)12 - AIC:-2365.892284196455
#ARIMA(1, 0, 0)x(1, 0, 1, 52)12 - AIC:-1950.972772140532
#ARIMA(1, 0, 0)x(1, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 0, 0)x(1, 1, 1, 52)12 - AIC:nan
#ARIMA(1, 0, 1)x(0, 0, 0, 52)12 - AIC:372.5044628282068     
#ARIMA(1, 0, 1)x(0, 0, 1, 52)12 - AIC:9083.281510795705
#ARIMA(1, 0, 1)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 0, 1)x(0, 1, 1, 52)12 - AIC:6071.64785596824
#ARIMA(1, 0, 1)x(1, 0, 0, 52)12 - AIC:-2089.2449870039572
#ARIMA(1, 0, 1)x(1, 0, 1, 52)12 - AIC:-1929.925530884988
#ARIMA(1, 0, 1)x(1, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 0, 1)x(1, 1, 1, 52)12 - AIC:nan
#ARIMA(1, 1, 0)x(0, 0, 0, 52)12 - AIC:-5251.66293223826 
#ARIMA(1, 1, 0)x(0, 0, 1, 52)12 - AIC:8233.103162467083
#ARIMA(1, 1, 0)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 1, 0)x(0, 1, 1, 52)12 - AIC:-3202.583612185782
#ARIMA(1, 1, 0)x(1, 0, 0, 52)12 - AIC:-4146.842877252098
#ARIMA(1, 1, 0)x(1, 0, 1, 52)12 - AIC:-5916.636927368082 <======   *
#ARIMA(1, 1, 0)x(1, 1, 0, 52)12 - AIC:-3202.583612185782
#ARIMA(1, 1, 0)x(1, 1, 1, 52)12 - AIC:-3200.583612185782
#ARIMA(1, 1, 1)x(0, 0, 0, 52)12 - AIC:-5242.946995244625
#ARIMA(1, 1, 1)x(0, 0, 1, 52)12 - AIC:8193.128146332323
#ARIMA(1, 1, 1)x(0, 1, 0, 52)12 - AIC:nan
#ARIMA(1, 1, 1)x(0, 1, 1, 52)12 - AIC:-3018.8321417660136
#ARIMA(1, 1, 1)x(1, 0, 0, 52)12 - AIC:-4902.063264828318
#ARIMA(1, 1, 1)x(1, 0, 1, 52)12 - AIC:-5051.314673560011
#ARIMA(1, 1, 1)x(1, 1, 0, 52)12 - AIC:-3200.583612185782
#ARIMA(1, 1, 1)x(1, 1, 1, 52)12 - AIC:-3016.8321417660136

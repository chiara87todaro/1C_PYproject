#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:40:31 2019

@author: chiara
"""

import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import matplotlib.pyplot as plt # data plot
import matplotlib
from datetime import datetime,date # date objects
import seaborn as sns # data plot 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import itertools
import warnings
from my_functions_1c import my_toOverallTS

# Set working paths
mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)

######################## LOAD DATA TRAIN
filePath="working_data/"+"1C_small_training.csv"
dataTrain=pd.read_csv(filePath, index_col=False) 
dataTrain=dataTrain.drop("Unnamed: 0",axis=1)
############################## OVERALL TS
#my_toOverallTS
dataTrain["true_date"]=pd.to_datetime(dataTrain["true_date"],format="%Y-%m-%d")
dataTrain["month"]=dataTrain["true_date"].apply(lambda d: d.month)    
dataTrain["week_num"]=dataTrain["true_date"].apply(lambda d: d.isocalendar()[1])    
dataTrain.head()

# aggregate/summarize over weeks
agg_ts={"month":{"month": "first"},"item_cnt_day":{"tot_cnt":np.sum}}
#        "date_block_num":{"date_block_num": "first"},        
overall_ts=dataTrain.groupby(["date_block_num","week_num"]).agg(agg_ts).reset_index()
overall_ts.columns=["date_block_num","week_num","month","tot_cnt"]
overall_ts["index"]=range(0,len(overall_ts["tot_cnt"]))
overall_ts.head() #114X5
###


fig, ax = plt.subplots(figsize=(8, 4))  
#sns.pointplot(ax=ax,x="week_num", y="tot_cnt",data=overall_ts)
sns.lineplot(ax=ax,x="index", y="tot_cnt",hue="month",data=overall_ts,legend="full",sort=False)
fig
# seasonal pattern, min in Jun-Jul max in Dec (see boxplot below)
# overall decreasing trend

sns.set(rc={'figure.figsize':(8,4)})
fig, ax = plt.subplots(1, 1)
#sns.boxplot(x="week_num", y="tot_cnt", hue="month",data=overall_ts)
sns.boxplot(ax=ax,x="month", y="tot_cnt", data=overall_ts)
ax.set_title("Monthly sales")
fig
# in winter(Jan,Nov,Dec) number of sales are higher
# decrement till Jul(min), then increase


#### Is it multiplicative or additive model?
seasonality=3
# additive?
add = sm.tsa.seasonal_decompose(overall_ts.tot_cnt,freq=seasonality,model="additive")
# multiplicative?
molt = sm.tsa.seasonal_decompose(overall_ts.tot_cnt,freq=seasonality,model="multiplicative")

plt.rcParams.update({'figure.figsize': (8,4)})
add.plot().suptitle('Additive %i weeks period' %seasonality, fontsize=8)
molt.plot().suptitle('Multiplicative %i weeks period' %seasonality, fontsize=8)
# additive with freq 3 
# seems to be non-stationary and with overall descending trend

seasonality=3
rollMean=overall_ts.rolling(window=seasonality,center=True,min_periods=1).mean()
rollStd=overall_ts.rolling(window=seasonality,center=True,min_periods=1).std()

sns.set(rc={'figure.figsize':(10,4)})
fig, ax = plt.subplots(1, 2)
sns.pointplot(ax=ax[0],x="week_num", y="tot_cnt", data=rollMean)
sns.pointplot(ax=ax[1],x="week_num", y="tot_cnt", data=rollStd)
ax[0].set_title("rolling mean [%i weeks windows]" %seasonality)
ax[1].set_title("rolling std [%i weeks windows]" %seasonality)
fig


# perfrom Dickey-Fuller test to check non-stat
#h0:the time series possesses a unit root and hence is non-stationary

dfTest = adfuller(overall_ts["tot_cnt"], autolag='AIC', regression='ct')
print(dfTest)
# weakly sales are non-stationary ( montly sales are non-stationary)

#ADF Statistic: -2.731246974980515
#p-value: 0.2233389131653165 > 0.05 => accept h0 (ts is non-stationary)
# 6, 107
#Critial Values:
#   1%, -4.0459709
#Critial Values:
#   5%, -3.4523480
#Critial Values:
#   10%,-3.15157557

#
# KPSS Test
# h0:the time series is stationary
# * 'c' : The data is stationary around a constant (default)
# * 'ct' : The data is stationary around a trend
kpssTest = kpss(overall_ts["tot_cnt"], regression='ct',store=True)
print(kpssTest)
# accept h0 => ts is (trend) stationary

#KPSS Statistic:0.11728819 
#p-value: 0.1  >0.05 => accept h0, ts is stationary
# Critial Values:
#   '10%': 0.119
#Critial Values:
#   '5%': 0.146
#Critial Values:
#   '2.5%': 0.176
#Critial Values:
#   '1%': 0.216

#both test say that monthly ts is non stationary => i need to remove trend and differencing
#both test say that weekly ts is trend stationary but consecutive differences are not
#=> no need to remove trend and but differencing

#observed: data series that has been decomposed.
#seasonal: seasonal component of the data series.
#trend: trend component of the data series.
#resid: residual component of the data series.

#######################
# In order to perform VARMAX model ts must be stationary and without seasonal el

# make data stationary
#overall_ts["stationary"] = overall_ts["tot_cnt"] - rollMean["tot_cnt"]
overall_ts["stationary1"] = overall_ts["tot_cnt"] - add.trend
overall_ts["stationary2"] = overall_ts["stationary1"] - add.seasonal
#n=12
#overall_ts["stationary2"] = overall_ts["stationary1"] - overall_ts["stationary1"].shift(n)

sns.set(rc={'figure.figsize':(10,4)})
fig, ax = plt.subplots(1, 2)
sns.pointplot(ax=ax[0],x="index", y="stationary1", data=overall_ts)
sns.pointplot(ax=ax[1],x="index", y="stationary2", data=overall_ts)
ax[0].set_title("ts after de-trend")
ax[1].set_title("ts after de-trend and %i de-seasoning" %seasonality)
fig
#plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

#for seasonality of 12 the series is stationary

overall_ts["stationary2"]
kpssTest2 = kpss(overall_ts["stationary2"][1:112], regression='ct',store=True)
print(kpssTest2)
# KPSS stat: 0.057268,
# p-val: 0.1, > 0.05 => accept h0, it's stationary
# {'10%': 0.119, '5%': 0.146, '2.5%': 0.176, '1%': 0.216},
dfTest2 = adfuller(overall_ts["stationary2"][1:112], autolag='AIC',regression="ct")
print(dfTest2)
#stat :-6.80449, p-val:3.89e-08 < 0.05 => 
#refuse h0 is stat
#usedlag: 9, nobs: 101, 
#{'1%': -4.0607042399, '5%': -3.459337589, '10%': -3.1556468}

#rollMean2=overall_ts.rolling(window=3,center=True,min_periods=1).mean()
#sns.pointplot(y=rollMean2)

overall_ts.head()
# stationary2 is a stationary time series
#################################################
# autocorrelation at different lags
from pandas.plotting import autocorrelation_plot
sns.set(rc={'figure.figsize':(8,4)})
autocorrelation_plot(overall_ts["tot_cnt"].tolist())
autocorrelation_plot(overall_ts["stationary2"][1:112].tolist())#?

# weekly sales seems completely uncorrelated

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Calculate ACF and PACF upto 15 lags
# ts(t)=Sum_j coeff_j*ts(t-j)

# ACF(lag) is corr in the interval ts(t) & ts(t+lag)
# PACF(lag) is corr between ts(t) & ts(t+lag) <=> coeff_lag


acf_53 = acf(overall_ts["tot_cnt"], nlags=53)
acf_53 # max=0.44 for lag=6
pacf_53 = pacf(overall_ts["tot_cnt"], nlags=53)
pacf_53  #max=0.45 for lag=6,min=-0.46 for lag=43/4=10,7 month

fig, axes = plt.subplots(1,2,figsize=(10,4))
plot_acf(overall_ts["tot_cnt"].tolist(), lags=53, ax=axes[0])
plot_pacf(overall_ts["tot_cnt"].tolist(), lags=53, ax=axes[1])

plot_acf(overall_ts["stationary2"].tolist(), lags=53, ax=axes[0])
plot_pacf(overall_ts["stationary2"].tolist(), lags=53, ax=axes[1])

# after 12-13-14-15 months there is a high dependency in the time series
# after 6 weeks corr is ~0.4 and 
# ts correlates (0.5) for 6-7 weeks shift 

seasonality=3
# additive?
add2 = sm.tsa.seasonal_decompose(overall_ts.stationary2[1:112],
                                freq=seasonality,model="additive")

add2.plot().suptitle('Additive %i weeks period' %seasonality, fontsize=8)

#########
# using the stationary data, create a multivariate forecasting model


#from statsmodels.tsa.statespace.varmax import VARMAX
#from statsmodels.tsa.vector_ar.var_model import VAR
#from statsmodels.tsa.arima_model import ARMA

time_series=pd.DataFrame({"ts":overall_ts["stationary2"][1:112]})
time_series.shape
#model_overall=VARMAX(time_series,order=(1,1)) # order AR,MA
#fit=model_overall.fit()
#pred_overall=fit.forecast()

######## ARMA models
#model01=ARMA(time_series["ts"],order=(0,1))
#fit01=model01.fit(trend="nc")
#fit01.aic #2105.63928    
#fit01.summary()
#
#model10=ARMA(time_series["ts"],order=(1,0))
#fit10=model10.fit(trend="nc")
#fit10.aic #2161.456945   #fit01.summary()
#
#model11=ARMA(time_series["ts"],order=(1,1))
#fit11=model10.fit(trend="nc")
#fit11.aic #2161.455    #fit01.summary()
#
#
#model02=ARMA(time_series["ts"],order=(0,2)) # <- lowest AIC
#fit02=model02.fit(trend="nc",start_params=(0,0))
#fit02.aic #2068.986218   
#fit02.summary()
#
#model03=ARMA(time_series["ts"],order=(0,3)) #
#fit0=model03.fit(trend="nc",start_params=(0,3)) #doesn't work
#
######### ARMA predictions
#fit01.predict(len(time_series["ts"]),len(time_series["ts"]))
##41.64
#fit10.predict(len(time_series["ts"]),len(time_series["ts"]))
##-372.463948
#fit11.predict(len(time_series["ts"]),len(time_series["ts"]))
##-372.463948
#fit02.predict(len(time_series["ts"]),len(time_series["ts"]))
## 552.951643
#overall_ts

##inconsistent!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

########### SARIMAX models
from statsmodels.tsa.statespace.sarimax import SARIMAX
m=52
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(1,1,0,m)]
#seasonal_pdq = [(x[0], x[1], x[2], m) for x in pdq]
warnings.filterwarnings("ignore") # specify to ignore warning messages

#param=pdq[7]
#param_seasonal=seasonal_pdq[6]
param=pdq[24]
param_seasonal=seasonal_pdq[0]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(time_series, order=param,seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}, AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#ARIMA(0, 0, 0)x(0, 0, 0, 52)12 - AIC:2191.4163228208636
#ARIMA(0, 0, 0)x(0, 1, 0, 52)12 - AIC:1183.8569466414535
#ARIMA(0, 0, 0)x(1, 0, 0, 52)12 - AIC:1164.0251691421233
#ARIMA(0, 0, 0)x(1, 1, 0, 52)12 - AIC:128.11965022634422 
#ARIMA(0, 0, 1)x(0, 0, 0, 52)12 - AIC:2078.3747803143992
#ARIMA(0, 0, 1)x(0, 1, 0, 52)12 - AIC:1110.4472390267786
#ARIMA(0, 0, 1)x(1, 0, 0, 52)12 - AIC:1118.0721788850708
#ARIMA(0, 0, 1)x(1, 1, 0, 52)12 - AIC:124.16646502170576 
#ARIMA(0, 1, 0)x(0, 0, 0, 52)12 - AIC:2299.7333204194188
#ARIMA(0, 1, 0)x(0, 1, 0, 52)12 - AIC:1233.5740539510543
#ARIMA(0, 1, 0)x(1, 0, 0, 52)12 - AIC:1213.5059603697825
#ARIMA(0, 1, 0)x(1, 1, 0, 52)12 - AIC:113.6774348603242  <--- 
#ARIMA(0, 1, 1)x(0, 0, 0, 52)12 - AIC:2169.7651905306984
#ARIMA(0, 1, 1)x(0, 1, 0, 52)12 - AIC:1152.7617850362658
#ARIMA(0, 1, 1)x(1, 0, 0, 52)12 - AIC:1160.218510027322
#ARIMA(0, 1, 1)x(1, 1, 0, 52)12 - AIC:125.15784096013127
#ARIMA(1, 0, 0)x(0, 0, 0, 52)12 - AIC:2142.207809593809
#ARIMA(1, 0, 0)x(0, 1, 0, 52)12 - AIC:1149.8397729237383
#ARIMA(1, 0, 0)x(1, 0, 0, 52)12 - AIC:1115.816075326018
#ARIMA(1, 0, 0)x(1, 1, 0, 52)12 - AIC:108.04653890188447 <--- 
#ARIMA(1, 0, 1)x(0, 0, 0, 52)12 - AIC:2067.8277802343
#ARIMA(1, 0, 1)x(0, 1, 0, 52)12 - AIC:1099.6807594771026
#ARIMA(1, 0, 1)x(1, 0, 0, 52)12 - AIC:1081.7493250147934
#ARIMA(1, 0, 1)x(1, 1, 0, 52)12 - AIC:108.82812632413523
#ARIMA(1, 1, 0)x(0, 0, 0, 52)12 - AIC:2220.725984574685
#ARIMA(1, 1, 0)x(0, 1, 0, 52)12 - AIC:1184.9189100722897
#ARIMA(1, 1, 0)x(1, 0, 0, 52)12 - AIC:1149.2840957374685
#ARIMA(1, 1, 0)x(1, 1, 0, 52)12 - AIC:104.7852514996399 <--- 
#ARIMA(1, 1, 1)x(0, 0, 0, 52)12 - AIC:2127.101532836692
#ARIMA(1, 1, 1)x(0, 1, 0, 52)12 - AIC:1120.2466707660924
#ARIMA(1, 1, 1)x(1, 0, 0, 52)12 - AIC:1117.1785762891038
#ARIMA(1, 1, 1)x(1, 1, 0, 52)12 - AIC:104.13761428714041 <========
            
#ARIMA(0, 0, 0)x(1, 1, 0, 52), AIC:128.11965022634422
#ARIMA(0, 0, 1)x(1, 1, 0, 52), AIC:124.16646502170576
#ARIMA(0, 0, 2)x(1, 1, 0, 52), AIC:126.80471292716484
#ARIMA(0, 1, 0)x(1, 1, 0, 52), AIC:113.6774348603242
#ARIMA(0, 1, 1)x(1, 1, 0, 52), AIC:125.15784096013127
#ARIMA(0, 1, 2)x(1, 1, 0, 52), AIC:122.9767242461235
#ARIMA(0, 2, 0)x(1, 1, 0, 52), AIC:99.33368898019553
#ARIMA(0, 2, 1)x(1, 1, 0, 52), AIC:111.94147476090457
#ARIMA(0, 2, 2)x(1, 1, 0, 52), AIC:110.00404339811348
#ARIMA(1, 0, 0)x(1, 1, 0, 52), AIC:108.04653890188447
#ARIMA(1, 0, 1)x(1, 1, 0, 52), AIC:108.82812632413523
#ARIMA(1, 0, 2)x(1, 1, 0, 52), AIC:110.81008316666139
#ARIMA(1, 1, 0)x(1, 1, 0, 52), AIC:104.7852514996399
#ARIMA(1, 1, 1)x(1, 1, 0, 52), AIC:104.13761428714041
#ARIMA(1, 1, 2)x(1, 1, 0, 52), AIC:104.41623840078793
#ARIMA(1, 2, 0)x(1, 1, 0, 52), AIC:89.34398683720734
#ARIMA(1, 2, 1)x(1, 1, 0, 52), AIC:88.44236001368095
#ARIMA(1, 2, 2)x(1, 1, 0, 52), AIC:89.11329190995114
#ARIMA(2, 0, 0)x(1, 1, 0, 52), AIC:91.59466354633251
#ARIMA(2, 0, 1)x(1, 1, 0, 52), AIC:93.40173674616543
#ARIMA(2, 0, 2)x(1, 1, 0, 52), AIC:101.83774587604307
#ARIMA(2, 1, 0)x(1, 1, 0, 52), AIC:85.00237108259043
#ARIMA(2, 1, 1)x(1, 1, 0, 52), AIC:86.24221129458505
#ARIMA(2, 1, 2)x(1, 1, 0, 52), AIC:85.70031116025395
#ARIMA(2, 2, 0)x(1, 1, 0, 52), AIC:68.27313032424159 <=========
#ARIMA(2, 2, 1)x(1, 1, 0, 52), AIC:69.28420298580042
#ARIMA(2, 2, 2)x(1, 1, 0, 52), AIC:69.45531747632694     
        
######################## best

fit111x110_52=results
fit111x110_52.summary()
#
#                                 Statespace Model Results                                 
#==========================================================================================
#Dep. Variable:                                 ts   No. Observations:                  111
#Model:             SARIMAX(1, 1, 1)x(1, 1, 0, 52)   Log Likelihood                 -48.069
#Date:                            Tue, 22 Oct 2019   AIC                            104.138
#Time:                                    10:53:47   BIC                            102.575
#Sample:                                         0   HQIC                            99.945
#                                            - 111                                         
#Covariance Type:                              opg                                         
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ar.L1         -1.1556     71.900     -0.016      0.987    -142.076     139.765
#ma.L1         -0.5205     86.512     -0.006      0.995    -170.081     169.040
#ar.S.L52      -0.4159      7.025     -0.059      0.953     -14.185      13.354
#sigma2       3.44e+07   3.19e+07      1.079      0.280   -2.81e+07    9.69e+07
#===================================================================================
#Ljung-Box (Q):                         nan   Jarque-Bera (JB):                 0.45
#Prob(Q):                               nan   Prob(JB):                         0.80
#Heteroskedasticity (H):               3.71   Skew:                             0.36
#Prob(H) (two-sided):                  0.42   Kurtosis:                         1.72
#===================================================================================
#
#Warnings:
#[1] Covariance matrix calculated using the outer product of gradients (complex-step).


fit111x110_52.plot_diagnostics(figsize=(10, 8))

pred=fit111x110_52.predict(start=len(time_series)+1)
pred #8654.028939

fit220x110_52=results
fit220x110_52.summary()
#Covariance Type:                              opg                                         
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ar.L1         -2.0191   6.39e+04  -3.16e-05      1.000   -1.25e+05    1.25e+05
#ar.L2         -1.0310   1.48e+04  -6.96e-05      1.000    -2.9e+04     2.9e+04
#ar.S.L52      -0.4069   8663.327   -4.7e-05      1.000    -1.7e+04     1.7e+04
#sigma2      8.428e+07     44.859   1.88e+06      0.000    8.43e+07    8.43e+07

fit220x110_52.plot_diagnostics(figsize=(10, 8))
pred=fit220x110_52.predict(start=len(time_series)+1)
pred #9737.342701

#stationary2 has negative values that is not good
overall_ts
#the next obs is 13500

###################################################################
########### SARIMAX on raw data
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(overall_ts["tot_cnt"], order=(0,1,0),seasonal_order=(1,1,0,52),
              enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
results.summary()
#AIC   179.275             #Covariance Type:                                                                
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#ar.S.L52      -0.4713      0.110     -4.299      0.000      -0.686      -0.256
#sigma2       1.68e+07   1.01e-09   1.66e+16      0.000    1.68e+07    1.68e+07
#==============================================================================
#differenting at 1 step for the trend, AR=1 + diff=1 for the season,
# seasonal coeff is -0.4713 and it is significant ....?

results.plot_diagnostics(figsize=(10, 8))
# standardized residual are close to 0....seems ok
# KDE and N(0,1) are very similar, which is good
# theoretical and sampel quantiles look close enough
pred=results.predict(start=len(overall_ts["tot_cnt"])+1)
pred # 115    11450.902819
preds= results.get_forecast(steps=52)
forec=pd.DataFrame({"index":preds.predicted_mean.index,"preds":preds.predicted_mean})
preds_ci = preds.conf_int()
preds_ci

#fig,ax= overall_ts["tot_cnt"].plot(label='observed', figsize=(8,8))
#preds.predicted_mean.plot(label='forecast') #ax=ax,
#ax.fill_between(preds_ci.index,preds_ci.iloc[:, 0],preds_ci.iloc[:, 1], color='k', alpha=.25)
#fig
#plt.legend()
#plt.show()

fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=overall_ts,label='observed')
sns.lineplot(ax=ax,x="index", y="preds", data=forec,label='forecast')
ax.set_title("SARIMAX: 52 weeks forecast")
fig


#pred #5727.606832 model=220x110_52  AIC: 132.657
#pred #27855.175636 model=000x110_52  AIC: 217.348
#pred #0 model=000x000_52  AIC: 2559.623
#pred #11450.902819 model=010x000_52  AIC: 179.275 *signif coeff
 ###########################################
 ###########################################
 ###########################################
 ###########################################
 ###########################################
 ####################### EXPONENTIAL SMOOTHING ##############################
from statsmodels.tsa.holtwinters import ExponentialSmoothing
data = overall_ts["tot_cnt"]
# create class
#d=[]
#for seas in np.arange(2, 53) :
#    model=ExponentialSmoothing(data,trend="add",damped=True,seasonal="add",seasonal_periods=seas)
##    model=ExponentialSmoothing(data,trend="mul",damped=True,seasonal="mul",seasonal_periods=seas)
#    d.append((seas,model.fit().aic))
#   
#modelADD_seasRange=pd.DataFrame(d,columns=["param","aic"]) 
#modelADD_seasRange["aic"].idxmin() #29     31  2018.771262
#modelADD_seasRange["aic"].idxmax() #47     49  2163.182370
#
#modelMUL_seasRange=pd.DataFrame(d,columns=["param","aic"]) 
#modelMUL_seasRange["aic"].idxmin() #29     31  1999.103765
#modelMUL_seasRange["aic"].idxmax() #47     49  2251.367739
# 
##modelMIX_seasRange=pd.DataFrame(d,columns=["param","aic"]) NaN
#
model = ExponentialSmoothing(data, trend="add",damped=True,seasonal="add",seasonal_periods=52)
#model = ExponentialSmoothing(data, trend="add",damped=False,seasonal="add",seasonal_periods=52)
# DUMPING the trend increase the values, which is good for the negative ones
# fit model

# smoothing_level (alpha): the smoothing coefficient for the level.
# smoothing_slope (beta): the smoothing coefficient for the trend.
# smoothing_seasonal (gamma): the smoothing coeff for the seasonal component.
# damping_slope (phi): the coefficient for the damped trend.

model_fit = model.fit(optimized=True)
model_fit.summary()
#Dep. Variable:                    endog   No. Observations:                  114
#Model:             ExponentialSmoothing   SSE                     2621538254.974
#Optimized:                         True   AIC                           2046.394
#Trend:                         Additive   BIC                           2202.358
#Seasonal:                      Additive   AICC                          2177.506
#Seasonal Periods:                    52   Date:                 Wed, 23 Oct 2019
#Box-Cox:                          False   Time:                         15:36:13
#==============================================================================
#                          coeff                 code              optimized      
#------------------------------------------------------------------------------
#smoothing_level               0.2418255                alpha           True
#smoothing_slope               0.0525096                 beta           True
#smoothing_seasonal             0.000000                gamma           True                    
# make prediction
pred = model_fit.forecast(steps=52) #.predict()
data_pred=pd.DataFrame({"tot_cnt":pred,"index":pred.index})
data_pred
#pred 2129.571705
# forecast 2129.571705, 15591.732232, 9872.994551, 10251.722765, 7154.313527, 
#-9107.745220, 6575.357608, 9460.414273, 13795.106829, 8925.107870, 
#-5334.767262, 6077.423734

#fig=plt.figure(figsize=(10,8))
#plt.plot(ax=fig,data.index, data, label='data')
#plt.plot(ax=fig,pred.index,pred, label='forecast')
##plt.plot(y_hat_avg.index,y_hat_avg['Holt_Winter'], label='Holt_Winter')
#plt.legend(loc='best')
#plt.savefig('Holt_Winters.jpg')
 

fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=overall_ts,label='observed')
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=data_pred,label='forecast')
ax.set_title("HWES: 52 weeks forecast")
fig
 
 # SARIMAX vs HWES
fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=overall_ts,label='observed')
sns.lineplot(ax=ax,x="index", y="preds", data=forec,label='SARIMAX')
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=data_pred,label='HWES')
ax.set_title("SARIMAX-HWES: 52 weeks forecast")
fig
 #HWES looks better(higher trend), but both are too low with negative values
 
 #################################################
 # use control data to see which one is the best
filePath="working_data/"+"1C_ctrl_training.csv"
dataCtrl=pd.read_csv(filePath, index_col=False) 
dataCtrl=dataCtrl.drop("Unnamed: 0",axis=1)
dataCtrl=dataCtrl.drop("Unnamed: 0.1",axis=1)
dataCtrl=dataCtrl.drop("Unnamed: 0.1.1",axis=1)
dataCtrl.describe()
dataCtrl.dtypes
 
overallTSctrl=my_toOverallTS(dataCtrl)
overallTSctrl.head(20)
 # ctrl data are in the past...
 #let's see how well is forecast as it is in the future
# 165-114=51
 overallTSctrl=overallTSctrl[0:52]
 overallTSctrl["index"]=range(114,166)

fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=overall_ts,label='observed')
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=overallTSctrl,label='ctrl')
sns.lineplot(ax=ax,x="index", y="preds", data=forec,label='SARIMAX')
sns.lineplot(ax=ax,x="index", y="tot_cnt", data=data_pred,label='HWES')
ax.set_title("ctrl-HWES: 52 weeks forecast")
fig

dataCtrl["item_price"].max()
 #################################################################
#  month \ id \ price* \ count* \  *mean over month and shops
agg_ts={"item_price":{"mean_price":np.mean},"item_cnt_day":{"mean_cnt":np.mean}}
data_ts=dataTrain.groupby(["date_block_num","item_id"]).agg(agg_ts).reset_index()
data_ts.set_index(["date_block_num"]).head(n=10)
data_ts.columns=["date_block_num","item_id","mean_price","mean_cnt"]
data_ts.head(n=10)
max(data_ts["mean_cnt"]) #118.57
data_ts.round({"mean_cnt":0}) # round col to 0 decimal digits 

# study seasonality and trend of data_ts

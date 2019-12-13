#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:52:52 2019

@author: chiara
"""

# Import libraries
import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import matplotlib.pyplot as plt # data plot
import matplotlib
from datetime import datetime,date # date objects
#import dateutil
import seaborn as sns # data plot 
#from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

# Set working paths
mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)
dataDir="/home/chiara/kaggle/1C_PYproject/data/"
dataDir1="competitive-data-science-predict-future-sales/"
dataTrainPath=dataDir+dataDir1+"sales_train_v2.csv"
data = pd.read_csv(dataTrainPath, index_col=False) #
data_dim=data.shape 
# there are 1935849 obs and 6 variables
data.keys() # variable names
# ['date', 'date_block_num', 'shop_id','item_id','item_price','item_cnt_day']
data.dtypes
#date               object
#date_block_num      int64
#shop_id             int64
#item_id             int64
#item_price        float64
#item_cnt_day      float64
# data summary 
data.describe()
#       date_block_num       shop_id       item_id    item_price  item_cnt_day
#count    2.935849e+06  2.935849e+06  2.935849e+06  2.935849e+06  2.935849e+06
#mean     1.456991e+01  3.300173e+01  1.019723e+04  8.908532e+02  1.242641e+00
#std      9.422988e+00  1.622697e+01  6.324297e+03  1.729800e+03  2.618834e+00
#min      0.000000e+00  0.000000e+00  0.000000e+00 -1.000000e+00 -2.200000e+01
#25%      7.000000e+00  2.200000e+01  4.476000e+03  2.490000e+02  1.000000e+00
#50%      1.400000e+01  3.100000e+01  9.343000e+03  3.990000e+02  1.000000e+00
#75%      2.300000e+01  4.700000e+01  1.568400e+04  9.990000e+02  1.000000e+00
#max      3.300000e+01  5.900000e+01  2.216900e+04  3.079800e+05  2.169000e+03

data.hist()
#plt.hist(data.date)#""
#plt.title("")
#plt.hist(data.date_block_num)#""
#plt.title("")
#plt.hist(data.shop_id)
#plt.title("")
#plt.hist(data.item_id)
#plt.title("")
#plt.hist(data.item_price,bins=50)
#plt.title("item price")
#plt.hist(data.item_cnt_day,bins=100)
#plt.title("daily count")


########################## CHECKS ##################

# is "date" a date object?
type(data["date"]) #pandas.core.series.Series
data["date"].min() #01.01.2013
data["date"].max() #31.12.2014
data["true_date"]=pd.to_datetime(data["date"],format="%d.%m.%Y")
min(data["true_date"]) #2013/01/01
max(data["true_date"]) #2015/10/31

## what's "date_block_num"?
data[data["date_block_num"]==0]
data.iloc[115689:115695,:]
data[data["date_block_num"]==12][1:10]
data[data["date_block_num"]==24][1:10]
data[data["date_block_num"]==33][1:10]
# it's a monthly counter

#################################################################
# create a new data frame with a date column and item_category_id
dataItemsPath=dataDir+dataDir1+"items.csv"
dataItems = pd.read_csv(dataItemsPath, index_col=False) #
dataItems.head()

dataItems["item_id"].nunique() #22170 it's unique
sum(dataItems["item_id"]==sorted(dataItems["item_id"])) #it's sorted..


dataTrain1=data
dataTrain1["true_date"]=pd.to_datetime(data["date"],format="%d.%m.%Y")

dataTrain1.head()

dataTrain=dataTrain1.join(dataItems.set_index('item_id'), on='item_id')
dataTrain=dataTrain.drop(["item_name","date"],axis=1)
dataTrain.head()
dataTrain.set_index("item_price")
dataTrain=dataTrain.drop(-1,axis=0)
dataTrain.reset_index()
badInd=dataTrain[dataTrain["item_price"]==-1].index
dataTrain=dataTrain.drop(badInd,axis=0) #remove row with bad entry
dataTrain[dataTrain["item_price"]==-1] 

filePath="working_data/"+"1C_new_training.csv"
dataTrain.to_csv(filePath,index="FALSE",encoding="utf8")
################################### SPLIT ctrl and train
filePath="working_data/"+"1C_new_training.csv"
dataTrain=pd.read_csv(filePath, index_col=False)
# data distribution
sns.set(rc={'figure.figsize':(18,10)})
dataTrain.hist()

dataCtrl=dataTrain[dataTrain["date_block_num"]<12]
filePath="working_data/"+"1C_ctrl_training.csv"
dataCtrl.to_csv(filePath,index="FALSE",encoding="utf8")

dataNewTrain=dataTrain[dataTrain["date_block_num"]>11]
filePath="working_data/"+"1C_small_training.csv"
dataNewTrain.to_csv(filePath,index="FALSE",encoding="utf8")
#dataTrain.to_csv(filePath,index="FALSE",encoding="utf8")

################################### LOAD dataTrain
filePath="working_data/"+"1C_small_training.csv"
dataTrain=pd.read_csv(filePath, index_col=False)
dataTrain.head(10)
dataTrain.keys()
# data distribution
sns.set(rc={'figure.figsize':(18,10)})
dataTrain.hist()

dataTrain.head(10)

########### CATEGORIES MONTHLY TREND
# plot monthly trend for item daily counts
data_category_trend=dataTrain.groupby(["date_block_num","item_category_id"])["item_cnt_day"].sum().reset_index()#
data_category_trend.head() # 3 columns
data_category_trend.describe()
fig1=sns.lineplot(x="date_block_num", y='item_cnt_day', hue='item_category_id',data=data_category_trend,legend=False)
fig1.set_title("Monthly item category trend")
plt.show(fig1)
DB1=data_category_trend.sort_values("item_cnt_day",ascending=False)
popular_cat=DB1["item_category_id"].head(n=100).unique()
#ind=[0]*DB.shape[0]
#for i in popular_cat:    
#    ind1=DB["item_category_id"]==i
#    ind=ind+ind1
#sum(ind)
#ind=bool(ind.tolist())
DB1=data_category_trend[data_category_trend.item_category_id.isin(popular_cat)]
DB1.shape
DB1["item_category_id"].nunique()
sns.pointplot(x="date_block_num", y='item_cnt_day', hue='item_category_id',data=DB1)
sns.lineplot(x="date_block_num", y='item_cnt_day', hue='item_category_id',data=DB1)
# RES
# global descending trends
# peaks in march and december

########### SHOP SALES 
data_shop_sales=dataTrain.groupby(["shop_id","item_category_id"])["item_cnt_day"].sum().reset_index()
data_shop_sales.head()
sns.set(rc={'figure.figsize':(18,10)})
sns.barplot(x="shop_id",y="item_cnt_day",estimator=sum,data=data_shop_sales)
#sns.boxplot(x="shop_id",y="item_cnt_day",fliersize,data=data_shop_sales)
# RES
# popular shops 31,25,54,57,28,55

############################# PRICE - COUNTS
data_price_sales1=dataTrain.groupby(["item_id","item_price"])["item_cnt_day"].sum().reset_index()
data_price_sales1.head(n=20)
#sum(data_price_sales1["item_price"]==-1)
#dataTrain[dataTrain["item_price"]==-1]
sns.set(rc={'figure.figsize':(18,10)})
sns.lineplot(x="item_price", y='item_cnt_day', data=data_price_sales1)
#sns.barplot(x="item_price", y='item_cnt_day', data=data_price_sales1)
#sns.barplot(x="item_cnt_day", y='item_price', data=data_price_sales1)

# RES
# cheaper objects are sold the most, no visible trend

#############
# How are the price distributed?
sns.distplot(a=data_price_sales1["item_price"])
data_price_sales1["item_price"].hist(bins=200)
prices=data_price_sales1["item_price"].unique()
prices_counts=data_price_sales1["item_price"].value_counts()
prices.min() #0.07 # 0.5
prices.max() #307980 #50999
sum(prices>1000) #8848 #6914
sum(prices>500) #13484 #10624
sum(prices>2000) #4483 #3332
sum(prices>10000) #989 #735
sum(prices>50000) #3 #1
#data_price_sales2=dataTrain.groupby(["item_id","item_price","date_block_num"])["item_cnt_day"].sum().reset_index()
price_ranges=[0,10,50,100,500,1000,5000,10000,50000,310000]
price_labs=["<10","<50","<100","<500","<1000","<5000","<10000","<50000","<310000"]
ranges=pd.cut(prices,bins = price_ranges,labels = price_labs)
x=ranges.value_counts()
sum(x)
price_dist=pd.DataFrame({"price":prices,"count":prices_counts,"range":ranges})
sns.set(rc={'figure.figsize':(18,10)})
# sns.barplot(x="price",y="count",hue="range",data=price_dist)
# RES
# most of the items are between 1000 and 5000
# more than 50% of the items are between 500 and 5000

#############
price_ranges=[0,10,50,100,500,1000,5000,10000,50000,310000]
price_labs=["<10","<50","<100","<500","<1000","<5000","<10000","<50000","<310000"]
data_price_sales2=dataTrain.groupby(["item_id","item_price","date_block_num","item_category_id"])["item_cnt_day"].sum().reset_index()
data_price_sales2["price_range"]=pd.cut(data_price_sales2["item_price"],bins = price_ranges,labels = price_labs)
data_price_sales2=data_price_sales2.sort_values("date_block_num")
data_price_sales2.head(n=10)
data_price_sales2.keys()
data_price_sales2=data_price_sales2.groupby(["date_block_num","price_range"])["item_cnt_day"].sum().reset_index()

sns.set(rc={'figure.figsize':(18,10)})
sns.lineplot(x="date_block_num", y="item_cnt_day",hue='price_range', data=data_price_sales2)

# RES
# <500 highest sales, decreasing trend
# <1000 & <5000 over 20000 sales stable trend, first decreas second incres
# <10 around 10000 sales quite stable trend
# others lower sales with stable trend

sns.set(rc={'figure.figsize':(18,10)})
sns.barplot(x="price_range", y="item_cnt_day",data=data_price_sales2)


# RES
# price >100 and <5000 => highest sales
#### correlation between price and sales?

CC=np.corrcoef(x=dataTrain["item_cnt_day"], y=dataTrain["item_price"], rowvar=False)
# 0.01119661 => it's not linear, more gaussian
# 0.00607
#CC=np.corrcoef(x=dataTrain, rowvar=False)

######### is an item price fixed?
def my_perc(x) :
    p=(np.std(x)/np.mean(x))*100 
    return p
    
agg_price={"item_price":{"mean_price":np.mean,"sd_price":np.std,"perc_var":my_perc}}
fix_price=dataTrain.groupby(["item_id"])["item_price"].agg(agg_price).reset_index()
fix_price.head(n=30)

np.mean(fix_price["item_price"]["perc_var"])
# the mean percentage variability on price is 14.017958589978722
# the mean percentage variability on price is 12.14313279296649
################ how often items and shops do appear?
data_freq=dataTrain.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].sum().reset_index()
#agg_freq={"shop_id":{"freq_shop":pd.value_counts}}
#shop_freq1=data_freq.agg(agg_freq)
#shop_freq.set_index(["shop_id"])
#agg_freq={"item_id":{"freq_item":pd.value_counts}}
#item_freq=data_freq.agg(agg_freq)
#data_freq["shop_id"]=data_freq["shop_id"].sort_values()
freq=data_freq["shop_id"].value_counts()
shop_freq=pd.DataFrame({"id":freq.index,"freq":freq})
shop_freq.head(n=10)
outPath="working_data/"+"1C_shopFreq.csv"
shop_freq.to_csv(outPath,index=False)
freq=data_freq["item_id"].value_counts()
item_freq=pd.DataFrame({"id":freq.index,"freq":freq})
item_freq.head(n=10)
outPath="working_data/"+"1C_itemFreq.csv"
item_freq.to_csv(outPath,index=False)
#freq=data_freq["item_category_id"].value_counts()
#item_freq=pd.DataFrame({"id":freq.index,"freq":freq})
#item_freq.head(n=10)
#data_freq.columns=["date_block_num","shop_id","item_id","tot_cnt"]
dataTrain["date_block_num"].value_counts()

freq.index


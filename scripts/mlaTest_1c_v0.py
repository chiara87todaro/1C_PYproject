#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:30:58 2019

@author: chiara
"""

import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
# import matplotlib.pyplot as plt # data plot
#import matplotlib
#from datetime import datetime,date # date objects
# import seaborn as sns # data plot 
# import statsmodels.api as sm
#import networkx as nx
# from sklearn.ensemble import RandomForestRegressor
import pickle

# Set working paths
mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)

#from my_functions_1c import my_prepareTrain
import my_functions_1c as ct

######################## LOAD DATA TRAIN & CATALOG
# create catalog
dfPath="working_data/"+"1C_catalog.csv"
# item_cat_price.to_csv(dfPath, header=True, index=False)
item_cat_price=pd.read_csv(dfPath,index_col=False)


filePath="working_data/"+"1C_small_training.csv"
data=pd.read_csv(filePath, index_col=False) 
data=data.drop("Unnamed: 0",axis=1)

# dataAll[dataAll["item_id"]==83]
# dataAll=data2.append(data)

dataTrain=ct.my_prepareTrain(data) #921400 rows x 9 columns

# dataTrainHM=ct.my_summaryHistoricFunc(dataTrain,f_mean=True,f_sum=False) #15:54-15:09
filePath="working_data/"+"1C_train_histoMean.csv"
dataTrainHM=pd.read_csv(filePath, index_col=False) 

dataTrain.reset_index()
dataTrainHM.reset_index()

D=pd.merge(dataTrain,dataTrainHM,how="left",on=["date_block_num","item_id","shop_id"])

# [y,X]=ct.my_df2arry_endo_exog(D,"month_cnt")
# rfModel=RandomForestRegressor(n_estimators=500,max_depth=10,random_state=18)
# rfFit=rfModel.fit(X,y) #17:09-17:26

# f=open("rf_fit.pckl","wb")
# pickle.dump(rfFit,f)
# f.close()

f=open("rf_fit.pckl","rb")
rfFit=pickle.load(f)
f.close()




##### TEST data set

filePath="/home/chiara/kaggle/1C_PYproject/data/competitive-data-science-predict-future-sales/"+"test.csv"
dataTest=pd.read_csv(filePath, index_col=False) 
dataTest.keys()
D.keys() # need to add date block, category, month, price, histo_mean_cnt
# order of features for RF!!!!!!!
# 'date_block_num', 'item_id', 'shop_id', 'category_id', 'month',
#        'item_price', 'month_cnt', 'histo_mean_cnt']
# i can merge with trainHM..
T=pd.merge(dataTest,D,how="left",on=["item_id","shop_id"])
# there are duplicates with different features

newTest=ct.my_prepareTest(dataTest,D,item_cat_price,new_month=True)
newTest.isnull().sum() #16800/214200 *100 = 7,84%

dfPath=mainPath+"working_data/"+"test_touse.csv"
newTest.to_csv(dfPath, header=True, index=False)



[yt,Xt]=ct.my_df2arry_endo_exog(newTest)
predTest=rfFit.predict(Xt)
pred_test=dataTest
pred_test["item_cnt_month"]=predTest
pred_test=pred_test.drop(["shop_id","item_id"],axis=1)
dfPath=mainPath+"working_data/"+"predictions_test.csv"
pred_test.to_csv(dfPath, header=True, index=False)

pred_test.describe()

#########
filePath="working_data/"+"1C_ctrl_training.csv"
data2=pd.read_csv(filePath, index_col=False) 
data2=data2.drop(["Unnamed: 0",'Unnamed: 0.1', 'Unnamed: 0.1.1'],axis=1)

dataCtrl=ct.my_prepareTrain(data2)
dataCtrl=dataCtrl.drop(["category_id","item_price","month_cnt"],axis=1)
dataCtrl.keys()
newCtrl=ct.my_prepareTest(dataCtrl,D,item_cat_price,new_month=False)
newCtrl.head()
dfPath=mainPath+"working_data/"+"ctrl_dataAStest.csv"
# newCtrl.to_csv(dfPath, header=True, index=False)

dfPath="working_data/"+"1C_ctrl_histoMean.csv"
dataCtrlHM=pd.read_csv(dfPath, index_col=False) 

newCtrl.keys()
dataCtrl.keys()
sum(newCtrl["histo_mean_cnt"]==dataCtrlHM["histo_mean_cnt"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:04:25 2019

@author: chiara
"""

import numpy as np # scientific calculation
import pandas as pd # data analysis
from datetime import datetime,date # date objects
import statsmodels.api as sm
import statsmodels.formula.api as smf
import calendar

# function that transform 1C project data into a overall time series
def my_perc(x) :
    p=(np.std(x)/np.mean(x))*100 
    return p

def my_toOverallTS(data):
    data["true_date"]=pd.to_datetime(data["true_date"],format="%Y-%m-%d")
    data["month"]=data["true_date"].apply(lambda d: d.month)    
    data["week_num"]=data["true_date"].apply(lambda d: d.isocalendar()[1])    
    # aggregate/summarize over weeks
    agg_ts={"month":{"month": "first"},"item_cnt_day":{"tot_cnt":np.sum}}
    overall_ts=data.groupby(["date_block_num","week_num"]).agg(agg_ts).reset_index()
    overall_ts.columns=["date_block_num","week_num","month","tot_cnt"]
    overall_ts["index"]=range(0,len(overall_ts["tot_cnt"]))
    return overall_ts


def my_prepareTrain(data):
    data["true_date"]=pd.to_datetime(data["true_date"],format="%Y-%m-%d")
    data["month"]=data["true_date"].apply(lambda d: d.month)
#data["month"]=data["true_date"].apply(lambda d: calendar.month_name[d.month]) #month name
    #aggregate
    agg_freq={#"item_id":{"item_freq": "count"},
#              "shop_id":{"shop_freq":"count"},
              "item_category_id":{"category_id":"first"},
              "month":{"month":"first"},
              "item_price":{"item_price":"first"},
              "item_cnt_day":{"month_cnt":"sum"}
#              "month":{"month2":"mean"},
#              "item_price":{"item_price2":"mean"}
              }
    dataTrain=data.groupby(["date_block_num","item_id","shop_id"]).agg(agg_freq).reset_index()
    dataTrain.columns=["date_block_num","item_id","shop_id",
                       "category_id",#"shop_freq","item_freq",
                       "month","item_price","month_cnt"]
    return dataTrain


def my_df2arry_endo_exog(data,target):
    endo=np.array(data[target], dtype=float)
    exog=np.array(data[[col for col in data.columns if col != target]], dtype=float)
    return [endo,exog]

def my_historicMean(data,date_block,col_name):
     # select obs of the previous months
     data_copy=data[data["date_block_num"]<date_block].groupby(["item_id","shop_id"])
     # calculate the mean of the selected column
     histoMean_past=data_copy[col_name].mean()
     # select obs of the current month
     histoMean_present=data[data["date_block_num"]==date_block].set_index(["item_id","shop_id"])[col_name]
     # set the mean of the current month to 0
     histoMean_present.values[:]=1 
     # add previous and current month
     histoMean_past=histoMean_past.append(histoMean_present)
     # calculate the mean between present and past (not very accurate)
     # sum the historic mean to the current, which is 0
     histoMean_all=histoMean_past.groupby(["item_id","shop_id"]).sum()
#     histoMean.reset_index()
     # select item-shop of the current month ?????  TODO 
     # or maybe it is better to do it afterwards all together...easier now..
     # create list of pairs item-shop
     item_shop=data[data["date_block_num"]==date_block][["item_id","shop_id"]].values.tolist()
     # create grouped df
     hmDF=histoMean_all.groupby(["item_id","shop_id"])
     # concatenate all 
     histoMean=pd.concat(hmDF.get_group(tuple(g)) for g in item_shop)
     return histoMean

def my_historicSum(data,date_block,col_name):
     # select obs of the previous months
     data_copy=data[data["date_block_num"]<date_block].groupby(["item_id","shop_id"])
     # calculate the mean of the selected column
     histoSum_past=data_copy[col_name].sum()
     # select obs of the current month
     histoSum_present=data[data["date_block_num"]==date_block].set_index(["item_id","shop_id"])[col_name]
     # set the sum of the current month to 0!!!!!!
     histoSum_present.values[:]=1 
     # add previous and current month
     histoSum_past=histoSum_past.append(histoSum_present)
     # calculate the total between present and past (not very accurate)
     # sum the historic mean to the current, which is 0
     histoSum_all=histoSum_past.groupby(["item_id","shop_id"]).sum()
#     histoMean.reset_index()
     # select item-shop of the current month ?????  TODO 
     # or maybe it is better to do it afterwards all together...easier now..
     # create list of pairs item-shop
     item_shop=data[data["date_block_num"]==date_block][["item_id","shop_id"]].values.tolist()
     # create grouped df
     hsDF=histoSum_all.groupby(["item_id","shop_id"])
     # concatenate all 
     histoSum=pd.concat(hsDF.get_group(tuple(g)) for g in item_shop)
     return histoSum

#def my_historicMean(data,column_name,replaceNaN=True):
#    period=data["date_block_num"].unique()
#    histoMean=np.empty(shape=(len(period),1))
#    for i in range(len(period)):
#        histoMean[i]=data[data["date_block_num"]<period[i]][column_name].mean()
#        if (replaceNaN==True):
#            if (np.isnan(histoMean[i])):
#                histoMean[i]=0          
#    return histoMean
    
def my_summaryHistoricFunc(data,f_mean=True,f_sum=True):
#    agg_hm={"month_cnt":{"histo_mean":data.apply(my_historicMean)}}
#    dataSumm=data.groupby(group_columns).aggregate(agg_hm)
#    return dataSumm
    # create period range
    period=data["date_block_num"].unique()
    # initialize auxiliary data
#    if(sum([f_mean==True,f_sum==True])==2):
#        summDF_aux=pd.DataFrame(columns=["date_block_num","item_id","shop_id","histo_f1_cnt","histo_f2_cnt"])
#        summDF=pd.DataFrame(columns=["date_block_num","item_id","shop_id","histo_f1_cnt","histo_f2_cnt"])
#    elif(sum([f_mean==True,f_sum==True])==1):
    summDF_aux=pd.DataFrame(columns=["date_block_num","item_id","shop_id"])
    summDF=pd.DataFrame(columns=["date_block_num","item_id","shop_id"])
#    else:
#        print("Something wrong..no code has been run!")
#        break

    summDF.set_index(["date_block_num","item_id","shop_id"])
    # for every month
    for i in range(len(period)): 
        # calculate the historic mean till (<=) that month
        if(f_mean==True):
            df=my_historicMean(data,period[i],"month_cnt").reset_index()
            summDF_aux["histo_mean_cnt"]=df["month_cnt"]
        if(f_sum==True):
            df=my_historicSum(data,period[i],"month_cnt").reset_index()
            summDF_aux["histo_sum_cnt"]=df["month_cnt"]
        # add the reference to that month
        df["date_block_num"]=np.ones(len(df))*period[i]
        # copy output in the right format
        summDF_aux["date_block_num"]=np.ones(len(df))*period[i]
        summDF_aux["item_id"]=df["item_id"]
        summDF_aux["shop_id"]=df["shop_id"]
        
        
        summDF_aux.set_index(["date_block_num","item_id","shop_id"])
        # append historic means all together
        summDF=summDF.append(summDF_aux,sort=False)
#        if(sum([f_mean==True,f_sum==True])==2):
#            summDF_aux=pd.DataFrame(columns=["date_block_num","item_id","shop_id","histo_f1_cnt","histo_f2_cnt"])
#        elif(sum([f_mean==True,f_sum==True])==1):
        summDF_aux=pd.DataFrame(columns=["date_block_num","item_id","shop_id"])
    
    return summDF

def my_rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


def my_calculate_RMSE_ACC(data,target_col,fitModel):
    [target,X]=my_df2arry_endo_exog(data,"month_cnt")
#    target=data[target_col]
#    data2=data.drop(target_col,axis=1)
    predictions=fitModel.predict(exog=X, transform=True)
#    err=abs(target-predictions)
    rmse=my_rmse(target,predictions) 
    err=abs(target-predictions)
    acc=100*(len([e for e in err if e<1])/len(err)) 
    res=[rmse,acc]
    return res
    

def my_compareFitModels(dataTrain,modelFormula,modelName,modelFamily,dataTest):
    [y,X]=my_df2arry_endo_exog(dataTrain,"month_cnt")
    [y_c,X_c]=my_df2arry_endo_exog(dataTest,"month_cnt")
    if(modelName,modelFamily)==("GLM","poisson"):
#        model = smf.glm(formula=modelFormula, data=dataTrain, family=sm.families.Poisson())
        model = sm.GLM(y,X,family=sm.families.Poisson())
    elif(modelName,modelFamily)==("GLM","gamma"):
#        model = smf.glm(formula=modelFormula, data=dataTrain, family=sm.families.Gamma())
         model = sm.GLM(y,X, data=dataTrain, family=sm.families.Gamma())
         
    fitModel=model.fit(method='nm', maxiter=500, maxfun=500)
    [rmse_in,acc_in]=my_calculate_RMSE_ACC(dataTrain,"month_cnt",fitModel)
    [rmse_out,acc_out]=my_calculate_RMSE_ACC(dataTest,"month_cnt",fitModel)
    
    accRes=pd.DataFrame({
            "model":[modelName],
            "formula":["stuff"],#modelFormula
            "family":[modelFamily],
            "aic":[fitModel.aic],
            "scale":[fitModel.scale],
            "log-likel":[fitModel.llf],
            "deviance":[fitModel.deviance],
            "chi2":[fitModel.pearson_chi2],
            "mean_err_perc":[np.mean((fitModel.bse/fitModel.params)*100)],
            "sign_pval_perc":[(len([n for n in fitModel.pvalues if n<=0.05])/len(fitModel.pvalues))*100],
            "rmse_in": [rmse_in],
            "rmse_out":[rmse_out],
            "acc_in": [acc_in],
            "acc_out":[acc_out]
            })
        
    return accRes
    
    
    
    
    
    
    
    
    
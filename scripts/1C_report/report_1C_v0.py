#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:09:46 2019

@author: chiara
"""

# IMPORT MODULES & SET PATH
import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import matplotlib.pyplot as plt # data plot
import seaborn as sns # data plot 
# import statsmodels.api as sm
#import networkx as nx
# import chart_studio.plotly as py
# import plotly.graph_objects as go
# from plotly.offline import plot
# from plotly.subplots import make_subplots

# from sklearn.ensemble import RandomForestRegressor


mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)

import my_functions_1c as ct 


#### LOAD DATA
basicPath="/home/chiara/kaggle/1C_PYproject/" #scripts/

filePath=basicPath+"data/competitive-data-science-predict-future-sales/"+"sales_train_v2.csv"
dataAll=pd.read_csv(filePath, index_col=False) 

print(dataAll.head(3))
print(dataAll.tail(3))

filePath1=basicPath +"data/competitive-data-science-predict-future-sales/"+"item_categories.csv"
itemCat=pd.read_csv(filePath1, index_col=False) 

itemCat[itemCat["item_category_id"]==28]["item_category_name"]

#filePath=basicPath+"data/competitive-data-science-predict-future-sales/"+"item_categories.csv"
#D=pd.read_csv(filePath, index_col=False) 

filePath=basicPath +"scripts/working_data/"+"1C_small_training.csv"
data=pd.read_csv(filePath, index_col=False) 
data=data.drop("Unnamed: 0",axis=1)

filePath=basicPath+"scripts/working_data/"+"1C_ctrl_training.csv"
data2=pd.read_csv(filePath, index_col=False) 
data2=data2.drop(["Unnamed: 0",'Unnamed: 0.1', 'Unnamed: 0.1.1'],axis=1)

dataAll=data2.append(data)
dataAll.head(4)

aN_itemID=dataAll["item_id"].value_counts().reset_index().sort_values("index")
aN_itemPrice=dataAll["item_price"].value_counts().reset_index().sort_values("index")
aN_categoryID=dataAll["item_category_id"].value_counts().reset_index().sort_values("index")
aN_shopID=dataAll["shop_id"].value_counts().reset_index().sort_values("index")

N_itemID=data["item_id"].value_counts().reset_index().sort_values("index")
N_itemPrice=data["item_price"].value_counts().reset_index().sort_values("index")
N_categoryID=data["item_category_id"].value_counts().reset_index().sort_values("index")
N_shopID=data["shop_id"].value_counts().reset_index().sort_values("index")


N_itemPrice[N_itemPrice["index"]==20949]
len(N_shopID) 
len(aN_shopID) 
len(N_itemID)
len(aN_itemID)
len(N_categoryID) 
len(aN_categoryID) 
N_itemPrice["index"].max()
abs(N_itemPrice["index"]).min()

# Presence % of items, shops, and categories
# for the price it would be better
#fig2 = go.Figure(
#    data=[go.Bar(y=[2, 1, 3])],
#    layout_title_text="A Figure Displayed with fig.show()"
#)
#fig2.show()

# fig1 = make_subplots(rows=2, cols=2,
#                      subplot_titles=("ID", "Price", "Category", "Shop"))

# fig1.add_trace(go.Bar(x=N_itemID["index"],y=N_itemID["item_id"]),row=1, col=1)
# fig1.add_trace(go.Bar(x=N_itemPrice["index"],y=N_itemPrice["item_price"]),row=1, col=2)
# fig1.add_trace(go.Bar(x=N_categoryID["index"],y=N_categoryID["item_category_id"]),row=2, col=1)
# fig1.add_trace(go.Bar(x=N_shopID["index"],y=N_shopID["shop_id"]),row=2, col=2)
# fig1.update_layout(height=600, width=800, title_text="Popularity of...", showlegend=False)
# #plot(fig2)

# plot(fig1, filename = "hist_dataAll.html", auto_open=False)
# plot(fig1, filename = "hist_dataAll_raw.html", include_plotlyjs=False, output_type='div')

# dataAll["item_id"]
# sns.set(rc={'figure.figsize':(10,12)})
# fig, ax = plt.subplots(2, 2)
# sns.barplot(x=N_itemID["index"],y=N_itemID["item_id"],ax=ax[0,0]) #
# sns.pieplot(x=N_itemPrice["item_price"],ax=ax[0,1])
# sns.pieplot(x=N_categoryID["item_category_id"],ax=ax[1,0])
# sns.barplot(x=N_shopID["index"],y=N_shopID["shop_id"],ax=ax[1,1])
# #ax[0,0].set_title("Itemid histogram")
# #ax[1].set_title("% of items sold in each shop for month")
# fig
x=itemCat[itemCat["item_category_id"]==40]["item_category_name"].reset_index()
print(x["item_category_name"])

########## FIG 1: countXdateBYcategory_lineplot

# data["true_date"]=pd.to_datetime(data["true_date"],format="%Y-%m-%d")
# data["month"]=data["true_date"].apply(lambda d: calendar.month_name[d.month]) #month name
# data["month"]=data["month"].apply(lambda s: s[0:3])
# data["year"]=data["true_date"].apply(lambda d: str(d.year)[2:4]) 
# data["MonYe"]=data["month"]+data["year"]
data_categ=data.groupby(["item_category_id"])["item_id"].unique()
# number of items for each category
DD1=data_categ.apply(lambda x:len(x)).reset_index()
DD1.columns=["item_category_id","num_items"]
data=ct.my_createMonthYear(data)

data_category_trend=data.groupby(["date_block_num","MonYe","item_category_id"])["item_cnt_day"].sum().reset_index()#
DB1=data_category_trend.sort_values("item_cnt_day",ascending=False)
popular_cat=DB1["item_category_id"].head(n=100).unique()

DB1=data_category_trend[data_category_trend.item_category_id.isin(popular_cat)]
DB1.shape
DB1["item_category_id"].nunique()
sns.set(rc={'figure.figsize':(25,50)})
fig1, ax1 = plt.subplots(2, 1)
sns.barplot(x="item_category_id",y="num_items",data=DD1,ax=ax1[0])
sns.pointplot(x='MonYe', y='item_cnt_day', #order=DB1["date_block_num"].unique(),
              hue='item_category_id',data=DB1,ax=ax1[1])
ax1[0].set_title("Number of items per category")
ax1[1].set_title("Popular categories trends")
ax1[0].set_xticklabels(ax1[0].get_xticklabels(), rotation=45)
fig1

###### FIG 2: cntXprice

price_ranges=[0,10,50,100,500,1000,5000,10000,50000,310000]
price_labs=["<10","<50","<100","<500","<1000","<5000","<10000","<50000","<310000"]
data_price_sales=data.groupby(["item_id","item_price","date_block_num","item_category_id"])["item_cnt_day"].sum().reset_index()
data_price_sales["price_range"]=pd.cut(data_price_sales["item_price"],bins = price_ranges,labels = price_labs)
data_price_sales=data_price_sales.sort_values("date_block_num")
data_price_sales.head(n=5)
data_price_sales.keys()
data_price_sales1=data_price_sales.groupby(["item_category_id"])["price_range"].unique()

data_price_sales11=pd.pivot_table(data_price_sales, #[["item_category_id","price_range"]]
                                   values="item_price",index=["item_category_id"],
                                   columns=["price_range"])
AA=data_price_sales11.notna()
B=AA.unstack().apply(lambda x: int(x)).reset_index()

data_price_sales2=data_price_sales.groupby(["date_block_num","price_range"])["item_cnt_day"].sum().reset_index()

#stripplot swarmplot
sns.set(rc={'figure.figsize':(25,15)})
fig2, ax2 = plt.subplots(1, 2)
sns.heatmap(AA,center=0.5, cmap="YlGnBu",ax=ax2[0])   
# sns.stripplot(y="item_category_id", x="price_range",data=B,ax=ax2[0])#
sns.barplot(x="price_range", y="item_cnt_day",data=data_price_sales2,ax=ax2[1])
fig2

######## FIG 3: cntXshop

data=ct.my_createMonthYear(data)
# tot num of items sold in each shop each month
data_shop_trend=data.groupby(["date_block_num","MonYe","shop_id"])["item_cnt_day"].sum().reset_index()#
DB3=data_shop_trend.sort_values("item_cnt_day",ascending=False)
popular_shop=DB3["shop_id"].head(n=100).unique()

DB3=data_shop_trend[data_shop_trend.shop_id.isin(popular_shop)]
DB3.shape
DB3["shop_id"].nunique()

# tot num of item_id sold in each shop
data_shop_sales=data.groupby(["shop_id","item_id"])["item_cnt_day"].sum().reset_index()
data_shop_sales.head()

aux=data_shop_sales.groupby(["shop_id"])["item_cnt_day"].sum()

sns.set(rc={'figure.figsize':(10,8)})
fig3, ax3 = plt.subplots(1, 2)
sns.barplot(x="shop_id",y="item_cnt_day",estimator=sum,data=data_shop_sales,ax=ax3[0])
sns.pointplot(x='MonYe', y='item_cnt_day',hue='shop_id',data=DB3,ax=ax3[1])
ax3[1].set_title("Popular shops trends")
ax3[1].set_xticklabels(ax3[1].get_xticklabels(), rotation=45)
fig3

################## FORECAST

dataTrain=ct.my_prepareTrain(data) #921400 rows x 9 columns
dataTrainHM=ct.my_summaryHistoricFunc(dataTrain,f_mean=True,f_sum=False)#~7min
dfPath=basicPath+"scripts/working_data/"+"1C_train_histoMean.csv"
dataTrainHM.to_csv(dfPath, header=True, index=False)
D=pd.merge(dataTrain,dataTrainHM,how="left",on=["date_block_num","item_id","shop_id"])

filePath="working_data/"+"1C_ctrl_training.csv"
data2=pd.read_csv(filePath, index_col=False) 
data2=data2.drop(["Unnamed: 0",'Unnamed: 0.1', 'Unnamed: 0.1.1'],axis=1)
dataCtrl=ct.my_prepareTrain(data2)
dataCtrlHM=ct.my_summaryHistoricFunc(dataCtrl,f_mean=True,f_sum=False) #takes almost 10 minutes
dfPath=basicPath+"scripts/working_data/"+"1C_ctrl_histoMean.csv"
# dataCtrlHM.to_csv(dfPath, header=True, index=False)
dataCtrlHM=pd.read_csv(dfPath, index_col=False)

C=pd.merge(dataCtrl,dataCtrlHM,how="left",on=["date_block_num","item_id","shop_id"])

## correlation between variables
Sigma=D.corr()

[y,X]=ct.my_df2arry_endo_exog(D,"month_cnt")
rfModel=RandomForestRegressor(n_estimators=500,max_depth=10,random_state=18)
rfFit=rfModel.fit(X,y) #17:09-17:26

f=open("rf_fit.pckl","rb")
rfFit=pickle.load(f)
f.close()

pred=rfFit.predict(X) #17:26-17:27

#### errors
abs_err=abs(y-pred)
err=y-pred
train_err=pd.DataFrame({"date_block_num":D["date_block_num"],
                        "item_id":D["item_id"],"shop_id":D["shop_id"],
                        "actual":D["month_cnt"],"predicted":pred,
                        "err":err})
dfPath=basicPath+"scripts/working_data/"+"1C_train_err.csv"
train_err.to_csv(dfPath, header=True, index=False)

np.mean(abs_err) #1.3330819427844776
np.max(abs_err) #1166.5251575783172
np.min(abs_err) #e-05
100*(len([e for e in err if e>0])/len(err)) #25.951378337312786
100*(len([e for e in err if e<0])/len(err)) #74.04862166268722
100*(len([e for e in abs_err if e<1])/len(abs_err)) #69.638, tuning: 72.93043195137834
100*(len([e for e in abs_err if e<2])/len(abs_err)) # n_estimators=500 :88.0572
np.mean([e for e in err if e<0]) #-0.89704447

dfPath=mainPath+"working_data/"+"ctrl_dataAStest.csv"
C=pd.read_csv(dfPath, index_col=False) 

[yC,XC]=ct.my_df2arry_endo_exog(C)
# rfModel=RandomForestRegressor(n_estimators=500,max_depth=10,random_state=18)
# rfFit=rfModel.fit(X,y) #17:09-17:26
predC=rfFit.predict(XC) #17:26-17:27



#### errors
abs_errC=abs(yC-predC)
errC=yC-predC
ctrl_err=pd.DataFrame({"date_block_num":C["date_block_num"],
                        "item_id":C["item_id"],"shop_id":C["shop_id"],
                        "actual":dataCtrl["month_cnt"],"predicted":predC,
                        "err":errC})
dfPath="working_data/"+"1C_ctrlAStest_err.csv"
ctrl_err.to_csv(dfPath, header=True, index=False)

############ LOAd ERR

dfPath=basicPath+"scripts/working_data/"+"1C_train_err.csv"
train_err=pd.read_csv(dfPath, index_col=False) 
dfPath=basicPath+"scripts/working_data/"+"1C_ctrl_err.csv"
ctrl_err=pd.read_csv(dfPath, index_col=False) 

train_err["int_predicted"]=round(train_err["predicted"])
train_err["int_err"]=round(train_err["actual"]-train_err["predicted"])

ctrl_err["int_predicted"]=round(ctrl_err["predicted"])
ctrl_err["int_err"]=round(ctrl_err["actual"]-ctrl_err["predicted"])

train_err["new"]="no"
train_err["data"]="train"
ctrl_err["data"]="ctrl"
all_err=train_err.append(ctrl_err,ignore_index=True)


dfPath=basicPath+"scripts/working_data/"+"1C_all_err.csv"
# all_err.to_csv(dfPath,header=True, index=False) 
all_err=pd.read_csv(dfPath, index_col=False) 

ctrl_err=all_err[all_err["data"]=="ctrl"]
ctrl_err=ctrl_err.drop("data",axis=1)

sum(all_err["predicted"]<0)
sum(all_err["err"]<0)

sns.set(rc={'figure.figsize':(12,6)})
fig5=sns.lmplot(x="actual",y="predicted",data=all_err,
                hue="new", col="data",sharey=True) 
fig5=fig5.ylim(-100, 2500)
fig5


# Figure 5 shows how the machine learnign algorithm actually performs. 
# The actual against the predicted sales are compared for 
# the train and control data set, on the left and right, respectively.
# Generally, the algorithm underestimates the actual sales, althogh in same 
# cases when the actual number of sold items is low, the error on the prediction 
# could be almost 500 items. This behaviour characterize in particular the 
# "new" items, as shown in the right panel.



sns.set(rc={'figure.figsize':(12,6)})
fig6=sns.lmplot(x="actual",y="err",data=all_err, hue="new", col="data")


# sns.set(rc={'figure.figsize':(12,6)})
# fig4, ax4 = plt.subplots(1, 2)
# sns.regplot(x="actual",y="predicted",data=train_err,ax=ax4[0])
# ax4[0].set_title("[train] x: actual | y: predicted")
# sns.regplot(x="actual",y="err",data=train_err, marker="+",ax=ax4[1])
# ax4[1].set_title("[train] x: actual | y: actual-pred")
# # sns.regplot(x="actual",y="err",data=train_err,ax=ax4[0,2])
# # ax4[0,2].set_title("[train] x: actual | y: act-pred")
# fig4

# sns.set(rc={'figure.figsize':(12,6)})
# fig5 #, ax5 = plt.subplots(1, 2)
# sns.lmplot(x="actual",y="predicted",data=ctrl_err, hue="new")
# # ax4[1,0].set_title("[control] x: actual | y: predicted")
# sns.lmplot(x="actual",y="err",data=ctrl_err, hue="new")
# # ax4[1,1].set_title("[control] x: actual | y: abs(act-pred)")
# # sns.regplot(x="actual",y="err",data=ctrl_err,ax=ax4[1,2])
# # ax4[0,2].set_title("[control] x: actual | y: act-pred")


ct.my_rmse(train_err["actual"],train_err["predicted"]) # 4.64
ct.my_rmse(ctrl_err["actual"],ctrl_err["predicted"]) # 6.71

####### diffrent predictions
# actual, predicted, histo_mean
newItems=list(set(data2["item_id"]).difference(set(data["item_id"])))
newItems.sort()

ctrl_err["new"]="no"
for it in newItems:
    ctrl_err.loc[ctrl_err["item_id"]==it,"new"]="yes" #takes a lot
        
ctrl_err[ctrl_err["item_id"]==12]
data_predictions=pd.DataFrame({"actual":ctrl_err["actual"],
                               "predicted":ctrl_err["predicted"],
                               "historic":C["histo_mean_cnt"],
                               "new":ctrl_err["new"]})

dfPath=mainPath+"working_data/"+"ctrl_dataPredictions.csv"
# data_predictions.to_csv(dfPath,header=True, index=False)
data_predictions=pd.read_csv(dfPath,index_col=False)


data_predictions=data_predictions.reset_index().set_index(["index","new"])#"actual","predicted","historic"
data_predictions2=data_predictions.stack().reset_index()
data_predictions2
data_predictions2.columns=["id","new","method","sales"]

data_predictions=data_predictions.reset_index()


sns.set(rc={'figure.figsize':(8,6)})
fig4, ax4 = plt.subplots(1, 1)
sns.scatterplot(x="id",y="sales",hue="method",x_jitter=0.2,y_jitter=0.2,
                data=data_predictions2[data_predictions2["id"]<600],ax=ax4)
# sns.scatterplot(x="actual",y="predicted",hue="new",x_jitter=0.2,y_jitter=0.2,
#                 data=data_predictions[data_predictions["index"]<10000],ax=ax4[1])
# sns.scatterplot(x="index",y="actual",hue="new",color="r",
#                 data=data_predictions[data_predictions["index"]<500],ax=ax4[1])
fig4

errCrmse=ctrl_err.groupby("new")["err"].mean()

### find feature for bad predictions

sns.set(rc={'figure.figsize':(8,6)})
fig6, ax6 = plt.subplots(1, 1)
sns.scatterplot(x="actual",y="predicted",hue="item_id",x_jitter=0.2,y_jitter=0.2,
                data=ctrl_err)#,ax=ax6[0]
sns.scatterplot(x="actual",y="predicted",hue="shop_id",x_jitter=0.2,y_jitter=0.2,
                data=ctrl_err,ax=ax6[1])
sns.scatterplot(x="actual",y="predicted",hue="date_block_num",x_jitter=0.2,y_jitter=0.2,
                data=ctrl_err,ax=ax6[2])
# sns.scatterplot(x="actual",y="predicted",hue="item_id",x_jitter=0.2,y_jitter=0.2,
#                 data=ctrl_err,ax=ax6[3])
fig6

ctrl_err.groupby("item_id")["err"].mean().sort_values()
# item 587 ->err=-156, item 805 ->err=130, 
ctrl_err.groupby("date_block_num")["err"].mean().sort_values() #~0
ctrl_err.groupby("shop_id")["err"].mean().sort_values() 
# shop 9 ->err=8, shop 42 ->err=-3, 

100*(sum(ctrl_err["err"]>10)/len(ctrl_err["err"]))

{11860} & set(newItems) #587

data_predictions["predicted"].mean() #2.33
data_predictions["predicted"].mean() #2.33

ct.my_rmse(data_predictions["actual"],data_predictions["predicted"]) #6,71
ct.my_rmse(data_predictions["actual"],data_predictions["historic"]) #6,25


[rmseCtrl,accCtrl]=ct.my_calculate_RMSE_ACC(data_predictions,"actual",
                                              predictions=data_predictions["historic"],
                                              fitModel=None,
                                              make_pred=False,thres=1)

#acc 71.35,  rmse 6.71 predicted , mean abs 1.45 ,sd abs 6.55
#acc 36.85,  rmse 6.25 historic, mean abs 1.85 ,sd abs 5.97
#                               , mean abs 1.72 ,sd abs 5.27
err= abs(data_predictions["actual"]- data_predictions["historic"])
err.mean() 
err.std()


#########




# linear regression actual vs err (non absolute)
from sklearn.linear_model import LinearRegression
model_pred = LinearRegression().fit(np.array(train_err["actual"]).reshape(-1,1), train_err["int_predicted"])
model_pred.coef_ #0.68175731 with pred
model_err = LinearRegression().fit(np.array(train_err["actual"]).reshape(-1,1), train_err["int_err"])
model_err.coef_ #0.31824269 with err
#model.intercept_ #0.47178088839165866 with err
r2p = model_pred.score(np.array(train_err["actual"]).reshape(-1,1), train_err["int_predicted"]) 
# 0.7581618697440436 variance explained
r2e = model_err.score(np.array(train_err["actual"]).reshape(-1,1), train_err["int_err"]) 
# 0.4058638684925433

[rmseTrain,accTrain]=ct.my_calculate_RMSE_ACC(train_err,"actual",
                                              predictions=train_err["predicted"],
                                              fitModel=None,
                                              make_pred=False,thres=5)
# thres=1
# rmseTrain  4.644675423212977
# accTrain 72.93043195137834

# thres=2
# rmseTrain  4.644675423212977
# accTrain 88.05719557195572

# thres=5
# rmseTrain  4.644675423212977
# accTrain 96.49023225526373

[rmseCtrl,accCtrl]=ct.my_calculate_RMSE_ACC(ctrl_err,"actual",
                                              predictions=ctrl_err["predicted"],
                                              fitModel=None,
                                              make_pred=False,thres=5)
accCtrl

# thres=1
# rmseCtrl 6.7065828266341425
# accCtrl 71.35260656891428

# thres=2
# rmseCtrl 6.7065828266341425
# accCtrl 87.66947205565023

# thres=5
# rmseCtrl 6.7065828266341425
# accCtrl 95.81460004304051

#######/// comparison with NAIVE RF
dataTrain=ct.my_prepareTrain(data) #921400 rows x 9 columns
dataTrain.keys()
[y,X]=ct.my_df2arry_endo_exog(dataTrain,"month_cnt")
rfModel=RandomForestRegressor(n_estimators=500,max_depth=10,random_state=18)
rfFit=rfModel.fit(X,y) #17:16-17:28
pred=rfFit.predict(X) #17:26-17:27

#### errors
abs_err=abs(y-pred)
err=y-pred
D=dataTrain
train_err=pd.DataFrame({"date_block_num":D["date_block_num"],
                        "item_id":D["item_id"],"shop_id":D["shop_id"],
                        "actual":D["month_cnt"],"predicted":pred,
                        "err":err})
dfPath=basicPath+"scripts/working_data/"+"1C_train_err_noHistMean.csv"
train_err.to_csv(dfPath, header=True, index=False)

np.mean(abs_err) #1.3330819427844776
np.max(abs_err) #1166.5251575783172
np.min(abs_err) #e-05
100*(len([e for e in err if e>0])/len(err)) #25.951378337312786
100*(len([e for e in err if e<0])/len(err)) #74.04862166268722
100*(len([e for e in abs_err if e<1])/len(abs_err)) #69.638, tuning: 72.93043195137834
100*(len([e for e in abs_err if e<5])/len(abs_err)) # n_estimators=500 :88.0572
np.mean([e for e in err if e<0]) #-0.89704447
# thres        1      2       5
# train_acc   69.00  86.72   96.17
# ctrl_acc    66.23  86.94   95.58

### ADDING histo Mean improves only by few 1-2% the train accuracy 


[yc,Xc]=ct.my_df2arry_endo_exog(dataCtrl,"month_cnt")
predc=rfFit.predict(Xc) #17:26-17:27

#### errors
abs_errc=abs(yc-predc)
errc=yc-predc
D=dataCtrl
ctrl_err=pd.DataFrame({"date_block_num":D["date_block_num"],
                        "item_id":D["item_id"],"shop_id":D["shop_id"],
                        "actual":D["month_cnt"],"predicted":predc,
                        "err":errc})
dfPath=basicPath+"scripts/working_data/"+"1C_ctrl_err_noHistMean.csv"
ctrl_err.to_csv(dfPath, header=True, index=False)

100*(len([e for e in abs_errc if e<2])/len(abs_errc)) #69.638, tuning: 72.93043195137834
100*(len([e for e in abs_errc if e<5])/len(abs_errc)) # n_estimators=500 :88.0572


dataAll=data2.append(data)

meanMonthlySales=dataAll.groupby(["date_block_num"])["item_cnt_day"].sum()
meanMonthlySales.mean()

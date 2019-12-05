#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:10:13 2019

@author: chiara
"""

import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import matplotlib.pyplot as plt # data plot
#import matplotlib
#from datetime import datetime,date # date objects
import seaborn as sns # data plot 
import statsmodels.api as sm
#import networkx as nx
from sklearn.ensemble import RandomForestRegressor

# Set working paths
mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)

#from my_functions_1c import my_prepareTrain
import my_functions_1c as ct
######################## LOAD DATA TRAIN
filePath="working_data/"+"1C_small_training.csv"
data=pd.read_csv(filePath, index_col=False) 
data=data.drop("Unnamed: 0",axis=1)
data.keys()
data.head()
dataTrain=ct.my_prepareTrain(data) #921400 rows x 9 columns
dataTrain.keys()
#["date_block_num","item_id","shop_id","item_freq","shop_freq",
# "category_freq", "month","item_price","month_cnt"]
dataTrain.reset_index()
dataTrain.iloc[10:20,0:5]
dataTrain.plot(subplots=True)
##############################################################################
##############################################################################
############# CHECKS/SUMMARIES
## is the item price fixed among shops? over months?
# price is not fixed among shops
# price is not fixed among months

dataPriceXShop=dataTrain[{"date_block_num","item_id","shop_id","item_price"}]
dataPriceXShop.head()
dataPriceXShop.shape
dataItemXShop_price=pd.pivot_table(dataPriceXShop,
                                   index=["date_block_num","item_id"],
                                   values="item_price",columns=["shop_id"])
dataItemXShop_price #[135451 rows x 55 columns]
dataItemXShop_price.keys()
dataItemXShop_price.index
dataItemXShop_price.loc[(33,33)] 
# all shops priced item 33 199, but shop 49 priced it 159
dataItemXShop_price.loc[(12,33)]


# which items are consistent/present among shops? over months?
33-12+1 # 22 months
nan_indices=dataItemXShop_price.isnull()
#dataItemXShop_count=pd.pivot_table(nan_indices,
#                                   index="item_id",columns=[""]
dataItemXShop_count=nan_indices.groupby("item_id").sum() #over months
dataItemXShop_count.max(axis=1).idxmax()
 #item 30 occurs 22 times in at least 1 shop
dataItemXShop_count.max(axis=1).max()
dataItemXShop_count.max(axis=1).idxmin() 
##item 0 occurs 1 times in at least 1 shop
dataItemXShop_count.max(axis=1).min()
itemPresence=dataItemXShop_count.sum(axis=1)/55 
#stability of item presence on average

itemPresence.plot(kind="hist",bins=22,figsize=(10,5),
                  title="Number of item occurrences in 22 month period") #sort_values(ascending=False).
# most items appear only once
sns.set(rc={'figure.figsize':(10,12)})
fig, ax = plt.subplots(1, 1)
sns.heatmap(dataItemXShop_count,ax=ax)
ax.set_title("Monthly appeareances of items in shops")
fig
 

######
dataItemXMonth_price=pd.pivot_table(dataTrain[{"date_block_num","item_id","item_price"}],
                  index=["item_id"],values="item_price",
                  columns=["date_block_num"],aggfunc={np.min,np.max})
dataItemXMonth_price.keys()
# item 22167
dataItemXMonth_price.loc[(22167)] 
# item 22167 varys min price from 284 to 155

nan_indices2=dataItemXMonth_price.iloc[:,range(0,22)].isnull()
#sum(nan_indices2.values.tolist()==nan_indices.values.tolist())
nan_indices2.iloc[0:10,0:10] #itemXmonths
nan_indices.iloc[0:10,0:10] #itemXshops

####
# each month, in how many shops each item occurs?
dataItemXMonth_count=pd.pivot_table(dataTrain[{"date_block_num","item_id","shop_id"}],
                  index=["item_id"],values="shop_id",
                  columns=["date_block_num"],aggfunc=pd.value_counts)
dataItemXMonth_count.iloc[17000:17005,0:5]
dataItemXMonth_count=dataItemXMonth_count.applymap(lambda x: np.nansum(x))

dataItemXMonth_count.keys()

dataItemXMonth_count.iloc[0:40,].transpose().plot.line()

sns.set(rc={'figure.figsize':(10,12)})
fig, ax = plt.subplots(1, 1)
sns.heatmap(dataItemXMonth_count,ax=ax)
ax.set_title("Item appearences in each month")
fig
# most items appear only a few times. 
# none item has a regular high appearence
#
#dataItemXMonth_count=dataItemXMonth_count.reset_index()
#dataItemXMonth_count.columns=["item_id",12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,33]
#dataItemXMonth_count.iloc[0:5,].transpose().plot.line()
#dataItemXMonth_count.keys()

####
# how many items each shop sell each month?

dataShopXMonth_count=pd.pivot_table(dataTrain[{"date_block_num","item_id","shop_id"}],
                  index=["shop_id"],values="item_id",
                  columns=["date_block_num"],aggfunc="count")
dataShopXMonth_perc=dataShopXMonth_count.applymap(lambda x: (x/17054)*100)

#dataShopXMonth_count.max().max()
sns.set(rc={'figure.figsize':(10,12)})
fig, ax = plt.subplots(1, 2)
sns.heatmap(dataShopXMonth_count,ax=ax[0])
sns.heatmap(dataShopXMonth_perc,ax=ax[1])
ax[0].set_title("Items sold in each shop for month")
ax[1].set_title("% of items sold in each shop for month")
fig
# shop 9,11,13,17,20,25,29,30,31..have more variety
# only 20% of items are sold in each shop, 
# and none is continuosly sold
###############################################################################
###############################################################################
############################### CREATE DF for prediction
dataTrain.plot(subplots=True)
# *keys
# date_block_num *
# item_id *
# shop_id *
# category_freq <-
# item_price
# item_freq <-
# shop_freq <-
# month 
# month_cnt !!!!TARGET

dataTrain.keys()
dataTrain.set_index(["date_block_num","shop_id","item_id"])
dataTrain.iloc[20:30,2:8]
#sum(dataTrain["item_freq"]==dataTrain["shop_freq"])
## Calculate correlation between variables
# all variables are highly correlated with "month_cnt" except the price

CC=dataTrain[["item_price","month_cnt","month"]].corr()#"item_freq",
CC
#             item_freq  category_id  item_price  month_cnt
#item_freq     1.000000    -0.073820    0.067416   0.521578
#category_id  -0.073820     1.000000   -0.228345  -0.010741
#item_price    0.067416    -0.228345    1.000000   0.022186
#month_cnt     0.521578    -0.010741    0.022186   1.000000

# Transform it in a links data frame (3 columns only):
links = C.stack().reset_index()
links.columns =["var1","var2","corr_val"]
# remove self correlation (cor(A,A)=1)
links_filtered=links.loc[ (links['var1'] != links['var2']) ]
links_filtered
# Build your graph
#G = nx.Graph()
G = nx.path_graph(0)
graph = {"freq":["price","count"],"price":["freq","count"],
         "count":["price","freq"]}
leng=1
#[('freq', 'price'), ('freq', 'count'), ('price', 'count')]
values=[0.067,0.522,0.022]
for vertex, edges in graph.items():
    G.add_node("%s" % vertex)
#    leng+=1
    for edge in edges:
        G.add_node("%s" % edge)
        G.add_edge("%s" % vertex, "%s" % edge, weight = leng)
#        print("'%s' connects with '%s'" % (vertex,edge))
# Create positions of all nodes and save them
#pos = nx.spring_layout(G)
pos={"price": [1.5,1.5],"freq": [0.5,1.5],"count": [1,1]}
labels ={('freq', 'price'): values[0], ('freq', 'count'): values[1], 
         ('price', 'count'): values[2]}
# Draw the graph according to node positions
nx.draw(G, pos, with_labels=True,node_size=3000)
# Create edge labels
#labels = {edg: str(values[G.edges[edg]]) for edg in G.edges}
# Draw edge labels according to node positions
pos_lab={"price": [1.25,1.25],"freq": [0.75,1.25],"count": [1,1.5]}
nx.draw_networkx_edge_labels(G, pos,font_color='red',edge_labels=labels)
plt.axis('off')
plt.show()
################
#import statsmodels.formula.api as smf
# Instantiate a gamma family model with the default link function.
#poisson_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
#form="month_cnt ~ date_block_num + item_id + shop_id + item_freq + category_id + month + item_price"
#form="month_cnt ~ date_block_num + item_freq + month + item_price"
#poisson_model = smf.glm(formula=form, data=dataTrain, family=sm.families.Poisson())
#poisson_fit = poisson_model.fit()
#dir(poisson_fit.mle_settings)
#poisson_fit.use_t
#print(poisson_fit.summary())
#
#             Generalized Linear Model Regression Results                  
#==============================================================================
#Dep. Variable:              month_cnt   No. Observations:               921400
#Model:                            GLM   Df Residuals:                   921392
#Model Family:                 Poisson   Df Model:                            7
#Link Function:                    log   Scale:                          1.0000
#Method:                          IRLS   Log-Likelihood:*                   -inf
#Date:                Fri, 15 Nov 2019   Deviance: *                  8.7344e+05
#Time:                        18:15:41   Pearson chi2:                 3.83e+06
#No. Iterations:                     7   *non-defined for Poisson family                               
#Covariance Type:            nonrobust        * non defined for scale=1                                 
#==================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
#----------------------------------------------------------------------------------
#Intercept          0.5517      0.003    163.637      0.000       0.545       0.558
#date_block_num     0.0013      0.000     10.540      0.000       0.001       0.002
#item_id        -9.174e-06   1.23e-07    -74.511      0.000   -9.41e-06   -8.93e-06
#shop_id           -0.0012   4.26e-05    -27.026      0.000      -0.001      -0.001
#item_freq          0.1936   8.63e-05   2244.772      0.000       0.193       0.194
#category_id       -0.0055    4.5e-05   -123.243      0.000      -0.006      -0.005
#month              0.0017      0.000      7.667      0.000       0.001       0.002
#item_price      1.289e-05   3.19e-07     40.347      0.000    1.23e-05    1.35e-05
#==================================================================================
# item_id, category_id have small weight


#                 Generalized Linear Model Regression Results                  
#==============================================================================
#Dep. Variable:              month_cnt   No. Observations:               921400
#Model:                            GLM   Df Residuals:                   921395
#Model Family:                 Poisson   Df Model:                            4
#Link Function:                    log   Scale:                          1.0000
#Method:                          IRLS   Log-Likelihood:                   -inf
#Date:                Fri, 15 Nov 2019   Deviance:                   9.1019e+05
#Time:                        18:40:30   Pearson chi2:                 3.78e+06
#No. Iterations:                     7                                         
#Covariance Type:            nonrobust                                         
#==================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
#----------------------------------------------------------------------------------
#Intercept          0.2137      0.003     81.395      0.000       0.209       0.219
#date_block_num     0.0004      0.000      3.000      0.003       0.000       0.001
#item_freq          0.1881   8.02e-05   2346.055      0.000       0.188       0.188
#month              0.0024      0.000     11.216      0.000       0.002       0.003
#item_price      2.899e-05   2.82e-07    102.951      0.000    2.84e-05    2.95e-05
#==================================================================================
# item_freq is obviously the larger coeff
filePath="working_data/"+"1C_ctrl_training.csv"
data2=pd.read_csv(filePath, index_col=False) 
data2=data2.drop(["Unnamed: 0",'Unnamed: 0.1', 'Unnamed: 0.1.1'],axis=1)
data2.keys()
data2.head()
dataCtrl=ct.my_prepareTrain(data2)
dataCtrl.keys()

dataCtrlHM=ct.my_summaryHistoricFunc(dataCtrl,f_mean=True,f_sum=False) #takes almost 10 minutes
#dataCtrl=pd.get_dummies(dataCtrl)
dataCtrl.reset_index()
dataCtrlHM.reset_index()

C=pd.merge(dataCtrl,dataCtrlHM,how="left",on=["date_block_num","item_id","shop_id"])

#target=dataCtrl["month_cnt"]
#dataCtrl=dataCtrl.drop("month_cnt",axis=1)
#predictions=poisson_fit.predict(exog=dataCtrl, transform=True)
#err=abs(target-predictions)
#err.plot()
#err.mean()
#err.max()
#err.min()
#
#rmse=my_rmse(target,predictions) #15.141159663472205
## not that bad...i should see the mean, std of the counts
#poisson_fit.params
#poisson_fit

dataTrainHM=ct.my_summaryHistoricFunc(dataTrain,f_mean=True,f_sum=False) #15:54-15:09
#dataTrainHM=ct.my_summaryHistoMean(dataTrain) #takes almost 10 minutes
#dataTrain=pd.get_dummies(dataTrain)
dataTrain.reset_index()
dataTrainHM.reset_index()

D=pd.merge(dataTrain,dataTrainHM,how="left",on=["date_block_num","item_id","shop_id"])
#D=D.drop("histo_f_cnt",axis=1)

CC=D.corr()
CC["month_cnt"]
sum(abs(CC.values)>0.4)

#models_param=[["month_cnt ~ date_block_num + item_freq + month + item_price","GLM","poisson"],
#        ["month_cnt ~ date_block_num + item_id + shop_id + item_freq + category_id + month + item_price","GLM","poisson"]
#        ]

models_param=[[D.keys(),"GLM","poisson"]]#,[D.keys(),"GLM","poisson"]
i=0
modelRes=pd.DataFrame(columns=["model","formula","family","aic",
                               "scale","log-likel","deviance","chi2",
                               "mean_err_perc","sign_pval_perc",
                               "rmse_in","rmse_out","acc_in","acc_out"])

for i in range(0,len(models_param)):
    aux=ct.my_compareFitModels(D,models_param[i][0],models_param[i][1],models_param[i][2],C)
    modelRes=modelRes.append(aux,sort=False).reset_index() #18:1018:13

modelRes.iloc[0:1,0:11]

[y,X]=ct.my_df2arry_endo_exog(D,"month_cnt")
model = sm.GLM(y,X, family=sm.families.Poisson())
fitModel=model.fit(method='nm', maxiter=100, maxfun=100)#18:15-18:16
predictions=fitModel.predict(exog=X, transform=True)
err=abs(y-predictions)
acc=100*(len([e for e in err if e<1])/len(err)) # <1:53,74%  <2: 88,28%
acc
err.mean()
#rmse_in=ct.my_calculateAccuracy(dataTrain,"month_cnt",fitModel)
#rmse_out=ct.my_calculateAccuracy(dataTest,"month_cnt",fitModel)
import my_functions_1c as ct
fitModel.summary()
#Dep. Variable:                      y   No. Observations:               921400
#Model:                            GLM   Df Residuals:                   921393
#Model Family:                 Poisson   Df Model:                            6
#Link Function:                    log   Scale:                          1.0000
#Method:                            nm   Log-Likelihood:                   -inf
#Date:                Sat, 30 Nov 2019   Deviance:                   1.0246e+07
#Time:                        18:24:07   Pearson chi2:                 2.08e+07
#No. Iterations:                   556                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#x1             0.0211   8.18e-05    258.570      0.000       0.021       0.021
#x2         -1.245e-05   9.22e-08   -134.923      0.000   -1.26e-05   -1.23e-05
#x3             0.0075   3.68e-05    204.837      0.000       0.007       0.008
#x4            -0.0013   3.61e-05    -35.865      0.000      -0.001      -0.001
#x5             0.0181      0.000     97.250      0.000       0.018       0.018
#x6           4.68e-05    2.6e-07    180.244      0.000    4.63e-05    4.73e-05
#x7             0.0112   1.04e-06   1.07e+04      0.000       0.011       0.011
#==============================================================================


[y,X]=ct.my_df2arry_endo_exog(D,"month_cnt")
rfModel=RandomForestRegressor(n_estimators=500,max_depth=10,random_state=18)
rfFit=rfModel.fit(X,y) #17:09-17:26
pred=rfFit.predict(X) #17:26-17:27

err=abs(y-pred)
err2=y-pred
np.mean(err) #1.3330819427844776
np.max(err) #1166.5251575783172
np.min(err) #e-05
100*(len([e for e in err2 if e>0])/len(err2)) #25.951378337312786
100*(len([e for e in err2 if e<0])/len(err2)) #74.04862166268722
100*(len([e for e in err if e<1])/len(err)) #69.638, tuning: 72.93043195137834
100*(len([e for e in err if e<2])/len(err)) # n_estimators=500 :88.0572
np.mean([e for e in err2 if e<0]) #-0.89704447

dataCtrlHM=ct.my_summaryHistoricFunc(dataCtrl,f_mean=True,f_sum=True) #takes almost 10 minutes
#dataCtrl=pd.get_dummies(dataCtrl)
dataCtrl.reset_index()
dataCtrlHM.reset_index()

C=pd.merge(dataCtrl,dataCtrlHM,how="left",on=["date_block_num","item_id","shop_id"])

[y_c,X_c]=ct.my_df2arry_endo_exog(C,"month_cnt")
rfFit_c=rfModel.fit(X_c,y_c)
pred_c=rfFit_c.predict(X_c)
err_c=abs(y_c-pred_c)
err2_c=y_c-pred_c
np.mean(err_c) #1.3580242780446712
np.max(err_c) # 442.8908487407861
np.min(err_c) # e-05
100*(len([e for e in err2_c if e>0])/len(err2_c)) #24.33243568640908
100*(len([e for e in err2_c if e<0])/len(err2_c)) # 75.66756431359092
100*(len([e for e in err_c if e<1])/len(err_c)) # 68.2189366664534
np.mean([e for e in err2_c if e<0]) #-0.89704447

sns.set(rc={'figure.figsize':(12,6)})
fig, ax1 = plt.subplots(2, 3)
sns.regplot(x=y,y=pred,ax=ax1[0,0])
ax1[0,0].set_title("[train] x: actual | y: predicted")
sns.regplot(y,err,ax=ax1[0,1])
ax1[0,1].set_title("[train] x: actual | y: abs(act-pred)")
sns.regplot(y,err2,ax=ax1[0,2])
ax1[0,2].set_title("[train] x: actual | y: act-pred")

sns.regplot(y_c,pred_c,ax=ax1[1,0])
ax1[1,0].set_title("[control] x: actual | y: predicted")
sns.regplot(y_c,err_c,ax=ax1[1,1])
ax1[1,1].set_title("[control] x: actual | y: abs(act-pred)")
sns.regplot(y_c,err2_c,ax=ax1[1,2])
ax1[1,2].set_title("[control] x: actual | y: act-pred")

fig

#########   plot tree
from sklearn.tree import export_graphviz
import pydot
featureNames=[col for col in dataTrain.columns if col != "month_cnt"]

tree=rfModel.estimators_[1]
export_graphviz(tree,out_file="tree.dot",rounded=True,precision=1,
                feature_names=featureNames)

(graph,)=pydot.graph_from_dot_file("tree.dot")
graph.write_png("tree.png")


featureNames=[col for col in D.columns if col != "month_cnt"]
featImp=list(rfModel.feature_importances_)
feat_imp=[(feat,round(imp,2)) for (feat,imp) in zip(featureNames,featImp)]
feat_imp

# linear regression actual vs err (non absolute)
from sklearn.linear_model import LinearRegression
model_pred = LinearRegression().fit(y.reshape(-1,1), pred)
model_pred.coef_ #0.67778392 with pred
model_err = LinearRegression().fit(y.reshape(-1,1), err2)
model_err.coef_ #0.32221608 with err
#model.intercept_ #0.47178088839165866 with err
r2p = model_pred.score(y.reshape(-1,1), pred) # 0.7592726478685866
r2e = model_err.score(y.reshape(-1,1), err2) #0.41616991455938823


#df=ct.my_summaryHistoMean(dataTrain,)
#
#aa=ct.my_historicMean(dataTrain,"month_cnt")
import my_functions_1c as ct
aaa=ct.my_historicMean(dataTrain,14,"month_cnt",replaceNaN=True)
aa=ct.my_historicMean(dataTrain,12,"month_cnt",replaceNaN=True)

DD=pd.DataFrame({"date_block_num":[1,1,1,1,2,2,2,2,3,3,3,3],
                 "item_id":[10,15,20,25,10,30,35,40,10,20,30,55],
                 "shop_id":[200,200,203,203,200,210,212,212,200,203,210,230],
                 "month_cnt":[1,2,3,4,2,4,6,8,3,6,9,12]})

bb=ct.my_summaryHistoMean(DD)
bb
bb2=ct.my_historicMean(DD,3,"month_cnt")
bb2=ct.my_historicSum(DD,3,"month_cnt")
bb2
#item_shop=DD[DD["date_block_num"]==1][["item_id","shop_id"]].values.tolist()
#DDg=DD.groupby(["item_id","shop_id"])
#DD2=pd.concat(DDg.get_group(tuple(g)) for g in item_shop)
#DD[DD[["item_id","shop_id"]]==item_shop[0]][["item_id","shop_id"]]
print(DD)
####################### eliminate item_freq
#dataTrain2=dataTrain.drop("item_freq",axis=1)
#dataTrain2.keys()
#rfModel2=RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=18)
#[y2,X2]=ct.my_df2arry_endo_exog(dataTrain2,"month_cnt")
#
#
#rfFit2=rfModel2.fit(X2,y2)
#pred2=rfFit2.predict(X2)
#err2=abs(y2-pred2) 
#err22=y2-pred2 
#np.mean(err2) #1.3341
#np.max(err2) #1212.77
#np.min(err2) #0.000115
#100*(len([e for e in err22 if e>0])/len(err22)) #25.72747992185804
#100*(len([e for e in err22 if e<0])/len(err22)) #74.27252007814195
#100*(len([e for e in err2 if e<1])/len(err2)) #69.00672889081832
#
#
#dataCtrl2=dataCtrl.drop("item_freq",axis=1)
#[y2_c,X2_c]=ct.my_df2arry_endo_exog(dataCtrl2,"month_cnt")
#rfFit2_c=rfModel2.fit(X2_c,y2_c)
#pred2_c=rfFit2_c.predict(X2_c)
#err2_c=abs(y2_c-pred2_c)
#err22_c=y2_c-pred2_c
#np.mean(err2_c) #1.3543138339118084
#np.max(err2_c) #444.1032
#np.min(err2_c) #1.4240848584812227e-06
#100*(len([e for e in err22_c if e>0])/len(err22_c)) #24.342905002588246
#100*(len([e for e in err22_c if e<0])/len(err22_c)) #75.65709499741176
#100*(len([e for e in err2_c if e<1])/len(err2_c)) # 68.27390057639403
#np.mean([e for e in err2_c if e<0])
#
#sns.set(rc={'figure.figsize':(12,6)})
#fig, ax1 = plt.subplots(2, 3)
#sns.regplot(x=y2,y=pred2,ax=ax1[0,0])
#ax1[0,0].set_title("[train] x: actual | y: predicted")
#sns.regplot(y2,err2,ax=ax1[0,1])
#ax1[0,1].set_title("[train] x: actual | y: abs(act-pred)")
#sns.regplot(y2,err22,ax=ax1[0,2])
#ax1[0,2].set_title("[train] x: actual | y: act-pred")
#
#sns.regplot(y2_c,pred2_c,ax=ax1[1,0])
#ax1[1,0].set_title("[control] x: actual | y: predicted")
#sns.regplot(y2_c,err2_c,ax=ax1[1,1])
#ax1[1,1].set_title("[control] x: actual | y: abs(act-pred)")
#sns.regplot(y2_c,err22_c,ax=ax1[1,2])
#ax1[1,2].set_title("[control] x: actual | y: act-pred")
#
#fig
#
#from sklearn.linear_model import LinearRegression
#model = LinearRegression().fit(y2.reshape(-1,1), err2)
#model.coef_ #0.63934656 with pred
#model.coef_ #0.381 with err
#model.intercept_ #0.47178088839165866 with err
#r2 = model.score(y2.reshape(-1,1), pred2) # 0.7181413899627395
#r2 = model.score(y2.reshape(-1,1), err2) #0.5377096163365533
##########   plot tree
#from sklearn.tree import export_graphviz
#import pydot
#featureNames2=[col for col in dataTrain2.columns if col != "month_cnt"]


tree=rfModel.estimators_[1]
export_graphviz(tree,out_file="tree.dot",rounded=True,precision=1,
                feature_names=featureNames)

(graph,)=pydot.graph_from_dot_file("tree.dot")
graph.write_png("tree.png")

featImp2=list(rfModel2.feature_importances_)
feat_imp2=[(feat,round(imp,2)) for (feat,imp) in zip(featureNames2,featImp2)]

df=pd.get_dummies(dataTrain)


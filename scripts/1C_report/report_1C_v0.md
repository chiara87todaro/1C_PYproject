# A report on sales forcasting
### Kaggle project

## Synopsis
Sales forecasting is a common problem that can be easily? managed using 
Machine Learning techniques.
In this report, sales records made by the Russian software developer, 
publisher and distributor company *1C Company* have been analysed. 
Based on past sales, the amount of items that will be sold in a certain 
shop is predicted
The forecast accuracy is XXX% with an error of XXXX.
==============================

```{python, echo=False}
# IMPORT MODULES & SET PATH
import os
import numpy as np # scientific calculation
import pandas as pd # data analysis
import matplotlib.pyplot as plt # data plot
import seaborn as sns # data plot 
import statsmodels.api as sm
#import networkx as nx
from sklearn.ensemble import RandomForestRegressor

import my_functions_1c as ct 

mainPath="/home/chiara/kaggle/1C_PYproject/scripts/"
os.chdir(mainPath)
```
## Data
```{python}
filePath="/home/chiara/kaggle/1C_PYproject/data/competitive-data-science-predict-future-sales/"+"sales_train_v2.csv"
data=pd.read_csv(filePath, index_col=False) 
data.head(5)
data.tail(5)
```

As shown above, data consist in  `python data.shape[0]` observations: 
each one indicates how many (*item_cnt_day*) items (*item_id*) have been 
sold in a certain day (*date*), in a certain shop (*shop_id*), at a certain 
price (*item_price*). An incremental index is assigned to each month: sales 
have been recorded from January 2013 to October 2015.





```python
filePath="working_data/"+"1C_small_training.csv"
data=pd.read_csv(filePath, index_col=False) 
data=data.drop("Unnamed: 0",axis=1)
```
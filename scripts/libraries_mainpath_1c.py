#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:37:01 2019

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

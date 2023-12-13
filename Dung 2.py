#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Tên đề tài: nghiên cứu tác động của một số yếu tố đến xu hướng khách hàng rời bỏ ngân hàng trong tương lai
#Đọc thư viện
#install các thư viện:  plotnine,  -U imbalanced-learn, imblearn
import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from collections import Counter
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#import imblearn

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('tableau-colorblind10')
import seaborn as sns
from plotnine import *
import plotly.graph_objects as go
import plotly.express as px
df = pd.read_csv("C:\\Users\\Admin\\Downloads\\BankChurners.csv")
df.head()


# In[5]:


df.info()


# In[7]:


#Loại bỏ các biến sau, bởi các biến không ảnh hưởng đến dự án nghiên cứu của nhóm
df = df.drop(columns =['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
df.info()


# In[8]:


#Tìm các giá trị không xác định trong các cột
df.isna().sum()


# In[9]:


#Xóa các giá trị không xác định
df_cleaned = df.dropna()
#Hiển thị DataFrame sau khi xóa các giá trị không xác định
print("C:\\Users\\Admin\\Downloads\\BankChurners.csv")
print(df_cleaned)


# In[ ]:





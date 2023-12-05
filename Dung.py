#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


df.info()


# In[8]:


len(df['CLIENTNUM'].unique())


# In[12]:


#Loại bỏ các biến sau, bởi các biến không ảnh hưởng đến dự án nghiên cứu của nhóm
#Gender - giới tính của khách hàng
#Education_Level - Trình độ học vấn của khách hàng, biến không ảnh làm ảnh h
#Marital_Status - Tình trạng hôn nhân của khách hàng
#Income_Category - Mức thu nhập của khách hàng
#Card_Category - Loại thẻ tín dụng của khách hàng
#Avg_Utilization_Ratio - Tỷ lệ sử dụng trung bình của thẻ tín dụng
#Dependent_count - Số người khách hàng 
#Avg_Utilization_Ratio - Tỷ lệ sử dụng trung bình của thẻ tín dụng.
#Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
#Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2

#Trong tổng người được khảo sát, có 16% số người không cung cấp thông tin thuộc một trong ba biến Education level, Income, Marital

df = df.drop(columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Avg_Utilization_Ratio', 'Dependent_count', 'Avg_Utilization_Ratio', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
df.info()


# In[ ]:





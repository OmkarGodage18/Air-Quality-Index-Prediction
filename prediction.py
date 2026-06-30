#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from data_preprocessing import preprocess


# In[2]:


live_data= pd.read_csv('live_data.csv')


# In[5]:


model = joblib.load('air_quality_model.pkl')


# In[4]:


live_data= preprocess(live_data)


# In[7]:


x_live=live_data.drop(columns='AQI')
y_live = live_data['AQI']

y_pred_live = model.predict(x_live)


# In[ ]:





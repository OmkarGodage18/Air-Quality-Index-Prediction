#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
print(sys.executable)



# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


# In[3]:


def prepare_train_val_data(train_data,val_data):

    x_train=train_data.drop(columns='AQI')
    y_train=train_data['AQI']

    x_val = val_data.drop(columns='AQI')
    y_val = val_data['AQI']

    return x_train, x_val , y_train, y_val


# In[4]:


def fit_and_evaluate_model(x_train, x_val, y_train, y_val):
    model = RandomForestRegressor(
        n_estimators=400,       
        max_depth=15,          
        min_samples_split=5,   
        min_samples_leaf=2,     
        random_state=42,
        n_jobs=-1)

    model.fit(x_train,y_train)
    y_pred= model.predict(x_val)

    r2 = r2_score(y_val,y_pred)
    mae = mean_absolute_error(y_val,y_pred)
    mse=mean_squared_error(y_val,y_pred)

    print("R2 Score:", r2)
    print("MAE:", mae)
    print("MSE:", mse)


    return model




# In[5]:


def get_important_features(model, features):

    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })

    importances = importances.sort_values(by='importance', ascending=False)

    return importances


# In[6]:


# feat_imp = get_important_features(model, X_train.columns)
# print(feat_imp)


# In[7]:


#get_ipython().system('jupyter nbconvert --to script model_building.ipynb')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3

# 1. Connect to the database
conn = sqlite3.connect(r"C:\Users\admin\Downloads\Regression.db")
cursor = conn.cursor()

# 2. Query the internal SQLite master table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# 3. Fetch and print names
tables = cursor.fetchall()

print("Tables in this database:")
for table in tables:
    print(f"- {table[0]}")

# 4. Close connection
conn.close()


# In[2]:


import sqlite3
import pandas as pd

# 1. Establish connection
db_path = r"C:\Users\admin\Downloads\Regression.db"
conn = sqlite3.connect(db_path)





        # 3. Read the table into a DataFrame
data = pd.read_sql_query(f"SELECT * FROM air_quality_index", conn)



# finally:
    # 4. Always close the connection
    # conn.close()


# In[3]:


data=data[data['City']=='Brasilia']


# In[4]:


data.head(10)


# In[5]:


data.describe(include='all')


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data['CO2']=data['CO2'].fillna(data['CO2'].median())


# In[9]:


data['City'].value_counts()


# In[10]:


data.columns


# In[17]:


def preprocess(data):

    data = data.copy() 
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data['Date'] = data['Date'].dt.tz_convert('America/Sao_Paulo')



    data['Date']=pd.to_datetime(data['Date'])
    data['hour']= data['Date'].dt.hour
    data['day']=data['Date'].dt.day
    data['month']=data['Date'].dt.month

    data.drop(columns=['Date','City'], inplace=True)
    return data




# In[12]:


# data= data.to_csv('full_data.csv', index=False)


# In[13]:


# data = data.sort_values('Date')


# In[14]:


# train_data=data.iloc[0:5270]
# val_data=data.iloc[5270:7027]
# live_data=data.iloc[7027:]


# In[15]:


# train_data=train_data.to_csv('train_data.csv', index=False)
# val_data=val_data.to_csv('val_data.csv', index=False)
# live_data=live_data.to_csv('live_data.csv', inde)


# In[16]:


get_ipython().system('jupyter nbconvert --to script data.preprocessing.ipynb')


# In[ ]:





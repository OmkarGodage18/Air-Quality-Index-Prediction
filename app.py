#!/usr/bin/env python
# coding: utf-8

# In[10]:


from flask import Flask, render_template,request
import pandas as pd
import joblib
from data_preprocessing import preprocess


# In[11]:


app = Flask(__name__)


# In[12]:


model = joblib.load("air_quality_model.pkl")


# In[13]:


live_data=pd.read_csv('live_data.csv')


# In[14]:


@app.route('/')
def home():
    random_row = live_data.sample(1).to_dict(orient='records')[0]
    return render_template('AQI.html', data=random_row, prediction=None, error=None)


# In[19]:


@app.route('/get_row', methods=['POST'])
def get_row():
    try:
        row_index = int(request.form['row_index'])  

        random_row = live_data.iloc[row_index].to_dict()

        return render_template(
            'AQI.html',
            data=random_row
        )

    except Exception as e:
        return render_template(
            'AQI.html',
            error=str(e),
            data={}
        )



# In[ ]:

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.form.to_dict()
        input_df = pd.DataFrame([data])

        input_df['City'] = 'Brasilia'
        numeric_cols = ['CO','CO2','NO2','SO2','O3','PM2.5','PM10']

        for col in numeric_cols: 
            input_df[col] = pd.to_numeric(input_df[col],errors='coerce')

        input_df = preprocess(input_df)

        prediction = model.predict(input_df)[0]

        return render_template(
            "AQI.html",
            prediction=f"Predicted AQI: {round(prediction,2)}",
            data=data
        )

    except Exception as e:
        return render_template(
            "AQI.html",
            error=str(e),
            data=request.form
        )
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from data_preprocessing import preprocess
from model_building import prepare_train_val_data,fit_and_evaluate_model,get_important_features


# In[ ]:


train_data= pd.read_csv('train_data.csv')
val_data=pd.read_csv('val_data.csv')
live_data=pd.read_csv('live_data.csv')


# In[ ]:


train_data= preprocess(train_data)
val_data=preprocess(val_data)
live_data=preprocess(live_data)


# In[ ]:


x_train ,x_val, y_train, y_val = prepare_train_val_data(train_data, val_data)


# In[ ]:


model = fit_and_evaluate_model(x_train, x_val, y_train, y_val)


# In[ ]:


feat_imp = get_important_features(model, x_train.columns)
print(feat_imp)


# In[ ]:


import joblib
joblib.dump(model,'air_quality_model.pkl')


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predictions
y_pred = model.predict(x_val)

# Metrics
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

metrics = ["MAE", "RMSE", "R² Score"]
values = [mae, rmse, r2]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.title("Model Performance")
plt.ylabel("Score")

for i,v in enumerate(values):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig("Accuracy.png", dpi=300)
plt.show()
import matplotlib.pyplot as plt
import pandas as pd

importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values("Importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=300)
plt.show()
import matplotlib.pyplot as plt

y_pred = model.predict(x_val)

plt.figure(figsize=(7,6))
plt.scatter(y_val, y_pred)

plt.plot(
    [y_val.min(), y_val.max()],
    [y_val.min(), y_val.max()],
    'r--'
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")

plt.tight_layout()
plt.savefig("Prediction.png", dpi=300)
plt.show()
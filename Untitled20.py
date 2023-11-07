#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb


# In[2]:


# Create a page dropdown
page = st.sidebar.selectbox(" ”” Hello there! I’ll guide you! Please select model””” ", ["Main Page", "Linear Regressor", 
          "XGB Regressor","LGBM Regressor","Compare Models"])


# In[64]:


data = pd.read_csv('titanic_read.csv')


# In[61]:


def report_metric(pred, test, model_name):
     # Creates report with mae, rmse and r2 metric and returns as df
     mae = mean_absolute_error(pred, test)
     mse = mean_squared_error(pred, test)
     rmse = np.sqrt(mse)
     r2 = r2_score(test, pred)
     metric_data = {"Metric": ["MAE", "RMSE", "R2"], model_name: [mae, rmse, r2]}
     metric_df = pd.DataFrame(metric_data)
     return metric_df


# In[62]:


def plot_preds(data_date,test_date, target, pred):
     # Plots prediction vs real 
     fig = plt.figure(figsize=(20,10))
     plt.plot(data_date, target, label = 'Real')
     plt.plot(test_date, pred, label = 'Pred')
     plt.legend()
     st.pyplot(fig)


# In[65]:


# Split train test and define test period.
test_period = -20
test = data[test_period:]
train = data[:test_period]


# In[66]:


x_trainm1 = train[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y_trainm1 = train[["Survived"]]
x_testm1 = test[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y_testm1 = test[["Survived"]]
lr = LinearRegression()
lr.fit(x_trainm1, y_trainm1)
m1pred = lr.predict(x_testm1)
metric1 = report_metric(m1pred, y_testm1, "Linear Regression")


# In[67]:


### Prepare for model 2 XGB Regressor
x_trainm2 = train[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y_trainm2 = train[["Survived"]]
x_testm2 = test[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y_testm2 = test[["Survived"]]
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# Fit the model
xgb.fit(x_trainm2, y_trainm2)
# Get prediction
m2pred = xgb.predict(x_testm2)
metric2 = report_metric(m2pred, y_testm2, "XGB Regression")


# In[73]:


if page == "Main Page":
     ### INFO
     st.title("Hello, welcome to sales predictor!")
     st.write("""
     This Applicaton Predicts the likelihhod of survial on board the Titanic with three different models: 
     # feautures used in prediction:
     - Date: date format time feature
     - col1: categorical feature
     - col2: second categorical feature
     - col3: third categorical feature
     - target: target variable to be predicted
                                    """)
     st.write("Lets plot sales data!")
    
elif page == "Linear Regressor":
     
    # Base model, it uses linear regression.
    st.title("Model 1: ")
    st.write("Model 1 works with linear regression as base model.")
    st.write("The columns it used are: 'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
    st.write(metric1)
    """
    ### Real vs Pred. Plot for 1. Model
    
    """
    plot_preds(data["Age"],test["Pclass"], data["Survived"], m1pred)
elif page == "XGB Regressor":
       # Model 2
      st.title("Model 2:")
      st.write("Model 2 works with XGB Regressor.")
      st.write("The columns it used are: 'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
      ### Real vs Pred. Plot for 2. Model
elif page == "XGB Regressor":
       # Model 2
      st.title("Model 2:")
      st.write("Model 2 works with XGB Regressor.")
      st.write("The columns it used are: 'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
      ### Real vs Pred. Plot for 2. Model
      plot_preds(data["Age"],test["sex"], data["Survived"], m2pred)
elif page == "Compare Models":
     # Compare models.
     st.title("Compare Models: ")
     st.title("Compare Models: ")
     all_metrics = metric1.copy()
     all_metrics["XGB Regression"] = metric2["XGB Regression"].copy()
     all_metrics["LGBM Regression"] = metric3["LGBM Regression"].copy()
     st.write(all_metrics)
     # Best Model
     st.title("Best Model /XGB Regressor: ")
     st.write("Lets plot best models predictions in detail.") 
     
     # Plot best model results.
     
     st.write("Best Model Predictions vs Real") 
    # Plot best model results.
     plot_preds(test["Sex"],test["Age"], test["Survived"], m2pred)
     # Show rowbase best result and real
     st.write("Best Model Predictions vs Real") 
     best_pred = pd.DataFrame(test[["Survived"]].copy()) 
     best_pred["pred"] = m2pred
     st.write(best_pred)
     


# In[ ]:





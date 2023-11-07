#!/usr/bin/env python
# coding: utf-8

# In[1]:


##import pandas as pd
##import numpy as np
##import matplotlib.pyplot as plt
##import plotly.express as px
import streamlit as st
##from sklearn import preprocessing
##from sklearn.linear_model import LinearRegression
##from xgboost import XGBRegressor
##from xgboost import plot_importance
##import matplotlib.pyplot as plt
##from sklearn.preprocessing import MinMaxScaler
##import lightgbm as lgb
##from lightgbm import LGBMRegressor
##from sklearn.metrics import r2_score
##from sklearn.metrics import mean_squared_error
##from sklearn.metrics import mean_absolute_error
##from sklearn.model_selection import GridSearchCV
import streamlit as st


# In[2]:


# Create a page dropdown
page = st.sidebar.selectbox(" ”” Hello there! I’ll guide you! Please select model””” ", ["Main Page", "Linear Regressor", 
          "XGB Regressor","LGBM Regressor","Compare Models"])


# In[19]:


if page == "Main Page":
     ### INFO
     st.title("Hello, welcome to sales predictor!")
     st.write("""
     This application predicts sales for the next 20 days with 3 different models
     # Sales drivers used in prediction:
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
    st.write("The columns it used are: col1, col2, col3,day_of_week, day_of_month, month, week_of_year, season")
    st.write(metric1)
    """
    ### Real vs Pred. Plot for 1. Model
    """
elif page == "XGB Regressor":
       # Model 2
      st.title("Model 2:")
      st.write("Model 2 works with XGB Regressor.")
      st.write("The columns it used are: col1, col2,col3,day_of_week, day_of_month,month, week_of_year, season")
      ### Real vs Pred. Plot for 2. Model
     
elif page == "Compare Models":
     # Compare models.
     st.title("Compare Models: ")
     
     # Best Model
     st.title("Best Model /XGB Regressor: ")
     st.write("Lets plot best models predictions in detail.") 
     
     # Plot best model results.
     
     st.write("Best Model Predictions vs Real") 
     


# In[ ]:





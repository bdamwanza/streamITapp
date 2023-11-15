#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
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
from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[107]:


# Create a page dropdown
page = st.sidebar.selectbox(" Hello there! I’ll guide you! Please select model ", ["Main Page", "Logistic Regression", 
          "XGB Regressor","LGBM Regressor","Compare Models"])


# In[72]:


data = pd.read_csv('titanic_read.csv')


# In[73]:


data = data.drop(columns=['PassengerId'])


# In[74]:


data.rename(columns={'male': 'Sex'}, inplace=True)


# In[75]:


def report_metric(pred, test, model_name):
     # Creates report with mae, rmse and r2 metric and returns as df
     mae = mean_absolute_error(pred, test)
     mse = mean_squared_error(pred, test)
     rmse = np.sqrt(mse)
     r2 = r2_score(test, pred)
     metric_data = {"Metric": ["MAE", "RMSE", "R2"], model_name: [mae, rmse, r2]}
     metric_df = pd.DataFrame(metric_data)
     return metric_df


# In[76]:


test_period = -20
test = data[test_period:]
train = data[:test_period]


# In[77]:


logmodel=LogisticRegression()


# In[78]:


x_train, x_test , y_train , y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size = 0.30 ,random_state = 101)


# In[79]:





# In[80]:





# In[81]:





# In[82]:


logmodel.fit(x_train,y_train)


# In[83]:


m1pred =logmodel.predict(x_test)


# In[86]:



metric1 = report_metric(m1pred, y_test, "Logistic Regression")


# In[87]:


print(metric1)


# In[88]:


x_tease = np.array([2,30,0,0,8.0292,1,1,0])
one=x_tease.reshape(1, -1)


# In[ ]:





# In[89]:


logmodel.predict(one)


# In[90]:


def show1(): 
    x_ax = range(len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, m1pred, label="predicted")
    plt.title("Titanic dataset test and predicted data")
    plt.xlabel('X')
    plt.ylabel('Price')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    st.pyplot()


# In[91]:


x_trainm2, x_testm2 , y_trainm2 , y_testm2 = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size = 0.30 ,random_state = 65)
##xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)


# In[92]:





# In[93]:





# In[ ]:





# In[94]:


learning_rate_range = np.arange(0.01, 1, 0.05)
test_XG = [] 
train_XG = []

for lr in learning_rate_range:
    xgb_classifier = xgb.XGBClassifier(learning_rate=lr)
    xgb_classifier.fit(x_train, y_train)
    train_XG.append(xgb_classifier.score(x_train, y_train))
    test_XG.append(xgb_classifier.score(x_test, y_test))


# In[95]:


m2pred = xgb_classifier.predict(x_testm2)
metric2 = report_metric(m2pred, y_testm2, "XGB Classification")


# In[96]:


print(metric2)


# In[97]:


xgb_classifier.predict(one)


# In[98]:


def show2(): 
    x_ax = range(len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_testm2, label="original")
    plt.plot(x_ax, m2pred, label="predicted")
    plt.title("Titanic dataset test and predicted data")
    plt.xlabel('X')
    plt.ylabel('Price')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 
    st.pyplot()


# In[65]:


x_trainm3, x_testm3 , y_trainm3 , y_testm3 = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size = 0.30 ,random_state = 101)


# In[66]:





# In[99]:


learning_rate_range = np.arange(0.01, 1, 0.05)
test_LGBM = [] 
train_LGBM = []

for lr in learning_rate_range:
    lgb_classifier = lgb.LGBMClassifier(learning_rate=lr)
    lgb_classifier.fit(x_trainm3, y_trainm3)
    train_accuracy = lgb_classifier.score(x_trainm3, y_trainm3)
    test_accuracy = lgb_classifier.score(x_testm3, y_testm3)
    train_LGBM.append(train_accuracy)
    test_LGBM.append(test_accuracy)
  
    # You can use the 'predictions' array as the predicted labels for the new dataset


# In[100]:


predictions = lgb_classifier.predict(one)


# In[101]:





# In[102]:


def show3(): 
    x_ax = range(len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_testm3, label="original")
    plt.plot(x_ax, m3pred, label="predicted")
    plt.title("Titanic dataset test and predicted data")
    plt.xlabel('X')
    plt.ylabel('Price')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    st.pyplot()


# In[104]:


lgb.plot_importance(lgb_classifier, height=.5)


# In[ ]:





# In[119]:


two=[]

if page == "Main Page":
     ### INFO
     st.title("Hello, Titanic Passenger Predictor !")
     st.write("""
     This Applicaton Predicts the likelihhod of survial on board the Titanic with three different models: 
     # feautures used in prediction include:
     - 'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S'
     - Survived: target variable to be predicted
                                    """)
     st.title("Enter Person's Pclass,Age,SibSp,Parch,Fare,Sex,Q,S to Get started")


     user_inputs = []

     for i in range(7):
        user_input = st.text_input(f"Enter {i + 1}:")
        if user_input:
            user_inputs.append(user_input)


     st.write("User Inputs:", user_inputs)
     
     two = user_inputs.copy()

     
    
elif page == "Logistic Regression":
     
    # Base model, it uses linear regression.
    st.title("Model 1: ")
    st.write("Model 1 works with linear regression as base model.")
    st.write("The columns it used are: 'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
    st.write(metric1)
    st.write(logmodel.predict(two))
    
    """
   
    
    
    
    
    """
   
elif page == "XGB Regressor":
       # Model 2
      st.title("Model 2:")
      st.write("Model 2 works with XGB Regressor.")
      st.write("The columns it used are:'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
      ### Real vs Pred. Plot for 2. Model
      st.write(metric2)
      st.write(xgb_classifier.predict(two))
      
elif page == "LGBM Regressor":
       # Model 2
      st.title("Model 3:")
      st.write("Model 3 works with LGBM Regressor.")
      st.write("The columns it used are:'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S' ")
      ### Real vs Pred. Plot for 2. Model
     
    
      st.write("the expected output for the users data",lgb_classifier.predict(two))  
      lgb.plot_importance(lgb_classifier, height=.5)
      
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
     
     
    


# In[ ]:





# In[ ]:





# In[ ]:





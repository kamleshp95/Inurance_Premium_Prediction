## Version Tracking packages

## Data Analysis packages
import numpy as np
import pandas as pd



## General Tools
import os
import re
import joblib
import json
import warnings


# sklearn library
import sklearn

### sklearn preprocessing tools
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer,FunctionTransformer,OneHotEncoder


# Error Metrics 
from sklearn.metrics import r2_score #r2 square
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  roc_curve,auc,accuracy_score,roc_auc_score
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb


#crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


#hyper parameter tunning
from sklearn.model_selection import GridSearchCV,cross_val_score,RandomizedSearchCV

# 
import pickle

df = pd.read_csv("insurance.csv")

df.duplicated().sum()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()

print("Total numeric columns are:", len(numeric_cols))
print(numeric_cols)

print("Total categorical columns are:", len(categorical_cols))
print(categorical_cols)

# standard scalar and one hot encoding

numeric_features = ['age', 'bmi', 'children']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_features = ['sex', 'smoker','region']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# train test split 
X = df.drop('expenses',axis=1)
y = df['expenses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_gb = Pipeline(steps=[('preprocessor',preprocessor),('gradient_boosting', GradientBoostingRegressor())])



# gradient boosting 

gradient_boosting_reg = Pipeline(steps=[('preprocessor', preprocessor),
                        ('gradient_boosting' ,
                         GradientBoostingRegressor(n_estimators=41, 
                                                   max_depth=3, 
                                                   min_samples_split=5,
                                                   random_state=42))])

gradient_boosting_reg.fit(X_train, y_train)

# Predicting the model
y_pred3 = gradient_boosting_reg.predict(X_test)

# Evaluation Metrics
gradient_boosting_mse = mean_squared_error(y_test, y_pred3)
gradient_boosting_rmse = mean_squared_error(y_test, y_pred3, squared=False)
gradient_boosting_r2_score = r2_score(y_test, y_pred3)

print(f"The Mean Squared Error using Gradient Boosting Regressor : {gradient_boosting_mse}")
print(f"The Root Mean Squared Error using Gradient Boosting Regressor : {gradient_boosting_rmse}")
print(f"The r2_sccore using Gradient Boosting Regressor : {gradient_boosting_r2_score}")

pickle.dump(gradient_boosting_reg, open('gbr_model_final.pkl', 'wb'))

with open ('gbr_model_final.pkl', 'rb') as grb:
    loaded_model= pickle.load(grb)
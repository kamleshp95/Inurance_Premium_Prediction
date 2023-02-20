import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sklearn

### sklearn preprocessing tools
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

with open ("model\gbr_model_final.pkl", 'rb') as grb:
    loaded_model= pickle.load(grb)

def get_prediction( age, sex, bmi,children,smoker, region):
    x_input= pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    prediction = loaded_model.predict(x_input)
    return prediction[0]        

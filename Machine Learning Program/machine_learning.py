import pandas as pd
import scipy.stats as stats
import numpy as np
from sqlalchemy import create_engine

url = "https://heartdisease4.s3.us-east-2.amazonaws.com/heart.csv"

df = pd.read_csv(url)


df = df[['thalach',
        'oldpeak',
         'ca',
         'cp',
         'exang',
         'chol',
         'age',
         'trestbps',
         'slope',
         'sex',
         'target'
        ]]

df = pd.get_dummies(df, columns=['ca',
                                     'cp',
                                    'exang',
                                    'sex',
                                    'slope'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df['age'].values.reshape(-1,1))
age_scaler = scaler.fit(df['age'].values.reshape(-1,1))

df['trestbps'] = scaler.fit_transform(df['trestbps'].values.reshape(-1,1))
trestbps_scaler = scaler.fit(df['trestbps'].values.reshape(-1,1))

df['chol'] = scaler.fit_transform(df['chol'].values.reshape(-1,1))
chol_scaler = scaler.fit(df['chol'].values.reshape(-1,1))

df['oldpeak'] = scaler.fit_transform(df['oldpeak'].values.reshape(-1,1))
oldpeak_scaler = scaler.fit(df['oldpeak'].values.reshape(-1,1))

df['thalach'] = scaler.fit_transform(df['thalach'].values.reshape(-1,1))
thalach_scaler = scaler.fit(df['thalach'].values.reshape(-1,1))

import pandas as pd
import scipy.stats as stats
import numpy as np
from sqlalchemy import create_engine
#from machine_learning import age_scaler, trestbps_scaler, chol_scaler, oldpeak_scaler, thalach_scaler
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

from sklearn.externals import joblib

age_scaler = joblib.load('age_scaler.sav')
trestbps_scaler = joblib.load('trestbps_scaler.sav')
chol_scaler = joblib.load('chol_scaler.sav')
oldpeak_scaler = joblib.load('oldpeak_scaler.sav')
thalach_scaler = joblib.load('thalach_scaler.sav')

#Load the saved model from the Jupyter Notebook
model = joblib.load('KNN_model.sav')

thalach = input('''Maximum Heart Rate Achieved (bpm):
    ''')
df = pd.DataFrame({'thalach': [thalach]})


oldpeak = input('''ST Depression Induced by Exercise Relative to Rest:
    ''')
df['oldpeak'] = oldpeak

ca = 5
while True:
    ca = input('''Number of Major Vessels (0-3) Colored by Fluoroscopy:
    ''')
    if ca in ['0','1','2','3']:
        if ca == '0':
            df['ca_0'] = 1
            df['ca_1'] = 0
            df['ca_2'] = 0
            df['ca_3'] = 0
            df['ca_4'] = 0
            break
        elif ca == '1':
            df['ca_0'] = 0
            df['ca_1'] = 1
            df['ca_2'] = 0
            df['ca_3'] = 0
            df['ca_4'] = 0
            break
        elif ca == '2':
            df['ca_0'] = 0
            df['ca_1'] = 0
            df['ca_2'] = 1
            df['ca_3'] = 0
            df['ca_4'] = 0
            break
        elif ca == '3':
            df['ca_0'] = 0
            df['ca_1'] = 0
            df['ca_2'] = 0
            df['ca_3'] = 1
            df['ca_4'] = 0
            break
        elif ca == '4':
            df['ca_0'] = 0
            df['ca_1'] = 0
            df['ca_2'] = 0
            df['ca_3'] = 0
            df['ca_4'] = 1
            break
    else:
        print('Invalid number of vessels')

cp = 5
while True:
    cp = input('''Chest Pain Type:
0: Typical Angina
1: Atypical Angina
2: Non-anginal Pain
3: Asymptomatic
    ''')
    if cp in ['0', '1', '2', '3']:
        if cp == '0':
            df['cp_0'] = 1
            df['cp_1'] = 0
            df['cp_2'] = 0
            df['cp_3'] = 0
            break
        elif cp == '1':
            df['cp_0'] = 0
            df['cp_1'] = 1
            df['cp_2'] = 0
            df['cp_3'] = 0
            break
        elif cp == '2':
            df['cp_0'] = 0
            df['cp_1'] = 0
            df['cp_2'] = 1
            df['cp_3'] = 0
            break
        elif cp == '3':
            df['cp_0'] = 0
            df['cp_1'] = 0
            df['cp_2'] = 0
            df['cp_3'] = 1
            break
    else:
        print('Invalid chest pain type')

exang = 5
while True:
    exang = input('''Exercise Induced Angina:
0: No
1: Yes
    ''')
    if exang in ['0', '1']:
        if exang == '0':
            df['exang_0'] = 1
            df['exang_1'] = 0
            break
        elif exang == '1':
            df['exang_0'] = 0
            df['exang_1'] = 1
            break
    else:
        print("Invalid input for Exercise Induced Angina")


chol = input('''Serum Cholesterol (mg/dl)
    ''')
df['chol'] = chol


age = input('''Age (years):
    ''')
df['age'] = age

trestbps = input('''Resting Blood Pressure (mmHg)
    ''')
df['trestbps'] = trestbps

slope = 5
while True:
    slope = input('''Slope of the Peak Excercise ST Segment
0: Upsloping
1: Flat
2: Downsloping
    ''')
    if slope in ['0', '1', '2']:
        if slope == '0':
            df['slope_0'] = 1
            df['slope_1'] = 0
            df['slope_2'] = 0
            break
        elif slope == '1':
            df['slope_0'] = 0
            df['slope_1'] = 1
            df['slope_2'] = 0
            break
        elif slope == '2':
            df['slope_0'] = 0
            df['slope_1'] = 0
            df['slope_2'] = 1
            break
    else:
        print("Invalid Slope of Peak Exercise ST Segment")

sex=5
while True:
    sex = input('''Sex:
0: Female
1: Male
    ''')
    if sex in ['0', '1']:
        if sex == '0':
            df['sex_0'] = 1
            df['sex_1'] = 0
            break
        elif sex == '1':
            df['sex_0'] = 0
            df['sex_1'] = 1
            break
    else:
        print("Invalid Sex")


df['age'] = age_scaler.transform(df['age'].values.reshape(1,-1))
df['trestbps'] = trestbps_scaler.transform(df['trestbps'].values.reshape(-1,1))
df['chol'] = chol_scaler.transform(df['chol'].values.reshape(-1,1))
df['oldpeak'] = oldpeak_scaler.transform(df['oldpeak'].values.reshape(-1,1))
df['thalach'] = thalach_scaler.transform(df['thalach'].values.reshape(-1,1))

df = df[['thalach',
'oldpeak',
'chol',
'age',
'trestbps',
'ca_0',
'ca_1',
'ca_2',
'ca_3',
'ca_4',
'cp_0',
'cp_1',
'cp_2',
'cp_3',
'exang_0',
'exang_1',
'sex_0',
'sex_1',
'slope_0',
'slope_1',
'slope_2']]

hooray = model.predict(df)

if hooray == 0:
    print('''Less than 50% Diameter Narrowing
    Unlikely to Have Heart Disease''')

if hooray == 1:
    print('''More than 50% Diameter Narrowing
    Likely to Have Heart Disease''')

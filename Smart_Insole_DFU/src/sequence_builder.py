import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv("Smart_Insole_DFU/data/feature_dataset.csv")


# --------------------------------
# Features
# --------------------------------

features = [

'FSR1',
'FSR2',
'FSR3',
'FSR4',

'TempLeft',
'TempRight',

'SpO2',

'Cadence',
'StanceTime',
'SwingTime',

'HeelPTI',
'MidPTI',
'ForePTI',
'ToePTI',

'TotalPTI',

'DF',
'TF',
'SF',

'EPTI'

]


target = 'RiskScore'


# --------------------------------
# Scale inputs
# --------------------------------

scaler = MinMaxScaler()

df[features] = scaler.fit_transform(df[features])


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models")

os.makedirs(model_path, exist_ok=True)

joblib.dump(
    scaler,
    os.path.join(model_path, "scaler.save")
)



# --------------------------------
# Build sequences
# --------------------------------

X = []
y = []


for session_id in df['SessionID'].unique():

    session = df[df['SessionID'] == session_id]

    session = session.sort_values("Time")


    X.append(

        session[features].values

    )


    y.append(

        session[target].mean()

    )



X = np.array(X)

y = np.array(y)



print("X shape :", X.shape)
print("y shape :", y.shape)



data_path = os.path.join(BASE_DIR, "data")

os.makedirs(data_path, exist_ok=True)

np.save(
    os.path.join(data_path, "X.npy"),
    X
)

np.save(
    os.path.join(data_path, "y.npy"),
    y
)


print("Saved successfully")
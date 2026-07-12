import pandas as pd
import numpy as np
import os


####################################################
# LOAD DATA
####################################################

BASE_DIR = os.path.dirname(
            os.path.dirname(
            os.path.abspath(__file__))
            )

DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(

        os.path.join(
            DATA_DIR,
            "raw_sensor_data.csv"
        )

)


####################################################
# PARAMETERS
####################################################

DT = 0.1


####################################################
# FEATURE EXTRACTION
####################################################

result = []


sessions = df.groupby("SessionID")


for sid, session in sessions:


    risk_label = session["RiskLabel"].iloc[0]


    ################################################
    # Cadence
    ################################################

    if risk_label == "Low":

        cadence = np.random.randint(95,111)

        stance = np.random.uniform(0.55,0.65)



    elif risk_label == "Medium":

        cadence = np.random.randint(80,96)

        stance = np.random.uniform(0.65,0.85)



    else:

        cadence = np.random.randint(60,81)

        stance = np.random.uniform(0.85,1.10)



    swing = max(

            0.1,

            (60/cadence)-stance

            )



    ################################################
    # PTIs
    ################################################

    heel_pti = np.sum(

            session["FSR1"]

            )*DT



    mid_pti = np.sum(

            session["FSR2"]

            )*DT



    fore_pti = np.sum(

            session["FSR3"]

            )*DT



    toe_pti = np.sum(

            session["FSR4"]

            )*DT



    total_pti = (

        heel_pti +

        mid_pti +

        fore_pti +

        toe_pti

    )



    ################################################
    # Distribution Factor
    ################################################

    DF = max(

        heel_pti,

        mid_pti,

        fore_pti,

        toe_pti

        )/total_pti




    ################################################
    # Temperature Factor
    ################################################

    temp_diff = abs(

        session["TempLeft"].mean()

        -

        session["TempRight"].mean()

    )


    TF = 1 + temp_diff/10




    ################################################
    # SpO2 Factor
    ################################################

    spo2 = session["SpO2"].mean()


    SF = 1 + (100-spo2)/50




    ################################################
    # EPTI
    ################################################

    EPTI = (

    (total_pti/2)

    *

    (1+0.5*DF)

    *

    TF

    *

    SF

    )




    ################################################
    # Scores
    ################################################

    EPTI_score = np.interp(

        EPTI,

        [1500,8000],

        [10,95]

    )



    Temp_score = np.interp(

        temp_diff,

        [0.2,3],

        [10,90]

    )



    SpO2_score = np.interp(

        spo2,

        [88,99],

        [90,10]

    )



    DF_score = np.interp(

        DF,

        [0.25,0.60],

        [10,80]

    )



    ################################################
    # Risk Score
    ################################################

    risk = (

        0.70*EPTI_score

        +

        0.15*Temp_score

        +

        0.10*SpO2_score

        +

        0.05*DF_score

    )


    risk += np.random.normal(0,3)


    risk = np.clip(

        risk,

        0,

        100

    )




    ################################################
    # Add features to all rows
    ################################################

    for _, row in session.iterrows():



        result.append([

            row["SessionID"],

            row["Time"],



            row["FSR1"],
            row["FSR2"],
            row["FSR3"],
            row["FSR4"],



            row["TempLeft"],
            row["TempRight"],



            row["SpO2"],



            cadence,

            stance,

            swing,



            heel_pti,
            mid_pti,
            fore_pti,
            toe_pti,



            total_pti,


            DF,


            TF,


            SF,


            EPTI,



            risk


        ])





####################################################
# DATAFRAME
####################################################


feature_df = pd.DataFrame(


    result,


    columns=[


        "SessionID",

        "Time",


        "FSR1",
        "FSR2",
        "FSR3",
        "FSR4",


        "TempLeft",
        "TempRight",


        "SpO2",


        "Cadence",

        "StanceTime",

        "SwingTime",



        "HeelPTI",
        "MidPTI",
        "ForePTI",
        "ToePTI",



        "TotalPTI",


        "DF",


        "TF",


        "SF",


        "EPTI",



        "RiskScore"

    ]

)



####################################################
# SAVE
####################################################


feature_df.to_csv(

    os.path.join(

        DATA_DIR,

        "feature_dataset.csv"

    ),

    index=False

)



print()

print("Feature Extraction Completed")

print()

print(feature_df.head())

print()

print("Shape : ",feature_df.shape)

import pandas as pd

df = pd.read_csv("Smart_Insole_DFU/data/feature_dataset.csv")

print(df.groupby("SessionID")["RiskScore"].first().describe())

print()

print(df.groupby("SessionID")["EPTI"].first().describe())

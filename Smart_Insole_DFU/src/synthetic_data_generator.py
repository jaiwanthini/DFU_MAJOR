import numpy as np
import pandas as pd
import os

np.random.seed(42)

####################################################
# CONFIGURATION
####################################################

SAMPLE_RATE = 10
SESSION_DURATION = 3

SAMPLES_PER_SESSION = SAMPLE_RATE * SESSION_DURATION

LOW_SESSIONS = 1000
MEDIUM_SESSIONS = 750
HIGH_SESSIONS = 750


####################################################
# SESSION GENERATOR
####################################################

def generate_session(session_id, risk):

    rows = []

    t = np.linspace(
        0,
        SESSION_DURATION,
        SAMPLES_PER_SESSION
    )

    ################################################
    # Risk-specific parameters
    ################################################

    if risk == "Low":

        baseline = 300
        amplitude = 70

        temp_left = np.random.uniform(31, 32)
        temp_diff = np.random.uniform(0.2, 0.8)

        spo2 = np.random.randint(96, 100)



    elif risk == "Medium":

        baseline = 450
        amplitude = 110

        temp_left = np.random.uniform(31, 33)
        temp_diff = np.random.uniform(1, 2)

        spo2 = np.random.randint(93, 96)



    else:

        baseline = 650
        amplitude = 180

        temp_left = np.random.uniform(32, 34)
        temp_diff = np.random.uniform(2, 3)

        spo2 = np.random.randint(88, 93)



    temp_right = temp_left + temp_diff


    ################################################
    # Generate one walking session
    ################################################

    for i, time in enumerate(t):

        progress = i / (SAMPLES_PER_SESSION - 1)

        ################################################
        # Physiological drift
        ################################################

        if risk == "Low":

            current_temp_left = (
                temp_left +
                np.random.normal(0, 0.02)
            )

            current_temp_right = (
                temp_right +
                np.random.normal(0, 0.02)
            )

            current_spo2 = spo2



        elif risk == "Medium":

            current_temp_left = (
                temp_left +
                0.2 * progress
            )

            current_temp_right = (
                temp_right +
                0.2 * progress
            )

            current_spo2 = (
                spo2 -
                int(progress * 1)
            )



        else:

            current_temp_left = (
                temp_left +
                0.5 * progress
            )

            current_temp_right = (
                temp_right +
                0.5 * progress
            )

            current_spo2 = (
                spo2 -
                int(progress * 2)
            )


        ################################################
        # Gait waveform
        ################################################

        x = 2 * np.pi * time


        heel = baseline + amplitude * (
            np.maximum(0, np.sin(x))
        ) ** 2


        mid = baseline + amplitude * (
            np.maximum(0, np.sin(x + 0.4))
        ) ** 2


        fore = baseline + amplitude * (
            np.maximum(0, np.sin(x + 0.8))
        ) ** 2


        toe = baseline + amplitude * (
            np.maximum(0, np.sin(x + 1.2))
        ) ** 2


        ################################################
        # Sensor noise
        ################################################

        heel += np.random.normal(0, 5)
        mid += np.random.normal(0, 5)
        fore += np.random.normal(0, 5)
        toe += np.random.normal(0, 5)


        ################################################
        # Save sample
        ################################################

        rows.append([

            session_id,

            round(time, 2),

            round(heel, 2),
            round(mid, 2),
            round(fore, 2),
            round(toe, 2),

            round(current_temp_left, 2),
            round(current_temp_right, 2),

            current_spo2,

            risk

        ])



    return rows


####################################################
# DATASET CREATION
####################################################

all_rows = []

session = 0


for _ in range(LOW_SESSIONS):

    all_rows.extend(

        generate_session(

            session,

            "Low"

        )

    )

    session += 1



for _ in range(MEDIUM_SESSIONS):

    all_rows.extend(

        generate_session(

            session,

            "Medium"

        )

    )

    session += 1



for _ in range(HIGH_SESSIONS):

    all_rows.extend(

        generate_session(

            session,

            "High"

        )

    )

    session += 1



####################################################
# DATAFRAME
####################################################

df = pd.DataFrame(

    all_rows,

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

        "RiskLabel"

    ]

)



####################################################
# SAVE
####################################################

BASE_DIR = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)


DATA_DIR = os.path.join(

    BASE_DIR,

    "data"

)


os.makedirs(

    DATA_DIR,

    exist_ok=True

)


csv_path = os.path.join(

    DATA_DIR,

    "raw_sensor_data.csv"

)


df.to_csv(

    csv_path,

    index=False

)



print(df.head())

print()

print("Dataset Created Successfully")

print("Shape :", df.shape)

print()

print("Saved at")

print(csv_path)

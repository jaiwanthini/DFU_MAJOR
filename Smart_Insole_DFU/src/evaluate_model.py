import os
import numpy as np

from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


X = np.load(
    os.path.join(DATA_DIR, "X.npy")
)

y = np.load(
    os.path.join(DATA_DIR, "y.npy")
)


model = load_model(

    os.path.join(

        MODEL_DIR,

        "dfu_lstm.keras"

    )

)


pred = model.predict(X)


print("\nFirst 10 Predictions\n")

for i in range(10):

    print(

        f"Actual    : {y[i]:.2f}"

    )



    print(

        f"Predicted : {pred[i][0]:.2f}"

    )



    print("----------------")
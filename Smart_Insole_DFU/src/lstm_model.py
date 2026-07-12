import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ==========================================================
# Reproducibility
# ==========================================================

np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# Load data
# ==========================================================

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))


print("\nDataset Shapes")
print("X :", X.shape)
print("y :", y.shape)


print("\ny statistics")

print("Min  :", y.min())
print("Max  :", y.max())
print("Mean :", y.mean())
print("Std  :", y.std())


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print("\nTraining")

print(X_train.shape)
print(y_train.shape)


print("\nTesting")

print(X_test.shape)
print(y_test.shape)


# ==========================================================
# Build Model
# ==========================================================

model = Sequential([

    Input(shape=(30,19)),


    LSTM(32),


    Dropout(0.2),


    Dense(

        16,

        activation='relu'

    ),


    Dense(

        1

    )

])


# ==========================================================
# Compile
# ==========================================================

model.compile(

    optimizer=Adam(

        learning_rate=0.001

    ),

    loss='mse',

    metrics=['mae']

)


model.summary()


# ==========================================================
# Callbacks
# ==========================================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=10,

    restore_best_weights=True

)



reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=5,

    min_lr=1e-6

)


# ==========================================================
# Train
# ==========================================================

history = model.fit(

    X_train,

    y_train,


    validation_split=0.2,


    epochs=100,


    batch_size=32,


    callbacks=[

        early_stop,

        reduce_lr

    ]

)


# ==========================================================
# Evaluate
# ==========================================================

loss, mae = model.evaluate(

    X_test,

    y_test

)


print("\n")
print("Test Loss :", loss)
print("Test MAE  :", mae)

# ==========================================================
# Save Model
# ==========================================================

model.save(

    os.path.join(

        MODEL_DIR,

        "dfu_lstm.keras"

    )

)

print("\nModel Saved Successfully")


# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history['loss']

)

plt.plot(

    history.history['val_loss']

)


plt.legend(

    [

        'Train',

        'Validation'

    ]

)


plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.title(

    "Training Curve"

)

plt.grid(True)

plt.show()

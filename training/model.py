"""
==========================================================
Smart Insole DFU Risk Prediction
LSTM Model Architecture
==========================================================
"""

import os
import sys
import tensorflow as tf

# Ensure project root is on path so config is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam

from config import LEARNING_RATE, RECURRENT_DROPOUT


# ==========================================================
# Build Model
# ==========================================================

def build_model(input_shape):

    model = Sequential(

        [

            Input(shape=input_shape),

            # --------------------------------------
            # First LSTM
            # --------------------------------------

            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    recurrent_dropout=RECURRENT_DROPOUT
                )
            ),

            BatchNormalization(),
            Dropout(0.30),

            # --------------------------------------
            # Second LSTM
            # --------------------------------------

            Bidirectional(
                LSTM(
                    32,
                    return_sequences=False,
                    recurrent_dropout=RECURRENT_DROPOUT
                )
            ),

            BatchNormalization(),
            Dropout(0.30),

            # --------------------------------------
            # Dense Layers
            # --------------------------------------

            Dense(
                32,
                activation="relu",
                kernel_regularizer=l2(1e-4)
            ),

            Dropout(0.30),

            Dense(
                3,
                activation="softmax"
            )

        ]

    )

    model.compile(
        # Task 4: Use Adam with gradient clipping
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        ),

        loss="categorical_crossentropy",

        # Task 5: Improve metrics
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


# ==========================================================
# Model Summary
# ==========================================================

def print_model_summary(input_shape):

    model = build_model(input_shape)

    print("\n")

    print("=" * 60)

    print("LSTM MODEL")

    print("=" * 60)

    model.summary()

    print("=" * 60)

    print("\n")


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print_model_summary(

        (30, 18)

    )
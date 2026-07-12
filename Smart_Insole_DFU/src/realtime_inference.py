import os
import joblib
import numpy as np

from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(
            os.path.dirname(
            os.path.abspath(__file__)
            )
            )


DATA_DIR=os.path.join(BASE_DIR,'data')
MODEL_DIR=os.path.join(BASE_DIR,'models')


X=np.load(

os.path.join(

DATA_DIR,

'X.npy'

)

)



model=load_model(

os.path.join(

MODEL_DIR,

'dfu_lstm.keras'

)

)



sample=X[10]


sample=sample.reshape(

1,

30,

19

)



pred=model.predict(

sample

)



print()

print(

"Predicted Risk Score"

)

print(

pred[0][0]

)
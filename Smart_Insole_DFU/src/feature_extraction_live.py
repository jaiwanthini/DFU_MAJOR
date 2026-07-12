import numpy as np


def extract_features(window):

    features=[]


    for i in range(4):

        features.append(

            np.sum(window[:,i])*0.1

        )


    total_pti=sum(features)


    df=max(features)/total_pti


    temp_diff=abs(

        np.mean(window[:,5])

        -

        np.mean(window[:,4])

    )


    tf=1+temp_diff/10


    spo2=np.mean(

        window[:,6]

    )


    sf=1+(100-spo2)/50



    epti=(

        total_pti/2

    )*(

        1+0.5*df

    )*tf*sf



    return [

        total_pti,

        df,

        tf,

        sf,

        epti

    ]
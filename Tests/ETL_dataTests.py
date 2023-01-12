import pandas as pd
from typing import List
from collections import Counter
import numpy as np
from TrainRideNode import TrainRideNode

from ETL_data import keepWeekendDays, keepWorkDays
def createTestSample():
    d = {'basic|uitvoer': ["1-3-2019 05:26:01", "1-3-2019 05:27:00", "1-3-2019 05:36:00","1-3-2019  05:37:00","1-3-2019  05:37:00","1-3-2019  05:37:00",
                  "1-3-2019  05:37:00","1-3-2019  05:37:00"],
         'delay': [0,1,2,1,2,0,5,2],
         'basic|drp_act': ["K_A", "K_V", "D","D","D","D","D","V"]}
    df = pd.DataFrame(data=d)

    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%m-%d-%Y %H:%M:%S')
    df['date'] = pd.to_datetime(df['basic|uitvoer']).dt.date
    return df

def keepActivity(df_input, act_val : List[str]):
    df_input = df_input.loc[df_input['basic|drp_act'].isin(act_val)]
    return df_input

def strToDT():
    df = createTestSample()
    #df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%d-%m-&Y %H:%M:%S.%f')
    return df

print(strToDT())
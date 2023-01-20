import pandas as pd
from typing import List
import datetime
from collections import Counter
import numpy as np
from TrainRideNode import TrainRideNode

from ETL_data_stations import keepWeekendDays, keepWorkDays
def createTestSample():
    d = {'basic|uitvoer': ["1-3-2019 05:26:01", "1-3-2019 05:27:00", "1-3-2019 05:36:00","1-3-2019  05:37:00"],
         'delay': [0,1,2,1],
         'basic|drp_act': ["V", "D", "A","D"],
         'basic_treinnr_treinserie': ["500E","500E","500E","500E"],
         'basic|treinnr': ["501","501","501","501"]}
    df = pd.DataFrame(data=d)

    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%m-%d-%Y %H:%M:%S')
    df['basic|plan'] = df['basic|uitvoer']
    df['date'] = pd.to_datetime(df['basic|uitvoer']).dt.date
    return df

def keepActivity(df_input, act_val : List[str]):
    df_input = df_input.loc[df_input['basic|drp_act'].isin(act_val)]
    return df_input

def strToDT():
    df = createTestSample()
    #df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%d-%m-&Y %H:%M:%S.%f')
    return df


def changeD(df_complete):
    df_filter = df_complete[(df_complete["basic|drp_act"] == 'D')]
    df_K_A = df_filter
    df_K_V = df_filter.copy()

    df_K_V["basic|drp_act"] = df_K_V["basic|drp_act"].replace('D', 'K_V')
    df_K_A["basic|drp_act"] = df_K_A["basic|drp_act"].replace('D','K_A')

    df_res = df_complete[~(df_complete["basic|drp_act"] == 'D')]
    df_res = df_res.append(df_K_A, ignore_index=True)
    df_res = df_res.append(df_K_V, ignore_index=True)

    df_res = df_res.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])

    df_res = df_res.reset_index(drop=True)

    return df_res



pd.set_option('display.max_columns', None)
print(changeD(createTestSample()))

#df1 = changeD(createTestSample())
# print(df1)

d = {'basic|uitvoer': ["1-3-2019", "1-3-2019", "1-3-2019","1-3-2019"],
         'delay': [0,1,2,1],
         'basic|drp_act': ["V", "D", "A","D"],
         'basic_treinnr_treinserie': ["500E","500E","500E","500E"],
         'basic|treinnr': ["501","501","501","501"]}
df = pd.DataFrame(data=d)
df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%m-%d-%Y')
print(df.dtypes)


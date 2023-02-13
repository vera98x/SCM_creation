import pandas as pd
from typing import List
from Load_transform_df import keepTrainseries, keepWorkDays, changeToD
import math


def createTestSample():
    d = {'basic|uitvoer': ["1-3-2019 05:26:01", "1-3-2019 05:27:00", "1-3-2019 05:36:00","2-3-2019  05:26:01"],
         'delay': [0,1,2,1],
         'basic|drp_act': ["V", "D", "A","D"],
         'basic|drp': ["Bkl", "Ma", "Utzl","Bkl"],
         'basic_treinnr_treinserie': ["600E","500E","500E","600E"],
         'basic|treinnr': ["501","501","501","501"]}
    df = pd.DataFrame(data=d)

    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%m-%d-%Y %H:%M:%S')
    df['basic|plan'] = df['basic|uitvoer']
    df['date'] = pd.to_datetime(df['basic|uitvoer']).dt.date
    return df

def createTestSample_minimal():
    d = {'day' : [1,1,2,2,3,3,4,5,5],
        'basic|treinnr': ["501","502","501","502","501","502","502", "501","502"],
         'drp' : ["dr1","dr2","dr1","dr2","dr1","dr2","dr2", "dr1","dr3"],
         'act': ["a", "k_v", "a", "v", "a", "v", "v", "a", "v"]
         }
    df = pd.DataFrame(data=d)
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

def getRealData():
    export_name = '../Data/2019-03-01_2019-05-31.csv'
    list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O', '8100E',
                           '8100O', '9000E', '9000O', '12600E',
                           '76200O', '78100E', '78100O', '79000E', '79000O',
                           #'80000E', '80000O', '80200E', '80200O', '89200E', '89200O', '93200E', '93200O'
                           ]

    # split dataframes in column
    df = pd.read_csv(export_name, sep=";")
    df = df[
        ['basic_treinnr_treinserie', 'basic|treinnr', 'basic|spoor', 'basic|drp', 'basic|drp_act', 'basic|plan',
         'basic|uitvoer']]

    # set types of columns
    df['basic_treinnr_treinserie'] = df['basic_treinnr_treinserie'].astype('string')
    df['basic|drp'] = df['basic|drp'].astype('string')
    # df['basic|spoor'] = df['basic|spoor'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%d-%m-%Y %H:%M')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%d-%m-%Y %H:%M')

    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    df['date'] = pd.to_datetime(df['basic|plan']).dt.date

    df = keepTrainseries(df, list_of_trainseries)

    #df = keepTrainseries(df, list_of_trainseries)
    df = changeToD(df)
    # group per trainserie and make uniform
    # TODO: changed this
    # gb = df.groupby(['basic_treinnr_treinserie'])
    # trainserieList = [gb.get_group(x) for x in gb.groups]
    # for trainserie_index in range(len(trainserieList)):
    # trainserie = trainserieList[trainserie_index]
    # trainserieList[trainserie_index] = makeDataUniform(trainserie)

    df = df[~df['date'].isnull()]
    df = keepWorkDays(df)

    return df

def findSched(df):
    #print(df)
    # TODO: only suitable for days with D
    # get the amount of days
    days_count = len(df.groupby('date'))
    # set treshold for amount of occurences
    min_occ = math.ceil(days_count*0.5)
    g = df.groupby(['basic|treinnr', 'basic|drp', 'basic|drp_act'])

    df = g.filter(lambda x: len(x) >= min_occ).reset_index(drop=True)
    #print("Removed variables: ", len(g.filter(lambda x: len(x) < min_occ).reset_index(drop=True)))

    # now we have all items that have a higher occurrence than the threshold from all days
    # since we want to have a schedule for one day, we keep all occurences once.
    df = df.drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'basic|drp_act'], keep='first').reset_index(drop=True)
    print("after: ", len(df))
    # since the trainnumber can have multiple actions at a station, we check if there are no duplicate actions
    print("duplicated actions", len(df[df.duplicated(['basic|treinnr', 'basic|drp'], keep=False)]))
    timestamp = pd.to_datetime("01-01-2000", format='%d-%m-%Y')
    df = df.assign(date=timestamp)
    df["delay"] = 0
    df= df.sort_values(by=['basic_treinnr_treinserie', "basic|treinnr", "basic|uitvoer"]).reset_index(drop=True)
    return df

findSched(getRealData())


import copy

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import math

def keepTrainseries(df_input : pd.DataFrame, act_val : List[str]) ->  pd.DataFrame:
    df_input = df_input[df_input['basic_treinnr_treinserie'].isin(act_val)]

    return df_input
def keepWorkDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['date'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] <= 5)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input
def keepWeekendDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['date'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] >= 6)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input

def changeToD(df_complete : pd.DataFrame) ->  pd.DataFrame:
    df_filter = df_complete[(df_complete["basic|drp_act"] == 'K_A') | (df_complete["basic|drp_act"] == 'A')
                            | (df_complete["basic|drp_act"] == 'K_V') | (df_complete["basic|drp_act"] == 'V')]

    # find the unique values of A or V: then Rs and Vs should be kept
    unique = df_filter.drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'date'], keep=False)

    # change the activity of "vertrek" to "doorkomst"
    df_complete["basic|drp_act"] = df_complete["basic|drp_act"].replace('K_V', 'D')
    df_complete["basic|drp_act"] = df_complete["basic|drp_act"].replace('V', 'D')

    #remove the A or K_A values
    df_complete = df_complete[~(df_complete["basic|drp_act"] == 'K_A') & ~(df_complete["basic|drp_act"] == 'A')]

    df_res = pd.concat([unique, df_complete]).drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'date' ], keep='first')

    df_res = df_res.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])

    df_res = df_res.reset_index(drop=True)
    return df_res

def makeDataUniform(df : pd.DataFrame, sched : pd.DataFrame) ->  pd.DataFrame:
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    # get first dataframe as example to compare from
    sched_date = sched.iloc[0]['date']

    df_new = pd.DataFrame(columns=df.columns)

    print("amount of days", len(grouped_by_date))
    # loop through every other frame, compare the columns
    for day_index in range(len(grouped_by_date)):
        day = grouped_by_date[day_index]
        day_date = day.iloc[0]['date']

        diff = pd.concat([sched, day]).drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'basic|drp_act', "global_plan"],
                                                         keep=False)
        # if a dataframe differs, print the data of the frame and show difference
        if (len(diff) != 0):
            #TODO: if length of dataframe is too small remove it anyway?

            # remove the extra activities
            remove_extra_activities = diff[~(diff["date"] == sched_date)]
            df_res = pd.concat([day, remove_extra_activities]).drop_duplicates(
                subset=['basic|treinnr', 'basic|drp', 'date'],
                keep=False)

            # add missing values
            add_extra_activities = diff[(diff["date"] == sched_date)]
            add_extra_activities['date'] = day_date
            add_extra_activities["sorting_time"] = add_extra_activities['basic|plan']
            # overlap the delays (if there are too many np.nan, the mv_fischer cannot handle it)
            add_extra_activities['basic|uitvoer'] = np.nan
            add_extra_activities['basic|plan'] = np.nan

            # TODO: when creating the dataset, remove the basic plan and basic uitvoer
            df_res['sorting_time'] = df_res['basic|plan']
            df_res = pd.concat([df_res, add_extra_activities])
            df_res = df_res.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "sorting_time"])
            # remove the sorting column
            df_res = df_res.drop(columns=['sorting_time'])
            df_res = df_res.reset_index(drop=True)

            grouped_by_date[day_index] = df_res

            df_new = pd.concat(grouped_by_date)

    return df_new

def toSeconds(x : datetime.time):
    try:
        return x.total_seconds()
    except:
        return np.nan

def addbufferColumn(df):
    df['buffer'] = (df['basic|plan'] - df['basic|plan'].shift(1)).map(lambda x: x.total_seconds())
    df.loc[(df['basic|drp_act'] != 'V') & (df['basic|drp_act'] != 'K_V'), 'buffer'] = 0
    df.loc[(df['basic|treinnr'] != df['basic|treinnr'].shift(1)) , 'buffer'] = 0
    df['buffer'] = df.buffer.fillna(0)
    return df

def retrieveDataframe(export_name : str, workdays : bool, list_of_trainseries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # split dataframes in column
    df = pd.read_csv(export_name, sep=";")
    df = df[
        ['basic_treinnr_treinserie','basic|treinnr', 'basic|spoor', 'basic|drp', 'basic|drp_act', 'basic|plan', 'basic|uitvoer']]

    # set types of columns
    df['basic_treinnr_treinserie'] = df['basic_treinnr_treinserie'].astype('string')
    df['basic|drp'] = df['basic|drp'].astype('string')
    #df['basic|spoor'] = df['basic|spoor'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%Y-%m-%d %H:%M:%S')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%Y-%m-%d %H:%M:%S')
    df["global_plan"] = df['basic|plan'].dt.floor('Min')
    df["global_plan"] = df["global_plan"].dt.time

    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    df['date'] = pd.to_datetime(df['basic|plan']).dt.date


    df = keepTrainseries(df, list_of_trainseries)
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    df = addbufferColumn(df)
    df = changeToD(df)

    df = df[~df['date'].isnull()]
    if workdays == None:
        pass
    elif workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)

    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    df = df.reset_index(drop=True)

    sched = findSched(df)

    df = makeDataUniform(df, sched)

    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    df = df.reset_index(drop=True)


    df['plan|time'] = pd.to_datetime(df['basic|plan']).dt.time
    df['uitvoer|time'] = pd.to_datetime(df['basic|uitvoer']).dt.time

    df['delay'] = df['basic|uitvoer'] - df['basic|plan']
    df['delay'] = df['delay'].map(toSeconds)

    return df[['date', 'basic_treinnr_treinserie','basic|treinnr', 'basic|spoor', 'basic|drp', 'basic|drp_act', "global_plan", 'delay', "buffer"]], sched

def findSched(df):
    df_sched = copy.deepcopy(df)

    # TODO: only suitable for days with D
    # get the amount of days
    days_count = len(df_sched.groupby('date'))
    print(days_count)
    # set treshold for amount of occurences
    min_occ = math.ceil(days_count*0.5)
    g = df_sched.groupby(['basic|treinnr', 'basic|drp', 'basic|drp_act', "global_plan"])
    df_sched = g.filter(lambda x: len(x) >= min_occ).reset_index(drop=True)
    #print("Removed variables: ", len(g.filter(lambda x: len(x) < min_occ).reset_index(drop=True)))

    # now we have all items that have a higher occurrence than the threshold from all days
    # since we want to have a schedule for one day, we keep all occurences once.
    df_sched = df_sched.drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'basic|drp_act'], keep='first').reset_index(drop=True)
    print("events per day: ", len(df_sched))
    # since the trainnumber can have multiple actions at a station, we check if there are no duplicate actions
    print("duplicated actions", len(df_sched[df_sched.duplicated(['basic|treinnr', 'basic|drp'], keep=False)]))
    timestamp = pd.to_datetime("01-01-2000", format='%d-%m-%Y')
    df_sched = df_sched.assign(date=timestamp)
    df_sched["delay"] = 0
    df_sched= df_sched.sort_values(by=['basic_treinnr_treinserie', "basic|treinnr", "basic|uitvoer"]).reset_index(drop=True)
    df_sched['plan|time'] = pd.to_datetime(df_sched['basic|plan']).dt.time

    print(df_sched)

    return df_sched

import copy

import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
from typing import List, Tuple
import math

def keepTrainseries(df_input : pd.DataFrame, act_val : List[str]) ->  pd.DataFrame:
    df_input = df_input[df_input['basic_treinnr_treinserie'].isin(act_val)]

    return df_input
def keepWorkDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    # translate date to number 1 - 7 and is ma -su
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    # weekdays are between 1-5
    df_input = df_input[(df_input['daynumber'] <= 5)]
    # drop the column again
    df_input = df_input.drop(columns=['daynumber'])
    return df_input
def keepWeekendDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    # translate date to number 1 - 7 and is ma -su
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    # weekdays are between 1-5
    df_input = df_input[(df_input['daynumber'] >= 6)]
    # drop the column again
    df_input = df_input.drop(columns=['daynumber'])
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
            # find activities that are not in the schedule
            remove_extra_activities = diff[~(diff["date"] == sched_date)]
            # add those activities again to the day dataframe, and drop the duplicates
            df_res = pd.concat([day, remove_extra_activities]).drop_duplicates(
                subset=['basic|treinnr', 'basic|drp', 'basic|drp_act', 'date'],
                keep=False)

            # add missing values
            add_extra_activities = diff[(diff["date"] == sched_date)]
            add_extra_activities = add_extra_activities.assign(date=day_date)
            add_extra_activities = add_extra_activities.assign(time = add_extra_activities["basic|plan"].dt.time)
            add_extra_activities.loc[:, "basic|plan"] = pd.to_datetime(add_extra_activities.date.astype(str) + ' ' + add_extra_activities.time.astype(str))
            df["basic|plan"] = df["basic|plan"].apply(lambda x: x + dt.timedelta(days=1) if x.hour <= 4 else x)
            add_extra_activities = add_extra_activities.drop(columns=["time"])

            # overlap the delays (if there are too many np.nan, the mv_fischer cannot handle it)
            add_extra_activities['basic|uitvoer'] = np.nan
            add_extra_activities['delay'] = np.nan

            # Combine the datafrem for the day with the extra activities
            df_r = pd.concat([df_res, add_extra_activities])
            # sort them
            df_r = df_r.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", 'basic|plan'])
            df_r = df_r.reset_index(drop=True)
            # add to the group again
            grouped_by_date[day_index] = df_r
    #create new datafame
    df_new = pd.concat(grouped_by_date)

    return df_new

def toSeconds(x : datetime.time):
    try:
        return x.total_seconds()
    except:
        return np.nan

def addbufferColumn(df):
    # sort the dataframe
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    # take the current plan and the plan of the previous event
    df['buffer'] = (df['basic|plan'] - df['basic|plan'].shift(1)).map(lambda x: x.total_seconds())
    # if those events are not at the same station, fill with 0
    df.loc[(df['basic|drp'] != df['basic|drp'].shift(1)) , 'buffer'] = 0
    # if those events are not between the same train, fill with 0
    df.loc[(df['basic|treinnr'] != df['basic|treinnr'].shift(1)) , 'buffer'] = 0
    # if those events are not at the same date, fill with 0
    df.loc[(df['date'] != df['date'].shift(1)), 'buffer'] = 0
    # for each event for which there was no 'upper' row to compare with, it is na, so also fill that with 0
    df['buffer'] = df.buffer.fillna(0)
    return df

def addTravelTimeColumn(df):
    # Make sure that all activities are occuring once per location (aka make it D)
    # sort the dataframe
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    # take the current plan and the plan of the previous event and substract with buffer
    df['traveltime'] = (df['basic|plan'] - df['basic|plan'].shift(1)).map(lambda x: x.total_seconds())
    df['traveltime'] = df['traveltime'] - df['buffer']
    # if those events are not between the same train, fill with 0
    df.loc[(df['basic|treinnr'] != df['basic|treinnr'].shift(1)) , 'traveltime'] = 0
    # if those events are not at the same date, fill with 0
    df.loc[(df['date'] != df['date'].shift(1)), 'traveltime'] = 0
    # for each event for which there was no 'upper' row to compare with, it is na, so also fill that with 0
    df['traveltime'] = df.traveltime.fillna(0)
    return df

def removeCancelledTrain(df):
    # only keep the non nan values
    df = df[df['vklvos_plan_actueel'].notna()]
    return df


def retrieveDataframe(export_name : str, workdays : bool, list_of_trainseries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # split dataframes in column
    df = pd.read_csv(export_name, sep=";")
    df = df[
        ["nvgb_verkeersdatum", 'basic_treinnr_treinserie','basic|treinnr', 'basic|spoor', 'basic|drp', 'basic|drp_act', 'basic|plan', 'basic|uitvoer', 'vklvos_plan_actueel', 'wissels']]
    # set types of columns
    df['basic_treinnr_treinserie'] = df['basic_treinnr_treinserie'].astype('string')
    df['basic|drp'] = df['basic|drp'].astype('string')
    #df['basic|spoor'] = df['basic|spoor'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')
    df['nvgb_verkeersdatum'] = pd.to_datetime(df['nvgb_verkeersdatum'], format='%Y-%m-%d').dt.date
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%Y-%m-%d %H:%M:%S')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%Y-%m-%d %H:%M:%S')
    # use this column to determine if a train is cancelled
    df['vklvos_plan_actueel'] = pd.to_datetime(df['vklvos_plan_actueel'], format='%Y-%m-%d %H:%M:%S')
    # rond het plan af op hele minuten
    df["global_plan"] = df['basic|plan'].dt.floor('Min')
    df["global_plan"] = df["global_plan"].dt.time
    df['delay'] = df['basic|uitvoer'] - df['basic|plan']
    df['delay'] = df['delay'].map(toSeconds)
    # if the basic|uitvoer is empty, fill it with the value of basic|plan
    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    # rename column
    df['date'] = df['nvgb_verkeersdatum']
    #todo replace this with real wissel data
    #df = df.assign(wissels = "MP$283$R,MP$281B$R,MP$281A$R,MP$271A$L,MP$269$L,MP$263A$R,MP$247$R,MP$243B$R,MP$241A$L")
    df = df.assign(speed = 80)

    # only keep the desired train series
    df = keepTrainseries(df, list_of_trainseries)
    # add a buffer column
    df = addbufferColumn(df)
    # Remove all arrival events, keep the departure events and call it 'D'
    #df = changeToD(df)
    # add a traveltime column (important to have after changeToD())
    df = addTravelTimeColumn(df)
    # only keep the dates that are known
    df = df[~df['date'].isnull()]
    # only keep working days
    if workdays == None:
        pass
    elif workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)

    # find the general schedule of the df
    sched = findSched(df)
    # then remove the cancelled trains (since they do have a planning, delete them after the schedule is found and before to make all data uniform
    df = removeCancelledTrain(df)
    # according to this schedule, impute or remove entries
    df = makeDataUniform(df, sched)

    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    df = df.reset_index(drop=True)

    return df[['date', 'basic_treinnr_treinserie','basic|treinnr', 'basic|spoor', 'basic|drp', 'basic|drp_act', "basic|plan" ,"global_plan", 'delay', "buffer", "traveltime", "wissels", "speed"]], sched

def findSched(df):
    df_sched = copy.deepcopy(df)
    df_sched = df_sched.sort_values(by=['date', 'basic_treinnr_treinserie', "basic|treinnr", "basic|plan"])
    df_sched = df_sched.reset_index(drop=True)
    # TODO: only suitable for days with D
    # get the amount of days
    days_count = len(df_sched.groupby('date'))
    print(days_count)
    # set treshold for amount of occurences
    min_occ = math.ceil(days_count*0.5)
    g = df_sched.groupby(['basic|treinnr', 'basic|drp', 'basic|drp_act', "global_plan"])
    # only keep the events that occurs larger than the threshold
    df_sched = g.filter(lambda x: len(x) >= min_occ).reset_index(drop=True)

    # now we have all items that have a higher occurrence than the threshold from all days
    # since we want to have a schedule for one day, we keep all occurences once (now all different dates are included).
    df_sched = df_sched.drop_duplicates(subset=['basic|treinnr', 'basic|drp', 'basic|drp_act'], keep='first').reset_index(drop=True)
    print("events per day: ", len(df_sched))
    # since the trainnumber can have multiple actions at a station, we check if there are no duplicate actions
    print("duplicated actions", len(df_sched[df_sched.duplicated(['basic|treinnr', 'basic|drp', 'basic|drp_act'], keep=False)]))
    # add a old timestamp to it, to recognise the schedule entries
    timestamp = pd.to_datetime("01-01-2000", format='%d-%m-%Y')

    df_sched = df_sched.assign(date=timestamp)
    # update the basic|plan of the schedule
    df_sched = df_sched.assign(time=df_sched["basic|plan"].dt.time)
    df_sched.loc[:, "basic|plan"] = pd.to_datetime(df_sched.date.astype(str) + ' ' + df_sched.time.astype(str))
    df_sched["basic|plan"] = df_sched["basic|plan"].apply(lambda x: x + dt.timedelta(days=1) if x.hour <= 4 else x)
    df_sched = df_sched.drop(columns=["time"])

    df_sched["delay"] = 0
    df_sched= df_sched.sort_values(by=['basic_treinnr_treinserie', "basic|treinnr", "basic|plan"]).reset_index(drop=True)

    return df_sched

# clean data
import pandas as pd
import numpy as np
from TrainRideNode import TrainRideNode
from OLD.StationNode import StationNode
from typing import List
from Load_transform_df import retrieveDataframe

# The values in the CSV file differ sometimes to here are global vlaues
basic_treinnr = 'basic_treinnr' #'basic|treinnr'
basic_drp = 'basic|drp'
basic_drp_act = "basic|drp_act"
basic_spoor = 'basic|spoor'
basic_plan = 'basic|plan'
basic_uitvoer = 'basic|uitvoer'

def retrieveDataframe(export_name, workdays = None):
    # todo impute values for missing 'spoor' values

    df = pd.read_csv(export_name, sep=";")
    df = df[['nvgb_verkeersdatum','basic|treinnr', 'basic|drp', 'basic|drp_act', 'basic|spoor', 'basic|plan', 'basic|uitvoer', 'basic_treinnr_treinserie']]

    # if the basic uitvoer has not been registered, take the planned time.
    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%d-%m-%Y %H:%M')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%d-%m-%Y %H:%M')
    df['date'] = df['nvgb_verkeersdatum']
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

    df['basic|drp'] = df['basic|drp'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')

    df["basic|drp_act"] = df["basic|drp_act"].replace('V', 'K_V')
    df["basic|drp_act"] = df["basic|drp_act"].replace('A', 'K_A')

    #df['drp_with_act'] = df['basic|treinnr'] + "_" + df['basic|drp'] + '_' + df['basic|drp_act']

    df['plan|time'] = pd.to_datetime(df['basic|plan']).dt.time
    df['uitvoer|time'] = pd.to_datetime(df['basic|uitvoer']).dt.time

    # if there is no date, even after imputing values, remove these rows
    test_df = pd.isna(df)
    df = df.loc[test_df['date'] == False]

    df['delay'] = df['basic|uitvoer'] - df['basic|plan']
    df['delay'] = df['delay'].map(lambda x: x.total_seconds())
    df = changeD(df)
    df = keepActivity(df, ["K_V", "V", "K_A", "A"])
    if workdays == None:
        return df

    if workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)

    return df
def keepActivity(df_input, act_val : List[str]):
    df_input = df_input.loc[df_input['basic|drp_act'].isin(act_val)]
    return df_input
def keepTrainseries(df_input, act_val : List[str]):
    df_input = df_input[df_input['basic_treinnr_treinserie'].isin(act_val)]

    return df_input
def keepWorkDays(df_input):
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['date'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] <= 5)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input
def keepWeekendDays(df_input):
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['date'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] >= 6)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input

def getAllGroupLengths(grouped_by_date):
    total_lengths = [len(x) for x in grouped_by_date]
    d = {}
    for x in total_lengths:
        d[x] = d.get(x, 0) + 1
    return d


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

def dfToTrainRides(df):
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    print(getAllGroupLengths(grouped_by_date))
    total_lengths = [len(x) for x in grouped_by_date]
    most_common_length = max(set(total_lengths), key=total_lengths.count)
    print()
    uniform_data = list(filter(lambda item: len(item) == most_common_length, grouped_by_date))
    data = uniform_data
    data_len = len(data[0])

    dataset_with_classes = np.zeros((0, data_len))

    for day_df in data:
        dataset_day = np.zeros((0,))
        for index, trainride in day_df.iterrows():
            node = TrainRideNode(trainride['basic|treinnr'], trainride['basic|drp'], trainride['basic|spoor'],
                                 trainride['basic|drp_act'], trainride['delay'], trainride['plan|time'])
            dataset_day = np.append(dataset_day, node)
        dataset_with_classes = np.r_[dataset_with_classes, [dataset_day]]
    return dataset_with_classes

def getDataSetWith_TRN(export_name, workdays, list_of_trainseries):
    df = retrieveDataframe('Data/6100_jan_nov_2022.csv')
    print(len(df))
    df = keepTrainseries(df, list_of_trainseries)
    print(len(df))
    if workdays == None:
        pass
    elif workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)
    print(len(df))
    # group on stations
    gb = df.groupby(['basic|drp'])
    print("Amount of Stations:", len(gb.groups))
    grouped_by_station = [gb.get_group(x) for x in gb.groups]
    print(len(grouped_by_station))

    stationsWithData = []

    for df_g in grouped_by_station:
        gb_day = df_g.groupby(['date'])
        print("Amount of days:", len(gb_day.groups))
        grouped_by_station_day = [gb_day.get_group(x) for x in gb_day.groups]
        total_lengths = [len(x) for x in grouped_by_station_day]
        most_common_length = max(set(total_lengths), key=total_lengths.count)
        uniform_data = list(filter(lambda item: len(item) == most_common_length, grouped_by_station_day))
        print(len(uniform_data), len(grouped_by_station_day))
        name = uniform_data[0].iloc[0]['basic|drp']
        data = uniform_data
        stationsWithData.append(StationNode(name, data))

    return stationsWithData



# Create format for data to feed in PC algorithm
def class_dataset_to_delay_columns_pair(dataset_with_classes):
    array_with_delays_2d = np.zeros((0, len(dataset_with_classes[0])))
    column_names = np.array(list(map(lambda x: x.getID(), dataset_with_classes[0])))
    for day in dataset_with_classes:
        delays = np.array(list(map(lambda x: x.getDelay(), day)))

        array_with_delays_2d = np.r_[array_with_delays_2d, [delays]]
    return (array_with_delays_2d, column_names)
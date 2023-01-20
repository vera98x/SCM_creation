# clean data
import pandas as pd
import numpy as np
from TrainRideNode import TrainRideNode
from StationNode import StationNode
from typing import List
from test6100 import retrieveDataframeNew

def keepTrainseries(df_input, act_val : List[str]):
    # print(df_input['basic_treinnr_treinserie'])
    # print(type(df_input['basic_treinnr_treinserie'].iloc[0]))
    # print(act_val)
    # print(df_input['basic_treinnr_treinserie'].isin(act_val))
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
    df = retrieveDataframeNew(export_name)
    print(len(df))
    df = keepTrainseries(df, list_of_trainseries)
    print(len(df))
    if workdays == None:
        pass
    elif workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)

    # group on stations
    dataset_with_classes = dfToTrainRides(df)

    return dataset_with_classes


# Create format for data to feed in PC algorithm
def class_dataset_to_delay_columns_pair(dataset_with_classes):
    array_with_delays_2d = np.zeros((0, len(dataset_with_classes[0])))
    column_names = np.array(list(map(lambda x: x.getID(), dataset_with_classes[0])))
    for day in dataset_with_classes:
        delays = np.array(list(map(lambda x: x.getDelay(), day)))

        array_with_delays_2d = np.r_[array_with_delays_2d, [delays]]
    return (array_with_delays_2d, column_names)
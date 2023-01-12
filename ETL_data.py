# clean data
import pandas as pd
import numpy as np
from TrainRideNode import TrainRideNode
from typing import List

# The values in the CSV file differ sometimes to here are global vlaues
basic_treinnr = 'basic_treinnr' #'basic|treinnr'
basic_drp = 'basic|drp'
basic_drp_act = "basic|drp_act"
basic_spoor = 'basic|spoor'
basic_plan = 'basic|plan'
basic_uitvoer = 'basic|uitvoer'

def retrieveDataframe(export_name, workdays = True):
    # todo impute values for missing 'spoor' values

    df = pd.read_csv(export_name, sep=";")
    df = df[['basic|treinnr', 'basic|drp', 'basic|drp_act', 'basic|spoor', 'basic|plan', 'basic|uitvoer']]

    # if the basic uitvoer has not been registered, take the planned time.
    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    #df['basic|plan'] = pd.to_datetime(df['basic|plan'])
    #df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'])
    #TODO: not in every file the format is the same
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%Y%m%d %H:%M:%S.%f')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%Y%m%d %H:%M:%S.%f')

    #df['basic|treinnr'] = df['basic|treinnr'].astype(int)
    df['basic|drp'] = df['basic|drp'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')
    df['drp_with_act'] = df['basic|treinnr'] + "_" + df['basic|drp'] + '_' + df['basic|drp_act']
    df['date'] = pd.to_datetime(df['basic|uitvoer']).dt.date
    df['plan|time'] = pd.to_datetime(df['basic|plan']).dt.time
    df['uitvoer|time'] = pd.to_datetime(df['basic|uitvoer']).dt.time

    df['delay'] = df['basic|uitvoer'] - df['basic|plan']
    df['delay'] = df['delay'].map(lambda x: x.total_seconds())
    if workdays:
        df = keepWorkDays(df)
    else:
        df = keepWeekendDays(df)
    df = keepActivity(df, ["K_V", "V", "K_A", "A"])

    return df
def keepActivity(df_input, act_val : List[str]):
    df_input = df_input.loc[df_input['basic|drp_act'].isin(act_val)]
    return df_input
def keepWorkDays(df_input):
    df_input['daynumber'] = df_input['basic|uitvoer'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['basic|uitvoer'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] <= 5)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input
def keepWeekendDays(df_input):
    df_input['daynumber'] = df_input['basic|uitvoer'].apply(lambda x: int(x.strftime("%u")))
    df_input['dayname'] = df_input['basic|uitvoer'].apply(lambda x: x.strftime("%A"))
    df_input = df_input[(df_input['daynumber'] >= 6)]
    df_input = df_input.drop(columns=['daynumber', 'dayname'])
    return df_input

def getAllGroupLengths(grouped_by_date):
    total_lengths = [len(x) for x in grouped_by_date]
    d = {}
    for x in total_lengths:
        d[x] = d.get(x, 0) + 1
    return d

def getDataSetWith_TRN(export_name):
    df = retrieveDataframe(export_name, True)
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


"""### Create format for data to feed in PC algorithm"""


# Create format for data to feed in PC algorithm
def class_dataset_to_delay_columns_pair(dataset_with_classes):
    array_with_delays_2d = np.zeros((0, len(dataset_with_classes[0])))
    column_names = np.array(list(map(lambda x: x.getID(), dataset_with_classes[0])))
    for day in dataset_with_classes:
        delays = np.array(list(map(lambda x: x.getDelay(), day)))

        array_with_delays_2d = np.r_[array_with_delays_2d, [delays]]
    return (array_with_delays_2d, column_names)
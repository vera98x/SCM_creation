# clean data
import pandas as pd
import numpy as np
from TrainRideNode import TrainRideNode

def retrieveDataframe(export_name):
    # todo impute values for missing 'spoor' values

    df = pd.read_csv(export_name, sep=";")
    df = df[['basic|treinnr', 'basic|drp', 'basic|drp_act', 'basic|spoor', 'basic|plan', 'basic|uitvoer']]

    # if the basic uitvoer has not been registered, take the planned time.
    df['basic|uitvoer'] = df['basic|uitvoer'].fillna(df['basic|plan'])
    df['basic|plan'] = pd.to_datetime(df['basic|plan'], format='%Y%m%d %H:%M:%S.%f')
    df['basic|uitvoer'] = pd.to_datetime(df['basic|uitvoer'], format='%Y%m%d %H:%M:%S.%f')

    df['basic|treinnr'] = df['basic|treinnr'].astype(int)
    df['basic|drp'] = df['basic|drp'].astype('string')
    df['basic|treinnr'] = df['basic|treinnr'].astype('string')
    df['basic|drp_act'] = df['basic|drp_act'].astype('string')
    df['drp_with_act'] = df['basic|treinnr'] + "_" + df['basic|drp'] + '_' + df['basic|drp_act']
    df['date'] = pd.to_datetime(df['basic|uitvoer']).dt.date
    df['plan|time'] = pd.to_datetime(df['basic|plan']).dt.time
    df['uitvoer|time'] = pd.to_datetime(df['basic|uitvoer']).dt.time

    df['delay'] = df['basic|uitvoer'] - df['basic|plan']
    df['delay'] = df['delay'].map(lambda x: x.total_seconds())

    return df


def getDataSetWith_TRN(export_name):
    df = retrieveDataframe(export_name)
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]

    total_lengths = [len(x) for x in grouped_by_date]
    most_common_length = max(set(total_lengths), key=total_lengths.count)
    print()
    uniform_data = list(filter(lambda item: len(item) == most_common_length, grouped_by_date))

    data = uniform_data
    dataLen = len(data[0])
    sampleSize = len(data)

    dataset_with_classes = np.zeros((0, dataLen))

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
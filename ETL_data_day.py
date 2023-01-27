# clean data
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from TrainRideNode import TrainRideNode


def getAllGroupLengths(grouped_by_date : List[pd.DataFrame]) -> Dict[int, int]:
    total_lengths = [len(x) for x in grouped_by_date]
    d = {}
    for x in total_lengths:
        d[x] = d.get(x, 0) + 1
    return d

def dfToTrainRides(df : pd.DataFrame) -> np.array:
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



# Create format for data to feed in PC algorithm
def TRN_matrix_to_delay_matrix_columns_pair(dataset_with_classes : np.array) -> Dict[str, Union[np.array, str]]:
    array_with_delays_2d = np.zeros((0, len(dataset_with_classes[0]))).astype(int)
    column_names = np.array(list(map(lambda x: x.getID(), dataset_with_classes[0])))
    for day in dataset_with_classes:
        delays = np.array(list(map(lambda x: int(x.getDelay()), day)))
        array_with_delays_2d = np.r_[array_with_delays_2d, [delays]]
    resDict = {}
    resDict['delay_matrix'] = array_with_delays_2d
    resDict['column_names'] = column_names
    return resDict
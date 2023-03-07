# clean data
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from TrainRideNode import TrainRideNode

def dfToTrainRides(df : pd.DataFrame) -> np.array:
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]

    data = grouped_by_date
    data_len = len(data[0])

    dataset_with_classes_new = np.empty((len(data), data_len)).astype(TrainRideNode)

    for index_x, day_df in enumerate(data):
        for index_y, trainride in day_df.iterrows():
            i_y = index_y % data_len
            node = TrainRideNode(trainride['basic_treinnr_treinserie'], trainride['basic|treinnr'], trainride['basic|drp'], trainride['basic|spoor'],
                                 trainride['basic|drp_act'], trainride['delay'], trainride["global_plan"], trainride["buffer"], trainride["date"])
            dataset_with_classes_new[index_x, i_y] = node
    return dataset_with_classes_new



# Create format for data to feed in PC algorithm
def TRN_matrix_to_delay_matrix(dataset_with_classes : np.array) -> np.array:
    array_with_delays_2d_new = np.zeros((len(dataset_with_classes), len(dataset_with_classes[0]))).astype(float)

    for index, day in enumerate(dataset_with_classes):
        array_with_delays_2d_new[index] = np.array(list(map(lambda x: x.getDelay(), day)))

    return array_with_delays_2d_new
from TrainRideNode import TrainRideNode
import numpy as np

class StationNode:
    def __init__(self, stationName, dataDays):
        self.stationName = stationName
        self.dataDays = dataDays
        self.dataset_with_classes = self.dataToTrainRide()

    def dataToTrainRide(self):
        data_len = len(self.dataDays[0])
        dataset_with_classes = np.zeros((0, data_len))
        for day_df in self.dataDays:
            dataset_day = np.zeros((0,))
            for index, trainride in day_df.iterrows():
                node = TrainRideNode(trainride['basic|treinnr'], trainride['basic|drp'], trainride['basic|spoor'],
                                     trainride['basic|drp_act'], trainride['delay'], trainride['plan|time'])
                dataset_day = np.append(dataset_day, node)
            dataset_with_classes = np.r_[dataset_with_classes, [dataset_day]]
        return dataset_with_classes
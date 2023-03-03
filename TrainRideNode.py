import datetime
import numpy as np
class TrainRideNode:
    def __init__(self, trainSerie : str, trainRideNumber: int, stationName: str, platformNumber: int, activity: str, delay: int,
                 plannedTime : datetime.time, buffer : int):
        self.trainSerie = trainSerie
        self.trainRideNumber = trainRideNumber
        self.stationName = stationName
        self.platformNumber = platformNumber
        self.activity = activity
        self.delay = delay
        self.plannedTime = plannedTime
        self.buffer = buffer

    def __str__(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.platformNumber}_{self.activity}_{self.delay}'

    def __repr__(self):
        return self.__str__()

    def isSameLocation(self, trn) -> bool:
        return (self.stationName == trn.stationName and self.platformNumber == trn.platformNumber)

    def getDelay(self) -> int:
        try:
            return int(self.delay)
        except:
            return np.nan

    def getTrainRideNumber(self):
        return self.trainRideNumber

    def getTrainSerie(self):
        return self.trainSerie

    def getStation(self):
        return self.stationName

    def getPlannedTimeStr(self):
        return str(self.plannedTime.hour) + ":" + str(self.plannedTime.minute) + ":" + str(self.plannedTime.second)

    def getPlannedTime(self):
        return self.plannedTime

    def getID(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.activity}_{self.getPlannedTimeStr()}'

    def getSmallerID(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.activity}'

    def getPlatform(self):
        return self.platformNumber

    def getBuffer(self):
        return self.buffer
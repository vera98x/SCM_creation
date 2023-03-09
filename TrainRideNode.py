import datetime
import numpy as np
class TrainRideNode:
    def __init__(self, trainSerie : str, trainRideNumber: int, stationName: str, platformNumber: int, activity: str, delay: float, globalTime : datetime.time,
                 plannedTime : datetime, buffer : int, date : datetime.date, traveltime :float, wissels : str, speed: int):
        self.trainSerie = trainSerie
        self.trainRideNumber = trainRideNumber
        self.stationName = stationName
        self.platformNumber = platformNumber
        self.activity = activity
        self.delay = delay
        self.globalTime = globalTime
        self.plannedTime = plannedTime
        self.buffer = buffer
        self.date = date
        self.traveltime = traveltime
        self.wissels = wissels.split("$")[1::2]
        self.speed = speed

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

    def getGlobalTimeStr(self):
        return str(self.globalTime.hour) + ":" + str(self.globalTime.minute) + ":" + str(self.globalTime.second)

    def getPlannedTime_time(self):
        return self.plannedTime.time()

    def getPlannedTime(self):
        return self.plannedTime

    def getID(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.activity}_{self.getGlobalTimeStr()}'

    def getSmallerID(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.activity}'

    def getPlatform(self):
        return self.platformNumber

    def getBuffer(self):
        return self.buffer

    def getDate(self):
        return self.date

    def getTraveltime(self):
        return self.traveltime
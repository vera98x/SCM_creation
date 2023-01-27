
class TrainRideNode:
    def __init__(self, trainRideNumber: int, stationName: str, platformNumber: int, activity: str, delay: int,
                 plannedTime):
        self.trainRideNumber = trainRideNumber
        self.stationName = stationName
        self.platformNumber = platformNumber
        self.activity = activity
        self.delay = delay
        self.plannedTime = plannedTime

    def __str__(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.platformNumber}_{self.activity}_{self.delay}'

    def __repr__(self):
        return self.__str__()

    def isSameLocation(self, trn) -> bool:
        return (self.stationName == trn.stationName and self.platformNumber == trn.platformNumber)

    def getDelay(self) -> int:
        return self.delay

    def getTrainRideNumber(self):
        return self.trainRideNumber

    def getStation(self):
        return self.stationName

    def getPlannedTimeStr(self):
        return str(self.plannedTime.hour) + ":" + str(self.plannedTime.minute) + ":" + str(self.plannedTime.second)

    def getPlannedTime(self):
        return self.plannedTime

    def getID(self):
        return f'{self.trainRideNumber}_{self.stationName}_{self.activity}_{self.getPlannedTimeStr()}'

    def getPlatform(self):
        return self.platformNumber
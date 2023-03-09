import copy
from datetime import date, datetime, time
from causallearn.graph.GeneralGraph import GeneralGraph
from FAS import FAS_method
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class Train_type(Enum):
    SPR = 1
    IC = 2

class Station_type(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

class Weekday(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5

class Peak(Enum):
    NONPEAK = 0
    PEAK = 1

class NN_input_sample:
    def __init__(self, trn, row_info):
        self.label = trn.getSmallerID()
        self.date = row_info[0]
        self.trn_delay = row_info[1]
        self.dep_delay = row_info[2:] # consists of 3 items
        # Todo: read this directly from data
        self.type_train = Train_type.SPR if(trn.getTrainSerie() == "8100E" or trn.getTrainSerie() == "8100O") else Train_type.IC
        self.type_station = Station_type.MEDIUM if trn.getStation() == "Mp" else Station_type.SMALL
        self.day_of_the_week = self.getDay(self.date)
        self.peakhour = self.getPeak(trn.getPlannedTime_time())
        self.buffer = trn.getBuffer()
        self.traveltime = trn.getTraveltime()
        list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O',
                               '8100E', '8100O', '9000E', '9000O', '12600E',
                               # '32200E''32200O','32300E','32300O',
                               '76200O', '78100E', '78100O', '79000E', '79000O'
                               # ,'80000E','80000O','80200E','80200O','89200E','89200O','93200E','93200O'
                               # , '301800O','332200E','406200O'
                               ]
        self.trainserie = list_of_trainseries.index(trn.getTrainSerie())

    def getDay(self, trn_date : datetime.date) -> Weekday:
        day = trn_date.weekday()
        if day == 0:
            return Weekday.MONDAY
        if day == 1:
            return Weekday.TUESDAY
        if day == 2:
            return Weekday.WEDNESDAY
        if day == 3:
            return Weekday.THURSDAY
        else:
            return Weekday.FRIDAY
    def getPeak(self, t : datetime.time) -> Peak:
        # note, this program only uses weekdays. In weekends it is always non-peak
        interval_peak = [(time(6,30,0), time(9,0,0)), (time(16,00,0), time(18,00,0))]
        for peak_int in interval_peak:
            if t >= peak_int[0] and t < peak_int[1]:
                return Peak.PEAK
        return Peak.NONPEAK

class Dependency_node:
    def __init__(self, trn):
        self.trn = trn
        self.prev_event = None
        self.prev_platform = None
        self.dep = [] # dependencies

    def get_causal_trains(self):
        l = []
        l.append(self.trn)
        if self.prev_event != None:
            l.append(self.prev_event)
        if self.prev_platform != None:
            l.append(self.prev_platform)
        if len(self.dep) > 0:
            l += self.dep
        return l

    def get_causal_trains_columns(self):
        dep = (self.dep + [None,None,None])[:3] # get the list of dependencies and add if necessary 3 None
        l = [self.trn, self.prev_event, self.prev_platform] + dep
        columns = [str(i) + "_dep_" + str(i) if x == None else x.getSmallerID() for i, x in enumerate(l)]
        missing = [x for x in columns if '_dep_' in x]
        return columns, missing


class NN_samples:
    def __init__(self, gg: GeneralGraph, fas_method: FAS_method, df):
        self.schedule_dependencies = []
        self.gg = gg
        self.id_trn = fas_method.id_trn_dict
        self.df = df
        self.list_nn_input = []

    def gg_to_nn_input(self, events : List[Tuple[str, str]]) -> None:
        '''
        updates the sample_data list with Dependency_nodes
        :param events:
        :return None:
        '''
        # get all nodes from the graph
        nodes = self.gg.get_nodes()
        # fidn for each node its dependencies
        for node in nodes:
            # retrieve the corresponding train, This is the train from the schedule
            trn = self.id_trn[node.get_name()]
            # initialize a dependency node
            node_trn = Dependency_node(trn)
            # get the parents of the node
            parents = self.gg.get_parents(node)
            print("-------------------------------")
            print(trn.getID())
            # only use the stations we know all dependencies from, else the prediction is not possible
            if((trn.getStation(), trn.getTrainSerie()) not in events):
                continue
            print("continue with", (trn.getStation(), trn.getTrainSerie()))
            for parent in parents:
                # the the parent trn
                p_trn = self.id_trn[parent.get_name()]
                # if the parent is in front of the next train at the same platform
                if (trn.getPlatform() == p_trn.getPlatform() and trn.getStation() == p_trn.getStation()):
                    # if this variable is already filled
                    if(node_trn.prev_platform != None):
                        # check which train was really the last train at that point, the other train is then a "general" dependency
                        if(node_trn.prev_platform.getPlannedTime() < p_trn.getPlannedTime()):
                            node_trn.dep.append(node_trn.prev_platform)
                            node_trn.prev_platform = p_trn
                            print("platform Update", p_trn.getID())
                        else:
                            node_trn.dep.append(p_trn)
                            print("Platform, but not last", p_trn.getID())
                    # if prev_platform is not filled, fill it
                    else:
                        node_trn.prev_platform = p_trn
                        print("platform", p_trn.getID())
                # if parent and trn have the same train number, assign it to the prev event
                elif (trn.getTrainRideNumber() == p_trn.getTrainRideNumber()):
                    node_trn.prev_event = p_trn
                    print("prev", p_trn.getID())
                # the rest are general dependencies
                else:
                    node_trn.dep.append(p_trn)
                    print("other parent", p_trn.getID())
            print("-------------------------------")
            dep_list = node_trn.dep
            sorted_dep_list = sorted(dep_list, key=lambda x: x.getPlannedTime(), reverse=False)
            node_trn.dep = sorted_dep_list
            self.schedule_dependencies.append(node_trn)

        return None

    def findDelaysFromData(self) -> None:
        '''
        For each node in the graph, find all occurances in the dataframe per date
        Updates the list_of_delays variable
        :return None:
        '''
        # for each train and its dependencies
        for node in self.schedule_dependencies:
            df_filter = self.df.copy()
            # create a new column event to filter easily since this is the same as trn.getSmallerID()
            df_filter["event"] = df_filter.apply(lambda x: x['basic|treinnr']+"_"+x['basic|drp']+"_"+x['basic|drp_act'], axis=1)
            # create the corresponding columns
            events = list(map(lambda x: x.getSmallerID(), node.get_causal_trains()))
            # filter the events
            df_filter = df_filter[df_filter['event'].isin(events)]
            # minimalize the columns of the df
            df_filter = df_filter[['event', "delay", "date"]]
            # change the df such that per date there is one row with delays per events
            df_filter = df_filter.pivot(index='date', columns='event', values='delay')
            # reset index such that the data column is also included
            df_filter = df_filter.reset_index()
            missing_columns = []
            column_order, missing_columns = node.get_causal_trains_columns()
            column_order  = ["date"] + column_order
            df_filter[missing_columns] = 0 # add the missing columns and set all delays of the missing columns to 0 #TODO: Maybe change it to -100 or something?
            df_filter = df_filter[column_order] # set teh correct ordering as provided in the column order
            # fore each row in the dataframe, add it to the list
            for row in df_filter.values:
                self.list_nn_input.append(NN_input_sample(node.trn, row))

    def NN_input_class_to_matrix(self, filename):
        x = []
        y = []
        for item in self.list_nn_input:
            row = [item.label, item.trainserie ,*item.dep_delay, item.type_train.value, item.type_station.value, item.day_of_the_week.value, item.peakhour.value, item.buffer, item.traveltime]
            x.append(row)
            y.append(item.trn_delay)

        #create dataframe to save intermediate computation time next time
        df = pd.DataFrame(x, columns=["id", "trainserie","prev_event", "prev_platform", "dep1", "dep2", "dep3", "type_train", "type_station", "day_of_the_week", "peakhour", "buffer", "traveltime"])
        # maybe add travel time between each stop?
        df = df.assign(delay = y)
        df.to_csv(filename, index = False, sep = ";")
        return np.array(x), np.array(y)

    def filterPrimaryDelay(self):
        p_delays = []
        for event in self.list_nn_input:
            if all(x <= 0 for x in event.dep_delay):
                p_delays.append(event.trn_delay)
        return p_delays
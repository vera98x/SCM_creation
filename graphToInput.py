import copy

from causallearn.graph.GeneralGraph import GeneralGraph
from FAS import FAS_method
from enum import Enum
from typing import Dict
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

class NN_delay_sample:
    def __init__(self, label, delays, buffer):
        self.label = label
        self.trn_delay = delays[0]
        self.dep_delay = delays[1:]
        self.type_train = Train_type.SPR
        self.type_station = Station_type.SMALL
        self.day_of_the_week = Weekday.MONDAY
        self.peakhour = Peak.NONPEAK
        self.buffer = buffer


class NN_sample:
    def __init__(self, trn):
        self.trn = trn
        self.prev_event = None
        self.prev_platform = None
        self.dep = []

    def getListSample(self):
        l = []
        l.append(self.trn)
        if self.prev_event != None:
            l.append(self.prev_event)
        if self.prev_platform != None:
            l.append(self.prev_platform)
        if len(self.dep) > 0:
            l += self.dep
        return l


class NN_samples:
    def __init__(self, gg: GeneralGraph, fas_method: FAS_method, df):
        self.sample_data = []
        self.gg = gg
        self.id_trn = fas_method.id_trn_dict
        self.df = df
        self.list_of_delays = []

    def gg_to_nn_input(self, events):
        nodes = self.gg.get_nodes()
        for node in nodes:
            trn = self.id_trn[node.get_name()]
            samp = NN_sample(trn)
            parents = self.gg.get_parents(node)
            print("-------------------------------")
            print(trn.getID())
            # only use the stations we know all dependencies from
            if((trn.getStation(), trn.getTrainSerie()) not in events):
                continue
            print("continue with", (trn.getStation(), trn.getTrainSerie()))
            for parent in parents:
                p_trn = self.id_trn[parent.get_name()]
                if (trn.getPlatform() == p_trn.getPlatform() and trn.getStation() == p_trn.getStation()):
                    if(samp.prev_platform != None):
                        if(samp.prev_platform.getPlannedTime() < p_trn.getPlannedTime()):
                            samp.dep.append(samp.prev_platform)
                            samp.prev_platform = p_trn
                            print("platform Update", p_trn.getID())
                        else:
                            samp.dep.append(p_trn)
                            print("Platform, but not last", p_trn.getID())
                    else:
                        samp.prev_platform = p_trn
                        print("platform", p_trn.getID())
                elif (trn.getTrainRideNumber() == p_trn.getTrainRideNumber()):
                    samp.prev_event = p_trn
                    print("prev", p_trn.getID())
                else:
                    samp.dep.append(p_trn)
                    print("other parent", p_trn.getID())
            print("-------------------------------")
            self.sample_data.append(samp)

        return None

    def findDelaysFromData(self):
        for sample in self.sample_data:
            df_filter = self.df.copy()
            df_filter["event"] = df_filter.apply(lambda x: x['basic|treinnr']+"_"+x['basic|drp']+"_"+x['basic|drp_act'], axis=1)
            events = list(map(lambda x: x.getSmallerID(), sample.getListSample()))
            df_filter = df_filter[df_filter['event'].isin(events)]
            df_filter = df_filter[['event', "delay", "date"]]
            df_filter = df_filter.pivot(index='date', columns='event', values='delay')

            missing_columns = []
            column_order = []
            column_order.append(sample.trn.getSmallerID())
            if sample.prev_event != None:
                column_order.append(sample.prev_event.getSmallerID())
            else:
                missing_columns.append("prev_event")
                column_order.append("prev_event")
            if sample.prev_platform != None:
                column_order.append(sample.prev_platform.getSmallerID())
            else:
                missing_columns.append("prev_platform")
                column_order.append("prev_platform")
            i = 0
            for dep in sample.dep:
                column_order.append(dep.getSmallerID())
                i += 1
            while i < 3:
                missing_columns.append("dep"+str(i))
                column_order.append("dep"+str(i))
                i+=1
            df_filter[missing_columns] = 0
            df_filter = df_filter[column_order]

            for row in df_filter.values:
                self.list_of_delays.append(NN_delay_sample(sample.trn.getSmallerID(), row, sample.trn.getBuffer()))

    def NN_delay_sample_to_matrix(self):
        x = []
        y = []
        for item in self.list_of_delays:
            row = [item.label, *item.dep_delay, item.type_train.value, item.type_station.value, item.day_of_the_week.value, item.peakhour.value, item.buffer]
            x.append(row)
            y.append(item.trn_delay)

        df = pd.DataFrame(x, columns=["id", "prev_event", "prev_platform", "dep1", "dep2", "dep3", "type_train", "type_station", "day_of_the_week", "peakhour", "buffer"])
        df = df.assign(delay = y)
        df.to_csv("Results/nn_input_filtered.csv", index = False, sep = ";")
        return np.array(x), np.array(y)

    def filterPrimaryDelay(self):
        p_delays = []
        for event in self.list_of_delays:
            if all(x <= 0 for x in event.dep_delay):
                p_delays.append(event.trn_delay)
        return p_delays
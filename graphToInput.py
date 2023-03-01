import copy

from causallearn.graph.GeneralGraph import GeneralGraph
from FAS import FAS_method
from typing import Dict

class NN_delay_sample:
    def __init__(self, label, delays):
        self.label = label
        self.trn_delay = delays[0]
        self.dep_delay = delays[1:]


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

    def gg_to_nn_input(self, filtered_station, filtered_serie):
        nodes = self.gg.get_nodes()
        print(filtered_serie, filtered_station)
        for node in nodes:
            trn = self.id_trn[node.get_name()]
            samp = NN_sample(trn)
            parents = self.gg.get_parents(node)
            print("-------------------------------")
            print(trn.getID())
            if(trn.getTrainSerie() != filtered_serie or trn.getStation() != filtered_station):
                continue
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
                self.list_of_delays.append(NN_delay_sample(sample.trn.getSmallerID(), row))

    def filterPrimaryDelay(self):
        p_delays = []
        for event in self.list_of_delays:
            if all(x == 0 for x in event.dep_delay):
                p_delays.append(event.trn_delay)
        return p_delays
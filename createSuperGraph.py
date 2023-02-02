"""### Create SuperGraph"""
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
import numpy as np
import time, datetime
from enum import Enum
from typing import Dict, Tuple
from TrainRideNode import TrainRideNode
from createBackground import variableNamesToNumber
from FastBackgroundKnowledge import FastBackgroundKnowledge

from createBackground import backgroundToGraph, createStationDict, addRequiredBasedTrainSerie,addForbiddenBasedOnStation, \
    addDependency, addForbiddenDependency, removeForbiddenDependency

class Graph_type(Enum):
    SUPER = 1
    MINIMAL = 2

class DomainKnowledge:
    def __init__(self, sched_with_classes : np.array, filename: str, type : Graph_type):
        self.graph_type = type
        self.sched_with_classes = sched_with_classes
        self.filename = filename
        self.trn_name_id_dict, self.id_trn_name_dict = variableNamesToNumber(sched_with_classes)
        f = lambda x: x.getID()
        self.column_names = np.array(list(map(f, sched_with_classes)))
        self.station_dict = createStationDict(sched_with_classes)
    def makeEverythingForbidden(self, train_serie_day: np.array, bk: FastBackgroundKnowledge) -> FastBackgroundKnowledge:
        for trn_index in range(len(train_serie_day)):
            trn = train_serie_day[trn_index]
            for other_trn_index in range(trn_index + 1, len(train_serie_day)):
                other_trn = train_serie_day[other_trn_index]
                bk = addForbiddenDependency(trn.getID(), other_trn.getID(), bk, self.trn_name_id_dict)
                bk = addForbiddenDependency(other_trn.getID(), trn.getID(), bk, self.trn_name_id_dict)
        return bk
    def addRequiredOfNotForbidden(self, train_serie_day: np.array, bk: FastBackgroundKnowledge) -> FastBackgroundKnowledge:
        for trn_index in range(len(train_serie_day)):
            trn = train_serie_day[trn_index]
            for other_trn in range(trn_index + 1, len(train_serie_day)):
                node1 = GraphNode(self.trn_name_id_dict[trn.getID()])
                node2 = GraphNode(self.trn_name_id_dict[other_trn.getID()])
                if not bk.is_forbidden(node1, node2):
                  bk = addDependency(trn.getID(), other_trn.getID(), bk, self.trn_name_id_dict)
        return bk

    def addRequiredBasedStation(self, bk: BackgroundKnowledge) -> FastBackgroundKnowledge:
        buffer = 30  # minutes
        for station, station_list in self.station_dict.items():
            station_list.sort(key=lambda x: x[0])
            for station_index in range(len(station_list)):
                (time_trn, trn) = station_list[station_index]
                for station2_index in range(station_index+1, len(station_list)):
                    (other_time_trn, other_trn) = station_list[station2_index]
                    if ((other_time_trn.hour - time_trn.hour) * 60 + (other_time_trn.minute - time_trn.minute)) <= buffer:
                        if(self.graph_type == Graph_type.SUPER):
                            bk = addDependency(trn.getID(), other_trn.getID(), bk, self.trn_name_id_dict)
                        if(self.graph_type == Graph_type.MINIMAL):
                            bk = removeForbiddenDependency(trn.getID(), other_trn.getID(), bk, self.trn_name_id_dict)
                            bk = removeForbiddenDependency(other_trn.getID(), trn.getID(), bk, self.trn_name_id_dict)
        return bk
    def createSuperGraph(self) -> FastBackgroundKnowledge:
        # make sure that the order of train rides is ordered by train numbers

        # create background knowledge, take into account:
        # 1. The train that follows its route (chain of actions), depends on the previous action of the train, thus there is a direct cause.
        # 2. Trains that are on the same station within x minutes may have a direct cause.
        # 3. Trains that are not in the same station cannot have a direct cause.

        bk = FastBackgroundKnowledge()
        # first add the required edges, themn the forbidden edges (forbidden edges checks if some edge was already required, then it does not add a forbidden edge)
        bk = self.makeEverythingForbidden(self.sched_with_classes, bk)
        bk = addRequiredBasedTrainSerie(self.sched_with_classes, bk, self.trn_name_id_dict)
        bk = self.addRequiredBasedStation(bk)
        return bk


    def get_CG_and_superGraph(self) -> Tuple[BackgroundKnowledge, CausalGraph]:

        print("Creating background knowledge")
        start = time.time()
        background = self.createSuperGraph()
        end = time.time()
        print("creating schedule took", end - start, "seconds")

        cg_sched = backgroundToGraph(background, self.column_names, self.trn_name_id_dict)
        # labels = column_names
        pdy = GraphUtils.to_pydot(cg_sched.G, labels=self.column_names)
        pdy.write_png(self.filename)
        background.required_rules_to_dict()
        background.forbidden_rules_to_dict()
        return background, cg_sched
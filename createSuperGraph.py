"""### Create SuperGraph"""
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
import numpy as np
import time, datetime
from typing import Dict, Tuple
from TrainRideNode import TrainRideNode
from createBackground import variableNamesToNumber
from FastBackgroundKnowledge import FastBackgroundKnowledge

from createBackground import backgroundToGraph, createStationDict, addRequiredBasedTrainSerie,addForbiddenBasedOnStation, \
    addDependency, addForbiddenDependency

def makeEverythingForbidden(train_serie_day: np.array, bk: BackgroundKnowledge,
                               mapper_dict: Dict[str,str]) -> BackgroundKnowledge:
    for trn_index in range(len(train_serie_day)):
        trn = train_serie_day[trn_index]
        for other_trn_index in range(trn_index + 1, len(train_serie_day)):
            other_trn = train_serie_day[other_trn_index]
            bk = addForbiddenDependency(trn.getID(), other_trn.getID(), bk, mapper_dict)
            bk = addForbiddenDependency(other_trn.getID(), trn.getID(), bk, mapper_dict)
    return bk
def addRequiredOfNotForbidden(train_serie_day: np.array, bk: BackgroundKnowledge,
                               mapper_dict: Dict[str,str]) -> BackgroundKnowledge:
    for trn_index in range(len(train_serie_day)):
        trn = train_serie_day[trn_index]
        for other_trn in range(trn_index + 1, len(train_serie_day)):
            node1 = GraphNode(mapper_dict[trn.getID()])
            node2 = GraphNode(mapper_dict[other_trn.getID()])
            if not bk.is_forbidden(node1, node2):
              bk = addDependency(trn.getID(), other_trn.getID(), bk, mapper_dict)
    return bk

def addRequiredBasedStation(bk: BackgroundKnowledge,
                               mapper_dict: Dict[str,str], station_dict : Dict[str, Tuple[datetime.time, TrainRideNode]]) -> BackgroundKnowledge:
    buffer = 15  # minutes
    for station, station_list in station_dict.items():
        station_list.sort(key=lambda x: x[0])
        for station_index in range(len(station_list)):
            (time_trn, trn) = station_list[station_index]
            for station2_index in range(station_index+1, len(station_list)):
                (other_time_trn, other_trn) = station_list[station2_index]
                if ((other_time_trn.hour - time_trn.hour) * 60 + (other_time_trn.minute - time_trn.minute)) <= buffer:
                    bk = addDependency(trn.getID(), other_trn.getID(), bk, mapper_dict)
    return bk
def createSuperGraph(sched_with_classes: np.array, mapper_dict: Dict[str,str]) -> BackgroundKnowledge:
    # make sure that the order of train rides is ordered by train numbers

    # create background knowledge, take into account:
    # 1. The train that follows its route (chain of actions), depends on the previous action of the train, thus there is a direct cause.
    # 2. Trains that are on the same station within x minutes may have a direct cause.
    # 3. Trains that are not in the same station cannot have a direct cause.
    station_dict = createStationDict(sched_with_classes)
    bk = FastBackgroundKnowledge()
    # first add the required edges, themn the forbidden edges (forbidden edges checks if some edge was already required, then it does not add a forbidden edge)
    bk = makeEverythingForbidden(sched_with_classes, bk, mapper_dict)
    bk = addRequiredBasedTrainSerie(sched_with_classes, bk, mapper_dict)
    bk = addRequiredBasedStation(bk, mapper_dict, station_dict)
    return bk


def get_CG_and_superGraph(sched_with_classes : np.array, filename: str) -> Tuple[BackgroundKnowledge, CausalGraph]:
    trn_name_id_dict, id_trn_name_dict = variableNamesToNumber(sched_with_classes)
    print("Creating background knowledge")
    start = time.time()
    background = createSuperGraph(sched_with_classes, trn_name_id_dict)
    end = time.time()
    print("creating schedule took", end - start, "seconds")
    f = lambda x: x.getID()
    column_names = np.array(list(map(f, sched_with_classes)))

    cg_sched = backgroundToGraph(background, column_names, trn_name_id_dict)
    # labels = column_names
    pdy = GraphUtils.to_pydot(cg_sched.G, labels=column_names)
    pdy.write_png(filename)
    background.required_rules_to_dict()
    background.forbidden_rules_to_dict()
    return background, cg_sched
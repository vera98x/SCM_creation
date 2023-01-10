"""### Create BackgroundKnowlegde"""
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
import numpy as np


def addDependency(node1_name: str, node2_name: str, bk: BackgroundKnowledge, mapper_dict: dict) -> BackgroundKnowledge:
    # add directed edge between the two nodes, node1 --> node2
    node1 = GraphNode(mapper_dict[node1_name])
    node2 = GraphNode(mapper_dict[node2_name])

    bk = removeForbiddenDependency(node1_name, node2_name, bk, mapper_dict)
    bk = removeForbiddenDependency(node2_name, node1_name, bk, mapper_dict)
    bk = bk.add_required_by_node(node1, node2)

    return bk


def removeDependency(node1_name: str, node2_name: str, bk: BackgroundKnowledge,
                     mapper_dict: dict) -> BackgroundKnowledge:
    # remove directed edge between the two nodes, node1 --> node2
    node1 = GraphNode(mapper_dict[node1_name])
    node2 = GraphNode(mapper_dict[node2_name])
    bk = bk.remove_required_by_node(node1, node2)
    return bk


def addForbiddenDependency(node1_name: str, node2_name: str, bk: BackgroundKnowledge,
                           mapper_dict: dict) -> BackgroundKnowledge:
    # add forbidden edge between the two nodes, node1 --> node2
    node1 = GraphNode(mapper_dict[node1_name])
    node2 = GraphNode(mapper_dict[node2_name])

    if bk.is_required(node1, node2):
        return bk

    bk = bk.add_forbidden_by_node(node1, node2)

    return bk


def removeForbiddenDependency(node1_name: str, node2_name: str, bk: BackgroundKnowledge,
                              mapper_dict: dict) -> BackgroundKnowledge:
    node1 = GraphNode(mapper_dict[node1_name])
    node2 = GraphNode(mapper_dict[node2_name])
    bk = bk.remove_forbidden_by_node(node1, node2)
    return bk


def addRequiredBasedTrainSerie(train_serie_day: np.array, bk: BackgroundKnowledge,
                               mapper_dict: dict) -> BackgroundKnowledge:
    # trainseries contains a 1D array containing TreinRideNode data of one day
    # for each train ride and every stop/station, add a chain of dependencies
    # s1->s2->s3
    prev_name = ""
    prev_trainNumber = ""
    for trn in train_serie_day:
        # skip first station
        if trn.getTrainRideNumber() != prev_trainNumber:
            prev_trainNumber = trn.getTrainRideNumber()
            prev_name = trn.getID()

            continue
        bk = addDependency(prev_name, trn.getID(), bk, mapper_dict)
        prev_name = trn.getID()

    return bk


def createStationDict(train_serie_day: np.array):
    station_dict = {}  # example: bkl: [(8, TrainRideNode1), (13, TrainRideNode2)]

    # for each station, list all the trains that arrive there with its arrival time and TrainRideNode
    for trn in train_serie_day:
        if trn.getStation() not in station_dict.keys():
            station_dict[trn.getStation()] = [(trn.getPlannedTime(), trn)]
        else:
            arrivalpairs = station_dict[trn.getStation()]
            station_dict[trn.getStation()] = arrivalpairs + [(trn.getPlannedTime(), trn)]
    return station_dict


def addRequiredBasedOnStation(train_serie_day: np.array, bk: BackgroundKnowledge, mapper_dict: dict,
                              station_dict: dict) -> BackgroundKnowledge:
    buffer = 60  # minutes
    # for each station, sort the list on arrival time and if the arrival time is within the buffer time, add a dependency
    for station_list in station_dict.values():
        station_list.sort(key=lambda x: x[0])
        prev = ""
        for (time_trn, trn) in station_list:
            if prev == "":
                prev = (time_trn, trn)
                continue
            if ((time_trn.hour - prev[0].hour) * 60 + (time_trn.minute - prev[0].minute)) <= buffer:
                bk = addDependency(prev[1].getID(), trn.getID(), bk, mapper_dict)
            prev = (time_trn, trn)
    return bk


def addForbiddenStationTimeWise(time_trn, trn, same_station_list, bk, mapper_dict):
    buffer = 60  # minutes
    for (same_time_trn, same_trn) in same_station_list:
        if same_trn.getID() == trn.getID():
            return bk
        if ((time_trn.hour - same_time_trn.hour) * 60 + (time_trn.minute - same_time_trn.minute)) > buffer:
            bk = addForbiddenDependency(same_trn.getID(), trn.getID(), bk, mapper_dict)
            bk = addForbiddenDependency(trn.getID(), same_trn.getID(), bk, mapper_dict)

    return bk


def addForbiddenBasedOnStation(train_serie_day: np.array, bk: BackgroundKnowledge, mapper_dict: dict,
                               station_dict: dict) -> BackgroundKnowledge:
    # add forbidden dependencies if they are not at the same station
    for station, station_list in station_dict.items():
        # for each train + station in the list
        # loop again through the list of stations and its arriving trains
        # and if that station is not the train station of that train, add a forbidden dependency both ways
        for (time_trn, trn) in station_list:
            for other_station, other_station_list in station_dict.items():
                if station == other_station:
                    bk = addForbiddenStationTimeWise(time_trn, trn, other_station_list, bk, mapper_dict)
                    break
                for (other_time_trn, other_trn) in other_station_list:
                    bk = addForbiddenDependency(other_trn.getID(), trn.getID(), bk, mapper_dict)
                    bk = addForbiddenDependency(trn.getID(), other_trn.getID(), bk, mapper_dict)
    return bk


def createBackgroundKnowledge(train_series: np.array, mapper_dict: dict) -> BackgroundKnowledge:
    # make sure that the order of train rides is ordered by train numbers

    # create background knowledge, take into account:
    # 1. The train that follows its route (chain of actions), depends on the previous action of the train, thus there is a direct cause.
    # 2. Trains that are on the same station within x minutes may have a direct cause.
    # 3. Trains that are not in the same station cannot have a direct cause.
    station_dict = createStationDict(train_series[0])
    bk = BackgroundKnowledge()
    # first add the required edges, themn the forbidden edges (forbidden edges checks if some edge was already required, then it does not add a forbidden edge)
    bk = addRequiredBasedOnStation(train_series[0], bk, mapper_dict, station_dict)
    bk = addRequiredBasedTrainSerie(train_series[0], bk, mapper_dict)
    bk = addForbiddenBasedOnStation(train_series[0], bk, mapper_dict, station_dict)

    return bk


def backgroundToGraph(bk: BackgroundKnowledge, column_names, mapper_dict) -> CausalGraph:
    # create all nodes
    nodes = [GraphNode(mapper_dict[i]) for i in column_names]
    # form to CausalGraph
    cg = CausalGraph(len(column_names), nodes)
    # It is not possible to add edges to the CG, so create GG and add the edges that are found in the required_rules_specs
    gg = GeneralGraph(nodes)
    for x in bk.required_rules_specs:
        gg.add_directed_edge(x[0], x[1])
    # for y in bk.forbidden_rules_specs:
    #   gg.add_directed_edge(y[0],y[1])
    #   gg.add_directed_edge(y[1],y[0])
    # add this GG to the CG
    cg.G = gg
    return cg


def variableNamesToNumber(day):
    counter = 1
    mapper_dict = {}
    for trn in day:
        mapper_dict[trn.getID()] = 'X' + str(counter)
        counter += 1

    return mapper_dict


def get_CG_and_background(dataset_with_classes, filename: str):
    day = dataset_with_classes[0]
    mapper_dict = variableNamesToNumber(day)
    background = createBackgroundKnowledge(dataset_with_classes, mapper_dict)
    f = lambda x: x.getID()
    column_names = np.array(list(map(f, day)))

    cg_sched = backgroundToGraph(background, column_names, mapper_dict)
    # labels = column_names
    pdy = GraphUtils.to_pydot(cg_sched.G, labels=column_names)
    pdy.write_png(filename)
    return background, cg_sched
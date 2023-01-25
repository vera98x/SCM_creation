from TrainRideNode import TrainRideNode
from createBackground import createStationDict, addForbiddenBasedOnStation, variableNamesToNumber, addRequiredBasedOnStation,addRequiredBasedTrainSerie
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
import datetime
import numpy as np


# TrainRideNode(trainride['basic|treinnr'], trainride['basic|drp'], trainride['basic|spoor'],trainride['basic|drp_act'], trainride['delay'], trainride['plan|time'])

def backgroundToGraphRequired(bk: BackgroundKnowledge, column_names, mapper_dict) -> CausalGraph:
    # create all nodes
    nodes = [GraphNode(mapper_dict[i]) for i in column_names]
    # form to CausalGraph
    cg = CausalGraph(len(column_names), nodes)
    # It is not possible to add edges to the CG, so create GG and add the edges that are found in the required_rules_specs
    gg = GeneralGraph(nodes)
    for x in bk.required_rules_specs:
        gg.add_directed_edge(x[0], x[1])
    # add this GG to the CG
    cg.G = gg
    return cg

def backgroundToGraphForbidden(bk: BackgroundKnowledge, column_names, mapper_dict) -> CausalGraph:
    # create all nodes
    nodes = [GraphNode(mapper_dict[i]) for i in column_names]
    # form to CausalGraph
    cg = CausalGraph(len(column_names), nodes)
    # It is not possible to add edges to the CG, so create GG and add the edges that are found in the required_rules_specs
    gg = GeneralGraph(nodes)
    for y in bk.forbidden_rules_specs:
      gg.add_directed_edge(y[0],y[1])
      gg.add_directed_edge(y[1],y[0])
    # add this GG to the CG
    cg.G = gg
    return cg

def getTestNodes():
    nodes = []
    nodes.append(TrainRideNode(1000, "Bkl", 0,"A", 0, datetime.time(0,0,0)))
    nodes.append(TrainRideNode(1000, "Mas", 0,"A", 0, datetime.time(0,45,0)))
    nodes.append(TrainRideNode(1000, "Utzl", 0,"A", 0, datetime.time(1,2,0)))

    nodes.append(TrainRideNode(1002, "Bkl", 0,"A", 0, datetime.time(0,30,0)))
    nodes.append(TrainRideNode(1002, "Mas", 0,"A", 0, datetime.time(1,15,0)))
    nodes.append(TrainRideNode(1002, "Utzl", 0,"A", 0, datetime.time(1,30,0)))

    nodes.append(TrainRideNode(1004, "Bkl", 0,"A", 0, datetime.time(1,0,0)))
    nodes.append(TrainRideNode(1004, "Mas", 0,"A", 0, datetime.time(1,45,0)))
    nodes.append(TrainRideNode(1004, "Utzl", 0,"A", 0, datetime.time(2,0,0)))

    nodes.append(TrainRideNode(1001, "Utzl", 0,"A", 0, datetime.time(0,0,0)))
    nodes.append(TrainRideNode(1001, "Mas", 0,"A", 0, datetime.time(0,40,0)))
    nodes.append(TrainRideNode(1001, "Bkl", 0,"A", 0, datetime.time(1,0,0)))

    nodes.append(TrainRideNode(1003, "Utzl", 0,"A", 0, datetime.time(0,0,0)))
    nodes.append(TrainRideNode(1003, "Mas", 0,"A", 0, datetime.time(1,10,0)))
    nodes.append(TrainRideNode(1003, "Bkl", 0,"A", 0, datetime.time(1,30,0)))
    return np.array(nodes)

day = getTestNodes()

mapper_dict = variableNamesToNumber(day)
station_dict = createStationDict(day)

f = lambda x: x.getID()
column_names = np.array(list(map(f, day)))

bk = BackgroundKnowledge()
bk = addForbiddenBasedOnStation([], bk, mapper_dict, station_dict)

bk = addRequiredBasedOnStation([], bk, mapper_dict, station_dict)
bk = addRequiredBasedTrainSerie(day, bk, mapper_dict)
bk = addForbiddenBasedOnStation([], bk, mapper_dict, station_dict)
print(bk.required_rules_specs)
print(bk.forbidden_rules_specs)

bk.forbidden_rules_to_dict()
bk.required_rules_to_dict()

print()

node1 = GraphNode("X1")
node2 = GraphNode("X4")

print(bk.is_forbidden(node1, node2))
print(bk.is_required(node1, node2))



# req_cg = backgroundToGraphRequired(bk, column_names, mapper_dict)
# GraphUtils.to_pydot(req_cg.G, labels=column_names).write_png("req.png")
# forb_cg = backgroundToGraphForbidden(bk, column_names, mapper_dict)
# GraphUtils.to_pydot(forb_cg.G, labels=column_names).write_png("forb.png")


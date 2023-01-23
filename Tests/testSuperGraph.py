from TrainRideNode import TrainRideNode
from createSuperGraph import get_CG_and_superGraph
import datetime
import numpy as np

def getTestNodes():
    nodes = []
    nodes.append(TrainRideNode(1000, "Bkl", 0,"A", 0, datetime.time(0,0,0)))
    nodes.append(TrainRideNode(1000, "Mas", 0,"A", 0, datetime.time(0,5,1)))
    nodes.append(TrainRideNode(1000, "Utzl", 0,"A", 0, datetime.time(0,10,2)))

    nodes.append(TrainRideNode(1002, "Bkl", 0,"A", 0, datetime.time(0,0,1)))
    nodes.append(TrainRideNode(1002, "Mas", 0,"A", 0, datetime.time(0,5,2)))
    nodes.append(TrainRideNode(1002, "Utzl", 0,"A", 0, datetime.time(0, 10,3)))

    nodes.append(TrainRideNode(1004, "Bkl", 0,"A", 0, datetime.time(0,0,2)))
    nodes.append(TrainRideNode(1004, "Mas", 0,"A", 0, datetime.time(0,5,3)))
    nodes.append(TrainRideNode(1004, "Utzl", 0,"A", 0, datetime.time(0,10,4)))

    nodes.append(TrainRideNode(1001, "Utzl", 0,"A", 0, datetime.time(0,10,1)))
    nodes.append(TrainRideNode(1001, "Mas", 0,"A", 0, datetime.time(0,5,4)))
    nodes.append(TrainRideNode(1001, "Bkl", 0,"A", 0, datetime.time(0,0,3)))

    nodes.append(TrainRideNode(1003, "Utzl", 0,"A", 0, datetime.time(2,0,0)))
    nodes.append(TrainRideNode(1003, "Mas", 0,"A", 0, datetime.time(2,0,5)))
    nodes.append(TrainRideNode(1003, "Bkl", 0,"A", 0, datetime.time(2,0,4)))
    return np.array(nodes)

trn_nodes = getTestNodes()
bk, cg_sched = get_CG_and_superGraph([trn_nodes], "test_supergraph.png")
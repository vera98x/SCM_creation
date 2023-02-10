from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
import numpy as np
import warnings
from causallearn.utils.cit import *

from typing import Dict, Tuple, List
from TrainRideNode import TrainRideNode
from FastBackgroundKnowledge import FastBackgroundKnowledge
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import datetime

class FAS_method():
    def __init__(self,method : str, data : np.array, filename : str, sched_with_classes: np.array, mapper_dict : Dict[str, str], column_names : List[str] = None, bk : BackgroundKnowledge = None):
        self.method = method
        self.data = data
        self.filename = filename
        self.id_trn_dict = self.createIDTRNDict(sched_with_classes)
        self.mapper_dict = mapper_dict
        self.column_names = column_names
        self.bk = bk
        self.alpha = 0.05


    def startupFAS(self):
        depth = -1

        if self.data.shape[0] < self.data.shape[1]:
            warnings.warn("The number of features is much larger than the sample size!")

        independence_test_method = CIT(self.data, method=self.method)

        ## ------- check parameters ------------
        if (self.bk is not None) and (type(self.bk) != BackgroundKnowledge and type(self.bk) != FastBackgroundKnowledge):
            raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
        ## ------- end check parameters ------------

        nodes = []
        for i in range(self.data.shape[1]):
            node_alias = f"X{i + 1}"
            node_name = self.mapper_dict[node_alias]
            node = GraphNode(node_alias)
            node.add_attribute("id", i)
            #trn = self.id_trn_dict[node_name]
            #node.add_attribute("trainnumber", trn.getTrainRideNumber())
            #node.add_attribute("time", trn.getPlannedTime())
            nodes.append(node)

        graph, sep_sets = fas(self.data, nodes, independence_test_method=independence_test_method, alpha=self.alpha,
                              knowledge=self.bk, depth=depth, verbose=False)
        # using fas will delete all attributes of the node, hence override the nodes with the one with the attributes
        #graph.nodes = nodes
        return graph, sep_sets

    def createIDTRNDict(self, sched_with_classes : np.array) -> Dict[str, TrainRideNode]:
        result_dict = {}
        for trn in sched_with_classes:
            result_dict[trn.getID()] = trn
        return result_dict

    def orientEdges(self, ggFas : GeneralGraph) -> GeneralGraph:
        nodes = ggFas.get_nodes()
        num_vars = len(nodes)
        for node in nodes:
            node_name = self.mapper_dict[node.get_name()]
            trn_time = self.id_trn_dict[node_name].getPlannedTime()
            node.add_attribute('time', trn_time)
        edges = ggFas.get_graph_edges()
        # empty the complete graph
        ggFas.graph = np.zeros((num_vars, num_vars), np.dtype(int))
        # add new nodes
        for edge in edges:
            # get nodes from edge
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            # map edges to TRN + get time
            trn1_time = node1.get_attribute('time')
            trn2_time = node2.get_attribute('time')
            #order in timewise
            if trn1_time > trn2_time:
                #add directed edge
                ggFas.add_directed_edge(node2, node1)

            else:
                # add directed edge
                ggFas.add_directed_edge(node1, node2)
        return ggFas

    def removePossibleDsep(self, ggFas, sep_sets):
        sp = SepsetsPossibleDsep(self.data, ggFas, self.method, self.alpha, self.bk, -1,
                                 -1, verbose= False)
        waiting_to_deleted_edges = []

        for edge in ggFas.get_graph_edges():
            node_x = edge.get_node1()
            node_y = edge.get_node2()

            sep_set = sp.get_sep_set(node_x, node_y)

            if sep_set is not None:
                waiting_to_deleted_edges.append((node_x, node_y, sep_set))

        for waiting_to_deleted_edge in waiting_to_deleted_edges:
            dedge_node_x, dedge_node_y, dedge_sep_set = waiting_to_deleted_edge
            ggFas.remove_edge(ggFas.get_edge(dedge_node_x, dedge_node_y))
            sep_sets[(ggFas.node_map[dedge_node_x], ggFas.node_map[dedge_node_y])] = dedge_sep_set

    def fas_with_background(self) -> GeneralGraph:
        with_or_without = "with" if self.bk != None else "without"
        print("start with FAS " + with_or_without + " background")
        start = time.time()
        gg_fas, sep_sets = self.startupFAS()
        end = time.time()
        print("FAS:", "it took", end - start, "seconds")
        self.orientEdges(gg_fas)
        end = time.time()
        print("creating SCM of FAS " + with_or_without + " background is done, it took", end - start, "seconds")

        pdy = GraphUtils.to_pydot(gg_fas, labels=self.column_names)
        pdy.write_png(self.filename)
        return gg_fas


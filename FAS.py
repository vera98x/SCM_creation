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

def startupFAS(dataset: np.array, independence_test_method: str='fisherz', alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge= None,
        **kwargs):

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and (type(background_knowledge) != BackgroundKnowledge and type(background_knowledge) != FastBackgroundKnowledge):
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------

    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    graph, sep_sets = fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                          knowledge=background_knowledge, depth=depth, verbose=verbose)
    return graph

def createIDTRNDict(sched_with_classes : np.array) -> Dict[str, TrainRideNode]:
    result_dict = {}
    for trn in sched_with_classes:
        result_dict[trn.getID()] = trn
    return result_dict

def orientEdges(ggFas : GeneralGraph, id_trn_dict : Dict[str, TrainRideNode], mapper_dict : Dict[str, str]):
    edges = ggFas.get_graph_edges()
    for edge in edges:
        # get nodes from edge
        node1 = edge.get_node1()
        node1_name = mapper_dict[node1.get_name()]
        node2 = edge.get_node2()
        node2_name = mapper_dict[node2.get_name()]
        # map edges to TRN + get time
        trn1_time = id_trn_dict[node1_name].getPlannedTime()
        trn2_time = id_trn_dict[node2_name].getPlannedTime()
        #order in timewise
        if trn1_time > trn2_time:
            #delete old undirected edge
            ggFas.remove_edge(edge)
            #add directed edge
            ggFas.add_edge(Edge(node1, node2, Endpoint.ARROW, Endpoint.TAIL))
        else:
            # delete old undirected edge
            ggFas.remove_edge(edge)
            # add directed edge
            ggFas.add_edge(Edge(node2, node1, Endpoint.ARROW, Endpoint.TAIL))
    return ggFas

def FAS(method : str, data : np.array, filename : str, id_trn_dict: Dict[str, TrainRideNode], mapper_dict : Dict[str, str], column_names : List[str] = None, bk : BackgroundKnowledge = None) -> GeneralGraph:
    with_or_without = "with" if bk != None else "without"
    print("start with FAS " + with_or_without + " background")
    start = time.time()
    gg_fas = startupFAS(data, method, background_knowledge= bk)
    end = time.time()
    print("FAS:", "it took", end - start, "seconds")
    orientEdges(gg_fas, id_trn_dict, mapper_dict)
    end = time.time()
    print("creating SCM of FAS " + with_or_without + " background is done, it took", end - start, "seconds")

    pdy = GraphUtils.to_pydot(gg_fas, labels=column_names)
    pdy.write_png(filename)
    return gg_fas

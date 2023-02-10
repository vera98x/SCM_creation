from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import filecmp

def gg2txt(gg : GeneralGraph, filename : str, id_trn_name_dict) -> None:
    nodes = gg.get_nodes()
    edges = gg.get_graph_edges()
    f = open(filename, "w")
    f.write("Graph Nodes: \n")
    for i, node in enumerate(nodes):
        node_name = id_trn_name_dict[node.get_name()]
        if i == 0:
            f.write(node_name)
        else:
            f.write(";" + node_name)
    f.write("\n\n")
    f.write("Graph Edges:\n")
    for i, edge in enumerate(edges):
        node_name1 = id_trn_name_dict[edge.get_node1().get_name()]
        node_name2 = id_trn_name_dict[edge.get_node2().get_name()]
        symbol1 = "<" if edge.get_endpoint1() == Endpoint.ARROW else "-"
        symbol2 = ">" if edge.get_endpoint2() == Endpoint.ARROW else "-"
        f.write(str(i+1)+". " + node_name1 + " " + symbol1 + "-" + symbol2 + " " + node_name2 + "\n")
    f.close()
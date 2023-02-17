from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import filecmp
import matplotlib.pyplot as plt
from Load_transform_df import retrieveDataframe
import numpy as np

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

def printpfd():
    # A custom function to calculate
    # probability distribution function
    def pdf(x):
        mean = np.mean(x)
        std = np.std(x)
        y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
        return y_out

    export_name = 'Data/2019-03-01_2019-05-31.csv'  # 'Data/Ut_2022-01-01_2022-12-10_2.csv' #'Data/6100_jan_nov_2022_2.csv'
    list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O', '8100E',
                           '8100O', '9000E', '9000O', '12600E', '76200O', '78100E', '78100O', '79000E', '79000O'
                           ]
    df, sched = retrieveDataframe(export_name, True, list_of_trainseries)
    df['delay'] = df['delay'].map(lambda x: int(x / 60))
    # To generate an array of x-values
    datacolumn = df['delay'].tolist()
    x = np.sort(np.asarray(datacolumn))

    # To generate an array of
    # y-values using corresponding x-values
    y = pdf(x)

    # Plotting the bell-shaped curve
    plt.style.use('seaborn')
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='black',
             linestyle='dashed')

    plt.scatter(x, y, marker='o', s=25, color='red')
    plt.show()

def plotDelay():
    export_name = 'Data/2019-03-01_2019-05-31.csv'  # 'Data/Ut_2022-01-01_2022-12-10_2.csv' #'Data/6100_jan_nov_2022_2.csv'
    list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O', '8100E',
                           '8100O', '9000E', '9000O', '12600E','76200O', '78100E', '78100O', '79000E', '79000O'
                           ]
    # extract dataframe and impute missing values
    df, sched = retrieveDataframe(export_name, True, list_of_trainseries)
    df.to_csv("mp_asn_modified.csv", index=False, sep = ";" )
    df['delay'] = df['delay'].map(lambda x: int(x/60))
    ma = df['delay'].max()
    mi = df['delay'].min()

    df.hist(column = 'delay', bins=list(range(mi, ma)))
    plt.xlim(-8, 10)
    plt.ylim(-3, 30000)
    plt.show()

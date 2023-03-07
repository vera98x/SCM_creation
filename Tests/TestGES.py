from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges

import numpy as np
import time

from df_to_trn import TRN_matrix_to_delay_matrix
from createSuperGraph import get_CG_and_superGraph
def completeGES(data, filename, score_func: str = 'local_score_BIC'):
    r = ges(data, score_func)
    # visualization using pydot #
    pdy = GraphUtils.to_pydot(r['G'])
    pdy.write_png(filename)

def backwardGES(X: ndarray, supergraph: GeneralGraph, score_func: str = 'local_score_BIC',
        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    maxP = None
    #variables from forward search:
    if score_func == 'local_score_marginal_multi':  # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel'][i] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)
    elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':  # Greedy equivalence search with BIC score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)
    #set because of bic:

    score_func = localScoreClass
    update1 = []
    G_step1 = [supergraph]
    G = supergraph
    record_local_score = [[] for i in range(N)]

    score = score_g(X, supergraph, score_func, parameters)  # initialize the score
    print('Initial score is:', score)

    # backward greedy search
    print('backward')
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []

    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
                        or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
                    Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                    Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
                    H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                    # for any subset of H0
                    sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                    S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                    # 1: check the condition,
                    # 2: check nothing and is valid;
                    for k in range(len(sub)):
                        if (S[k] == 1):
                            V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                            if (V):
                                # find those subsets that include sub(k)
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2  # and set their S to 2
                        else:
                            V = 1

                        if (V):
                            chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
                                                                                     record_local_score, score_func,
                                                                                     parameters)
                            # calculate the changed score after Insert operator
                            # desc{count} saves the corresponding (i,j,sub{k})
                            if (chscore < min_chscore):
                                min_chscore = chscore
                                min_desc = desc

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= 0:
                break
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}
    return Record

print('Now start test_ges_load_linear_10_with_local_score_BIC ...')
data_path = "data_linear_10.txt"
truth_graph_path = "graph_10.txt"
data = np.loadtxt(data_path, skiprows=1)
truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
print(type(truth_dag))
#truth_cpdag = dag2cpdag(truth_dag)
#num_edges_in_truth = truth_dag.get_num_edges()

# Run GES with default parameters: score_func='local_score_BIC', maxP=None, parameters=None
#res_map = ges(data, score_func='local_score_marginal_general', maxP=None, parameters=None)  # Run GES and obtain the estimated graph (res_map is Dict object，which contains the updated steps, the result causal graph and the result score.)
#print(res_map['score'])

start = time.time()
r = backwardGES(data, truth_dag)
end = time.time()
print()
print("creating GES backward is done, it took" , end - start, "seconds")
print("score:", r['score'])

# print("extracting file")
# export_name = '../Data/6100_jan_nov_2022_2.csv' #'Data/2019-03-01_2019-05-31.csv'
# list_of_trainseries = ['6100']
# dataset_with_classes = getDataSetWith_TRN(export_name, True, list_of_trainseries)
# print("extracting file done")
#
# print("translating dataset to 2d array for algo")
# smaller_dataset = dataset_with_classes[:,:20] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
# delays_to_feed_to_algo, column_names = class_dataset_to_delay_columns_pair(smaller_dataset)
# print("Creating background knowledge")
# start = time.time()
# bk, cg_sched = get_CG_and_superGraph(smaller_dataset, '../Results/sched.png') #get_CG_and_background(smaller_dataset, 'Results/sched.png')
# end = time.time()
# backwardGES(delays_to_feed_to_algo, cg_sched.G)
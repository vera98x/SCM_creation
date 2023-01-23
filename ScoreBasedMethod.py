from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges

import numpy as np
import time

from ETL_data_day import getDataSetWith_TRN, class_dataset_to_delay_columns_pair
from createSuperGraph import get_CG_and_superGraph


# the goal of the ScoreBasedMethod is to first implement the supergraph using domain knowledge and then adding the backward part of the greedy search to optimize this score.

def completeGES(data, filename):
    r = ges(data)
    # visualization using pydot #
    pdy = GraphUtils.to_pydot(r['G'], labels=column_names)
    pdy.write_png(filename)

def backward(X: ndarray, supergraph: GeneralGraph, score_func: str = 'local_score_BIC',
        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #variables from forward search:
    #set because of bic:
    N = X.shape[1]
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)
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


print("extracting file")
export_name = 'Data/6100_jan_nov_2022_2.csv' #'Data/2019-03-01_2019-05-31.csv'
list_of_trainseries = ['6100']
dataset_with_classes = getDataSetWith_TRN(export_name, True, list_of_trainseries)
print("extracting file done")
print(dataset_with_classes)
print("translating dataset to 2d array for algo")
smaller_dataset = dataset_with_classes[:,:80] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
delays_to_feed_to_algo, column_names = class_dataset_to_delay_columns_pair(smaller_dataset)
print("Creating background knowledge")
start = time.time()
bk, cg_sched = get_CG_and_superGraph(smaller_dataset, 'Results/sched.png') #get_CG_and_background(smaller_dataset, 'Results/sched.png')
end = time.time()
print("creating schedule took", end - start, "seconds")
method = 'mv_fisherz' #'fisherz'
print("start with GES and background")
start = time.time()
r = backward(delays_to_feed_to_algo, cg_sched.G)
end = time.time()
print("score:", r['score'])
print()
print("creating SCM with background is done, it took" , end - start, "seconds")
pdy = GraphUtils.to_pydot(r['G'], labels=column_names)
pdy.write_png('Results/6100_jan_nov_with_backg.png')
print("starting normal GES")
start = time.time()
completeGES(delays_to_feed_to_algo, 'Results/6100_jan_nov_wo_backgr.png')
end = time.time()
print("creating SCM without background is done, it took" , end - start, "seconds")
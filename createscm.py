# -*- coding: utf-8 -*-
"""createSCM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u_NWKEWIQxTYy_g3Wd95OW_2Bw7Rm1U8
"""

#dependencies
from causallearn.graph.Node import Node
from causallearn.graph.NodeType import NodeType
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.utils.GraphUtils import GraphUtils

import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot

"""### Compare two SCM's"""

# Compare two SCM's and return the differences
def compareSCM(cg1, cg2, mapper = (lambda a : a)):
  g1 = cg1.G.graph
  g2 = cg2.G.graph
  comparison = g1 == g2
  differences = []
  for node in range(len(comparison)):
    for i, val in enumerate(comparison[node]):
      if val == False:
        # don't add both differences (0,8) and (8,0), only (0,8)
        if (i,node) not in differences:
          differences.append((node, i))
  edges = [] 
  for g in [g1,g2]:
    for i,j in differences:
      if g[i][j] == -1:
        if g[j][i] == -1:
          edges.append(str(mapper(i+1)) + "\t----\t" + str(mapper(j+1)))
        else:
          edges.append(str(mapper(i+1))  + "\t--->\t" +  str(mapper(j+1)))
      if g[i][j] == 1:
        if g[j][i] == 1:
          edges.append(str(mapper(i+1))  + "\t<-->\t" +  str(mapper(j+1)))
        else:
          edges.append(str(mapper(i+1)) + "\t<---\t" +  str(mapper(j+1)))
      if g[i][j] == 0:
          edges.append(str(mapper(i+1))  + "\txxxx\t" +  str(mapper(j+1)))
  
  return list(zip(edges[:len(edges)//2], edges[len(edges)//2:]))


def createCGWithPC(data, filename, column_names, bk = None):
  cg_pc = pc(data, 0.05, fisherz, background_knowledge=bk)

  # visualization using pydot #
  pdy = GraphUtils.to_pydot(cg_pc.G,labels = column_names)
  pdy.write_png(filename)
  return cg_pc

def createCGWithFCI(method, data, filename, column_names = None, bk = None):
  ggFCI, edges = fci(data,independence_test_method = method, background_knowledge = bk)
  col_range = len(data[0])
  #nodes = [GraphNode(i) for i in column_names]
  cgFCI = CausalGraph(col_range, column_names)
  cgFCI.G = ggFCI

  # visualization using pydot #
  pdy = GraphUtils.to_pydot(ggFCI,labels = column_names)
  pdy.write_png(filename)
  return cgFCI

def backgroundToMatrix(bk: BackgroundKnowledge, column_names):
  if (bk == None):
    return None
  prior_knowledge = np.ones((len(column_names), len(column_names))) * -1
  dict_test = {}
  for (i, j) in bk.forbidden_rules_specs:
    index_i = (int(i.get_name().replace('X', ''))-1)
    index_j = (int(j.get_name().replace('X', ''))-1)

    prior_knowledge[index_i][index_j] = 0
    prior_knowledge[index_j][index_i] = 0

  for (i, j) in bk.required_rules_specs:
    index_i = (int(i.get_name().replace('X', ''))-1)
    index_j = (int(j.get_name().replace('X', ''))-1)

    prior_knowledge[index_i][index_j] = 1
    prior_knowledge[index_j][index_i] = 0

  return prior_knowledge

def createCGWithDirectLiNGAM(data, filename, column_names_par = None, bk = None):
  column_names = column_names_par
  pk = None
  if bk != None:
    pk = backgroundToMatrix(bk, column_names)

  if type(column_names_par) == np.ndarray:
    column_names = list(column_names_par)
  model = lingam.DirectLiNGAM(5, prior_knowledge=pk)
  model.fit(data)

  # visualize
  dot_graph = make_dot(model.adjacency_matrix_, labels=column_names)
  dot_graph.render(filename=filename)

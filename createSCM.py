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
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
import time
from typing import List

def createCGWithPC(data : np.array, filename : str, column_names : List[str], bk : BackgroundKnowledge= None) -> CausalGraph:
  cg_pc = pc(data, 0.05, fisherz, background_knowledge=bk)

  # visualization using pydot #
  pdy = GraphUtils.to_pydot(cg_pc.G,labels = column_names)
  pdy.write_png(filename)
  return cg_pc

def createCGWithFCI(method : str, data : np.array, filename : str, column_names : List[str] = None, bk : BackgroundKnowledge = None) -> CausalGraph:
  with_or_without = "with" if bk != None else "without"
  print("start with FCI "+ with_or_without +" background")
  start = time.time()
  ggFCI, edges = fci(data,independence_test_method = method, background_knowledge = bk)
  end = time.time()
  print()

  print("creating SCM "+ with_or_without+" background is done, it took", end - start, "seconds")
  col_range = len(data[0])
  #nodes = [GraphNode(i) for i in column_names]
  cgFCI = CausalGraph(col_range, column_names)
  cgFCI.G = ggFCI

  # visualization using pydot #
  pdy = GraphUtils.to_pydot(ggFCI,labels = column_names)
  pdy.write_png(filename)
  return cgFCI

def createCGWithGES(data, filename, score_func: str = 'local_score_BIC', column_names : List[str] = None):
  print("start with GES ")
  start = time.time()
  r = ges(data, score_func)
  end = time.time()
  print("creating GES is done, it took", end - start, "seconds")
  # visualization using pydot #
  pdy = GraphUtils.to_pydot(r['G'], labels=column_names)
  pdy.write_png(filename)

def backgroundToMatrix(bk: BackgroundKnowledge, column_names : List[str]) -> np.array:
  if (bk == None):
    return None
  prior_knowledge = np.ones((len(column_names), len(column_names))) * -1
  for (i, j) in bk.forbidden_rules_specs:
    index_i = (int(i.get_name().replace('X', ''))-1)
    index_j = (int(j.get_name().replace('X', ''))-1)

    prior_knowledge[index_i][index_j] = 0
    prior_knowledge[index_j][index_i] = 0

  for (i, j) in bk.required_rules_specs:
    index_i = (int(i.get_name().replace('X', ''))-1)
    index_j = (int(j.get_name().replace('X', ''))-1)

    prior_knowledge[index_i][index_j] = 0
    prior_knowledge[index_j][index_i] = 1

  return prior_knowledge

def createCGWithDirectLiNGAM(data, filename, column_names_par = None, bk = None) -> GeneralGraph:
  print("start with DirectLingam")
  start = time.time()

  column_names = column_names_par
  pk = None
  if bk != None:
    pk = backgroundToMatrix(bk, column_names)

  if type(column_names_par) == np.ndarray:
    column_names = list(column_names_par)

  model = lingam.DirectLiNGAM(5, prior_knowledge=pk)
  model.fit(data)

  end = time.time()
  print("creating Lingam is done, it took", end - start, "seconds")
  # visualize
  dot_graph = make_dot(model.adjacency_matrix_, labels=column_names)
  dot_graph.render(filename=filename)
  nodes = [GraphNode('X' + str(i)) for i in range(len(column_names))]
  gg_lingam = GeneralGraph(nodes)
  gg_lingam.graph = model.adjacency_matrix_
  pdy = GraphUtils.to_pydot(gg_lingam, labels=column_names)
  pdy.write_png(filename)
  return gg_lingam

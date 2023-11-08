#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### IMPORT PACKAGES
import networkx as nx
import numpy as np
import igraph as ig
from igraph import *
from scipy import stats
from pandas import DataFrame
import numpy as np
from igraph import Graph
from igraph import plot
from igraph import GraphBase
from igraph.clustering import*
from scipy import stats
import igraph
from igraph import Clustering
from igraph import VertexDendrogram
from igraph import VertexClustering
import pandas as pd
import networkx as nx

### FUNCTION DEFINITION
def normalize(matrix):
    return stats.zscore(matrix)


def graph3(x):
    test1 = np.array(x)
    graphx = Graph.Adjacency(test1.tolist(), mode='UNDIRECTED')
    graphx1= GraphBase.simplify(graphx,multiple=True, loops=True, combine_edges=None)
    return graphx1


def assortativity(x):
    coefficient_assortativity=GraphBase.assortativity_degree(x, directed=False)
    return coefficient_assortativity

def Shannon_entropy(distribution):
    entropia=stats.entropy(distribution)
    return entropia
def average_path_length(x):
    apl=GraphBase.average_path_length(x,directed=False, unconn=True)
    return apl

def betweenness(x):
    bc=np.mean(sorted(GraphBase.betweenness(x)))
    return bc

def closeness(x):
    cc=GraphBase.closeness(x)
    for i in cc:
        u= np.mean(i)
    return u

def diameter(x):
    d=GraphBase.diameter(x,directed=False)
    return d

def eigenvector(x):
    e=np.mean(sorted(GraphBase.eigenvector_centrality(x,directed=False)))
    return e

def hub_score(x):
    h=np.mean(sorted(GraphBase.hub_score(x,weights=None)))
    return h

#def independence_number(x):
 #   i=GraphBase.independence_number(x)
  #  return i

def knn(x):
    y= GraphBase.knn(x)
    for i in y:
        k= np.nanmean(sorted(i))
    return k

def pagerank(x):
    p=np.mean(Graph.pagerank(x, vertices=None, directed=True, damping=0.85))
    return p

def transitivity(x):
    t=GraphBase.transitivity_undirected(x)
    return t

#def modularity(x):
 #   m=GraphBase.modularity(x)
  #  return m

def mean_degree(x):
    t= np.mean(GraphBase.degree(x, mode='ALL', loops=True))
    return t

def second_moment(x):
    t=np.var(GraphBase.degree(x, mode='ALL', loops=True))
    return t

def entropy_degree_sequence(x):
    entropia=Shannon_entropy(sorted(GraphBase.degree(x, mode='ALL', loops=True)))
    return entropia

def Shannon_entropy(distribution):
    entropia=stats.entropy(distribution)
    return entropia


def complexidade(x):
   media_grau = np.mean(sorted(GraphBase.degree(x, mode='ALL', loops=True)))
   segundo_momento = np.var(sorted((GraphBase.degree(x, mode='ALL', loops=True))))
   return (segundo_momento / media_grau)
# np.seterr(divide='ignore')

def kcore(x):
    t=np.mean(Graph.coreness(x,mode='ALL'))
    return t

def nodal_eff(g):
    """
    This function calculates the nodal efficiency of a weighted graph object.
    Created by: Loukas Serafeim (seralouk), Nov 2017

    Args:
     g: A igraph Graph() object.
    Returns:
     The nodal efficiency of each node of the graph
    """

    sp = Graph.shortest_paths_dijkstra(g,weights = None)
    sp = np.asarray(sp)
    with np.errstate(divide='ignore'):
        temp = 1.0 / sp
    np.fill_diagonal(temp, 0)
    N = temp.shape[0]
    ne = ( 1.0 / (N - 1)) * np.apply_along_axis(sum, 0, temp)
    for i in ne:
        t=np.mean(sorted(ne))
    return t

def diversity(x):
    u=GraphBase.diversity(x, weights=None)
    return(np.mean(u))

def eccentricity(x):
    u=GraphBase.eccentricity(x)
    return np.mean(u)

def edge_conectivity(x):
    clusters    = x.clusters()
    giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
    #communities = giant.community_spinglass()
    t= GraphBase.edge_connectivity(giant)
    return t
def reciprocity(x):
    u= GraphBase.reciprocity(x,ignore_loops=True, mode="default")
    return u
def average_path_length(x):
    apl=GraphBase.average_path_length(x,directed=False, unconn=True)
    return apl

def tree_spaning(x):
    s=Graph.spanning_tree(x)
    return s

def community_fastgreedy(x):
    u= Graph.community_fastgreedy(x, weights=None)
    t= VertexDendrogram.as_clustering(u)
    return t

def community_infomap(x):
    u= Graph.community_infomap(x, edge_weights=None)

    return u



def community_leading_eigenvector(x):
    u= Graph.community_leading_eigenvector(x)
    return u

def community_label_propagation(x):
    u=Graph.community_label_propagation(x,weights=None)
    return u


def community_multilevel(x):
    u=Graph.community_multilevel(x, weights=None, return_levels=False)
    return u

def community_edge_betweenness(x):
    clusters    = x.clusters()
    giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
    #communities = giant.community_spinglass()
    #t= GraphBase.edge_connectivity(giant)
    u= Graph.community_edge_betweenness(giant, directed=False, weights=None)
    return u

def community_spinglass(x):
    clusters    = x.clusters()
    giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
    communities = giant.community_spinglass()

    #u=Graph.community_spinglass(x)
    #return u
    return communities

def community_walktrap(x):
    clusters    = x.clusters()
    giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
    #communities = giant.community_spinglass()
    #t= GraphBase.edge_connectivity(giant)
    u= Graph.community_walktrap(giant, weights=None)
    
#### FUNTION for compute all measure
def compute_all_features(list_graph):
    list_assortativity=[]
    list_average_path=[]
    list_betweenness=[]
    list_closeness=[]
    list_diameter=[]
    list_eigenvector=[]
    list_hub_score=[]
    list_knn=[]
    list_modularity=[]
    list_transitivity=[]
    list_pagerank=[]
    list_mean_distribution_degree=[]
    list_second_moment_degree=[]
    list_entropy_degree=[]
    list_spaning=[]

    list_complexidade=[]
    list_kcore=[]
    list_effiency=[]
    list_spaning1=[]
    list_community_fatgreedy=[]
    list_community_fatgreedy1=[]
    list_community_fatgreedy2=[]
    list_community_infomap=[]
    list_community_infomap1=[]
    list_community_infomap2=[]
    list_community_leading_eigenvector=[]
    list_community_leading_eigenvector1=[]
    list_community_leading_eigenvector2=[]
    list_community_label_propagation=[]
    list_community_label_propagation1=[]
    list_community_label_propagation2=[]
    list_community_multilevel=[]
    list_community_multilevel1=[]
    list_community_multilevel2=[]
    list_community_optimal_modularity=[]
    list_community_optimal_modularity1=[]
    list_community_optimal_modularity2=[]
    list_community_edge_betwennes=[]
    list_community_edge_betwennes1=[]
    list_community_edge_betwennes2=[]
    list_community_edge_betwennes3=[]
    list_community_spinglass=[]
    list_community_spinglass1=[]
    list_community_spinglass2=[]
    list_community_walktrap=[]
    list_community_walktrap1=[]
    list_community_walktrap2=[]
    list_community_walktrap3=[]
    list_density=[]
    list_diversity=[]
    list_eccentricity=[]
    list_edge_connectivity=[]
    list_reciprocity=[]
    list_modularity=[]
    list_label=[]
    list_final=[]
    for v in list_graph:
        list_assortativity.append(assortativity(v))
        list_average_path.append(average_path_length(v))
        list_betweenness.append(betweenness(v))
        list_closeness.append(closeness(v))
        list_diameter.append(diameter(v))
        list_eigenvector.append(eigenvector(v))
        list_hub_score.append(hub_score(v))
   
        list_knn.append(knn(v))
   
        list_transitivity.append(transitivity(v))
        list_pagerank.append(pagerank(v))
        list_spaning.append(tree_spaning(v))
        list_community_fatgreedy.append(community_fastgreedy(v))
        list_community_infomap.append(community_infomap(v))
        list_community_leading_eigenvector.append(community_leading_eigenvector(v))
        list_mean_distribution_degree.append(mean_degree(v))
        list_second_moment_degree.append(second_moment(v))
        list_entropy_degree.append(entropy_degree_sequence(v))
        list_complexidade.append(complexidade(v))
        list_kcore.append(kcore(v))
        list_effiency.append(nodal_eff(v))
        list_community_label_propagation.append(community_label_propagation(v))
        list_community_multilevel.append(community_multilevel(v))
        list_community_edge_betwennes.append(community_edge_betweenness(v))
        list_community_spinglass.append(community_spinglass(v))
   
        list_diversity.append(diversity(v))
        list_eccentricity.append(eccentricity(v))
        list_density.append(GraphBase.density(v, loops=False))
        list_edge_connectivity.append(edge_conectivity(v))
        list_reciprocity.append(reciprocity(v))

    for i in list_spaning:
        list_spaning1.append(average_path_length(i))

    for i in list_community_fatgreedy:
        list_community_fatgreedy1.append(VertexClustering.giant(i))
    for i in list_community_fatgreedy1:
        list_community_fatgreedy2.append(average_path_length(i))

    for i in list_community_infomap:
        list_community_infomap1.append(VertexClustering.giant(i))
    for i in list_community_infomap1:
        list_community_infomap2.append(average_path_length(i))

    for i in list_community_leading_eigenvector:
        list_community_leading_eigenvector1.append(VertexClustering.giant(i))
    for i in list_community_leading_eigenvector1:
        list_community_leading_eigenvector2.append(average_path_length(i))

    for i in list_community_multilevel:
        list_community_multilevel1.append(VertexClustering.giant(i))
    for i in list_community_multilevel1:
        list_community_multilevel2.append(average_path_length(i))

    for i in list_community_edge_betwennes:
        list_community_edge_betwennes1.append(VertexDendrogram.as_clustering(i))
    for i in list_community_edge_betwennes1:
        list_community_edge_betwennes2.append(VertexClustering.giant(i))
    for i in list_community_edge_betwennes2:
        list_community_edge_betwennes3.append(average_path_length(i))

    for i in list_community_spinglass:
        list_community_spinglass1.append(VertexClustering.giant(i))
    for i in list_community_spinglass1:
        list_community_spinglass2.append(average_path_length(i))



    for i in list_community_label_propagation:
        list_community_label_propagation1.append(VertexClustering.giant(i))
    for i in list_community_label_propagation1:
        list_community_label_propagation2.append(average_path_length(i))

    list_final = [list_assortativity, list_average_path, list_betweenness, list_closeness, list_diameter,list_eigenvector, list_hub_score, list_knn, list_transitivity, list_pagerank, list_spaning1,list_community_fatgreedy2, list_community_infomap2, list_community_leading_eigenvector2,list_mean_distribution_degree, list_second_moment_degree, list_entropy_degree, list_complexidade,list_kcore, list_effiency, list_community_multilevel2, list_community_label_propagation2,list_community_edge_betwennes3, list_community_spinglass2, list_diversity,list_eccentricity, list_density, list_reciprocity]# print list_spaning1
   # df = DataFrame(list_final)
    return list_final


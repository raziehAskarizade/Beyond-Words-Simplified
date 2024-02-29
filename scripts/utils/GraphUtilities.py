import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

def extract_sub_graph(graph_handler):
    num_sub_graph_edges = 100
    random_indices = torch.randint(0, graph_handler.edge_count, (num_sub_graph_edges,))
    edges: np.array = graph_handler.get_edges(random_indices).numpy()
    node_indices = np.unique(np.reshape(edges, -1))
    node_map = pd.DataFrame(data=list(range(len(node_indices))), index=node_indices, dtype=int)
    nodes = torch.Tensor(graph_handler.get_nodes(node_indices))
    edges[0] = np.squeeze(node_map.loc[edges[0]])
    edges[1] = np.squeeze(node_map.loc[edges[1]])
    edges = torch.Tensor(edges).int()
    return nodes, edges, node_indices, node_map

def reweight_hetero_graph( graph : HeteroData , triplet : tuple , weight):
    if not graph[triplet[0],triplet[1],triplet[2]]:
        return graph
    list_of_weights = graph[triplet[0],triplet[1],triplet[2]].edge_attr.tolist()
    for i in range(len(list_of_weights)):
        list_of_weights[i] = weight
    graph[triplet[0],triplet[1],triplet[2]].edge_attr = torch.tensor(list_of_weights, dtype=torch.float32)
    return graph
    
    
    

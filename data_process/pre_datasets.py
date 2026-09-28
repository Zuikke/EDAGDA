import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import random
import scipy.sparse as sp
import os
import sys
from torch_geometric.io import read_txt_array
import pandas as pd

sys.path.append('..')
from data_process import pre_arxiv

def edges_to_adj(edges, num_node):
    edge_source = [int(i[0]) for i in edges]
    edge_target = [int(i[1]) for i in edges]
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index

def edgeidx_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    return adj

def CSBM(setting, SIGMA):
    d = 3
    num_nodes = 6000
    MU = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if setting == "cond1":
        py = [1/3, 1/3, 1/3]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
    elif setting == "cond2":
        py = [1/3, 1/3, 1/3]
        B = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    elif setting == "card1":
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "card2":
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
        B = [[i/4 for i in row] for row in B]
    elif setting == "css1":
        py = [1/3, 1/3, 1/3]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "css2":
        py = [1/3, 1/3, 1/3]
        B = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "gss1":
        py = [1/2, 1/4, 1/4]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "gss2":
        py = [0.1, 0.3, 0.6]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    else:
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
    
    N = [int(num_nodes * i) for i in py]
    B = [[i*0.1 for i in row] for row in B]

    G = nx.stochastic_block_model(N, B)
    edge_list = list(G.edges)
    
    MU_0 = MU[0]
    MU_1 = MU[1]
    MU_2 = MU[2]
    C0 = np.random.multivariate_normal(mean=MU_0, cov=np.eye(d) * SIGMA**2, size=N[0])
    C1 = np.random.multivariate_normal(mean=MU_1, cov=np.eye(d) * SIGMA**2, size=N[1])
    C2 = np.random.multivariate_normal(mean=MU_2, cov=np.eye(d) * SIGMA**2, size=N[2])

    num_nodes = np.sum(N)
    print(num_nodes)
    node_idx = np.arange(num_nodes)
    features = np.zeros((num_nodes, C1.shape[1]))
    label = np.zeros((num_nodes))

    c0_idx = node_idx[list(G.graph['partition'][0])]
    c1_idx = node_idx[list(G.graph['partition'][1])]
    c2_idx = node_idx[list(G.graph['partition'][2])]

    features[c0_idx] = C0
    features[c1_idx] = C1
    features[c2_idx] = C2

    label[c1_idx] = 1
    label[c2_idx] = 2

    random.shuffle(c0_idx)
    random.shuffle(c1_idx)
    random.shuffle(c2_idx)

    features = torch.FloatTensor(features)
    label = torch.LongTensor(label)
    idx_source_train = np.concatenate((c0_idx[:int(0.6 * len(c0_idx))],
                                 c1_idx[:int(0.6 * len(c1_idx))], c2_idx[:int(0.6 * len(c2_idx))]))
    idx_source_valid = np.concatenate((c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
                                      c1_idx[int(0.6 * len(c1_idx)) : int(0.8 * len(c1_idx))], c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]))
    idx_source_test = np.concatenate((c0_idx[int(0.8 * len(c0_idx)):],
                                c1_idx[int(0.8 * len(c1_idx)):], c2_idx[int(0.8 * len(c2_idx)):]))
    idx_target_valid = np.concatenate((c0_idx[:int(0.2 * len(c0_idx))],
                                       c1_idx[:int(0.2 * len(c1_idx))], c2_idx[:int(0.2 * len(c2_idx))]))
    idx_target_test = np.concatenate((c0_idx[int(0.2 * len(c0_idx)):],
                                c1_idx[int(0.2 * len(c1_idx)):], c2_idx[int(0.2 * len(c2_idx)):]))
    num_nodes = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)

    graph.adj = adj
    graph.num_classes = 3
    graph.edge_weight = torch.ones(graph.num_edges)
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class
    print("done")
    return graph

def prepare_dblp_acm(raw_dir, name):
    docs_path = os.path.join(raw_dir, name, 'raw/{}_docs.txt'.format(name))
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    num_node = x.size(0)
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)
    y = torch.from_numpy(y).to(torch.int64)
    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = y
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class

    return graph


def load_struc2vec_embedding(emb_path, node_order):
    """
    Load struc2vec .emb file and return feature matrix aligned with node_order.

    Args:
        emb_path (str): Path to .emb file (e.g., "usa-airports.emb")
        node_order (list): List of node names in the order used for y and edge_index.

    Returns:
        torch.Tensor: [N, D] feature matrix
    """
    # Step 1: Read embedding file
    emb_dict = {}
    with open(emb_path, 'r') as f:
        header = f.readline().strip().split()
        num_nodes_file, dim = int(header[0]), int(header[1])

        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            node_id = parts[0]
            emb = list(map(float, parts[1:]))
            emb_dict[node_id] = emb

    # Step 2: Align with node_order
    assert len(node_order) == num_nodes_file, "Node count mismatch!"
    x_list = []
    for node in node_order:
        if node not in emb_dict:
            raise KeyError(f"Node {node} not found in embedding file!")
        x_list.append(emb_dict[node])

    x = torch.tensor(x_list, dtype=torch.float)  # [N, D]
    return x


def prepare_airports(data_dir="./data_files/airports", country="usa"):
    """
    Load Airports dataset with struc2vec embeddings as node features.
    """
    # 1. Load graph and labels (same as before)
    edge_path = os.path.join(data_dir, f"{country}-airports.edgelist")
    label_path = os.path.join(data_dir, f"labels-{country}-airports.txt")

    G = nx.read_edgelist(edge_path, nodetype=str)
    nodes = list(G.nodes())

    # Load labels
    node_to_label = {}
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node, label = parts
                node_to_label[node] = int(label)
    assert all(n in node_to_label for n in nodes)
    y = torch.tensor([node_to_label[n] for n in nodes], dtype=torch.long)

    # 2. Build edge_index
    edges = []
    for u, v in G.edges():
        i, j = nodes.index(u), nodes.index(v)  # 或用 dict 加速
        edges.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 3. Load struc2vec embeddings as x
    emb_file = os.path.join(data_dir, f"{country}.emb")
    x = load_struc2vec_embedding(emb_file, nodes)

    # 4. Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = int(y.max().item()) + 1
    data.node_names = nodes
    data.edge_weight = torch.ones(edge_index.shape[1])
    return data



def load_twitch(root_path, country) -> Data:
    target_path = os.path.join(root_path, country, f"{country}_target.csv")
    embedding_path = os.path.join(root_path, country, f"{country}_embedding.csv")
    edges_path = os.path.join(root_path, country, f"{country}_edges.csv")
    source_split_path = os.path.join(root_path, country, f"{country}_source_split.csv")
    target_split_path = os.path.join(root_path, country, f"{country}_target_split.csv")




    # 1. Load target file
    target_df = pd.read_csv(target_path)
    target_df = target_df.sort_values('new_id').reset_index(drop=True)

    # Extract labels
    y = torch.tensor(target_df['mature'].values, dtype=torch.long)  # [N]
    num_nodes = len(target_df)

    # 2. Load embeddings
    emb_df = pd.read_csv(embedding_path)

    emb_df = emb_df.sort_values('id')

    emb_df = emb_df.sort_values('id').reset_index(drop=True)
    feature_cols = [col for col in emb_df.columns if col.startswith('x_')]
    assert len(feature_cols) == 256
    x = torch.tensor(emb_df[feature_cols].values, dtype=torch.float)  # [N, 256]

    # 3. Load edges — NO MAPPING NEEDED!
    edges_df = pd.read_csv(edges_path)
    # Since 'from' and 'to' are already new_id, use them directly
    edge_index = torch.tensor(edges_df[['from', 'to']].values.T, dtype=torch.long)  # [2, E]

    # Make graph undirected (optional but common)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Build PyG Data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = 2
    data.num_nodes = num_nodes

    data.edge_weight = torch.ones(edge_index.shape[1])

    source_training_mask, source_validation_mask, source_testing_mask = twitch_load_source_split_masks(source_split_path)
    target_validation_mask, target_testing_mask = twitch_load_target_split_masks(target_split_path)

    data.source_training_mask = source_training_mask
    data.source_validation_mask = source_validation_mask
    data.source_testing_mask = source_testing_mask

    data.target_validation_mask = target_validation_mask
    data.target_testing_mask = target_testing_mask

    return data





def twitch_load_source_split_masks(csv_path):
    df = pd.read_csv(csv_path)

    if not {'idx', 'split'}.issubset(df.columns):
        raise ValueError("CSV without 'idx' and 'split'")

    train_idx = df[df['split'] == 'train']['idx'].values
    val_idx = df[df['split'] == 'val']['idx'].values
    test_idx = df[df['split'] == 'test']['idx'].values

    train_tensor = torch.from_numpy(train_idx).long()
    val_tensor = torch.from_numpy(val_idx).long()
    test_tensor = torch.from_numpy(test_idx).long()

    return train_tensor, val_tensor, test_tensor

def twitch_load_target_split_masks(csv_path):

    df = pd.read_csv(csv_path)


    valid_splits = {'val', 'test'}
    actual_splits = set(df['split'].unique())
    if not actual_splits.issubset(valid_splits):
        raise ValueError(f"split invaild value")

    val_idx = df[df['split'] == 'val']['idx'].values
    test_idx = df[df['split'] == 'test']['idx'].values

    val_tensor = torch.from_numpy(val_idx).long()
    test_tensor = torch.from_numpy(test_idx).long()

    return val_tensor, test_tensor




def prepare_arxiv(root, years):
    dataset = pre_arxiv.load_nc_dataset(root, 'ogb-arxiv', years)
    idx = (dataset.test_mask == True).nonzero().view(-1).numpy()
    np.random.shuffle(idx)
    num_training = idx.shape[0]
    adj = edgeidx_to_adj(dataset.graph['edge_index'][0], dataset.graph['edge_index'][1], dataset.graph['num_nodes'])
    edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph = Data(edge_index=edge_index, x=dataset.graph['node_feat'], y=dataset.label.view(-1))
    graph.adj = adj
    graph.source_training_mask = idx[0:int(0.6*num_training)]
    graph.source_validation_mask = idx[int(0.6*num_training):int(0.8*num_training)]
    graph.source_testing_mask = idx[int(0.8*num_training):]
    graph.target_validation_mask = idx[0:int(0.2*num_training)]
    graph.target_testing_mask = idx[int(0.2*num_training):]
    graph.source_mask = idx
    graph.target_mask = idx
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.num_classes = dataset.num_classes
    if torch.unique(graph.y).size(0) < graph.num_classes:
        print("miss classes")
        #return 
    graph.y_hat = graph.y 
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class
    return graph

def prepare_MAG(dir, name):
    graph = torch.load(os.path.join(dir, '{}_labels_20.pt'.format(name)))
    adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
    graph.adj = adj
    graph.edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph.num_classes = torch.max(graph.y) + 1
    
    idx = np.arange(graph.num_nodes)
    np.random.shuffle(idx)
    idx_len = idx.shape[0]
    graph.source_training_mask = idx[0:int(0.6*idx_len)]
    graph.source_validation_mask = idx[int(0.6*idx_len):int(0.8*idx_len)]
    graph.source_testing_mask = idx[int(0.8*idx_len):]
    graph.target_validation_mask = idx[0:int(0.2*idx_len)]
    graph.target_testing_mask = idx[int(0.2*idx_len):]
    graph.source_mask = idx
    graph.target_mask = idx

    graph.edge_weight = torch.ones(graph.num_edges)

    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class

    return graph


def random_mask(num_nodes, ratios, seed=None):
    """
    Generate random boolean masks for train/val/test.

    Args:
        num_nodes (int): Total number of nodes
        ratios (list): [train_ratio, val_ratio, test_ratio] or [val_ratio, test_ratio]
        seed (int, optional): Random seed for reproducibility

    Returns:
        tuple of torch.BoolTensor masks
    """
    if seed is not None:
        torch.manual_seed(seed)

    indices = torch.randperm(num_nodes)
    total = len(indices)

    if len(ratios) == 3:
        # Source: train / val / test
        train_end = int(total * ratios[0])
        val_end = train_end + int(total * ratios[1])
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = torch.zeros(total, dtype=torch.bool)
        val_mask = torch.zeros(total, dtype=torch.bool)
        test_mask = torch.zeros(total, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        return train_mask, val_mask, test_mask

    elif len(ratios) == 2:
        # Target: val / test (no train)
        val_end = int(total * ratios[0])
        val_idx = indices[:val_end]
        test_idx = indices[val_end:]

        val_mask = torch.zeros(total, dtype=torch.bool)
        test_mask = torch.zeros(total, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        return val_mask, test_mask
    else:
        raise ValueError("ratios must be length 2 or 3")
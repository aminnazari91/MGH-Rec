import os
import random
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import dgl
import torch
import pandas as pd

from collections import defaultdict


def set_seed(seed: int = 42):
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DGL
    dgl.random.seed(seed)

    print(f"[INFO] Global seed set to {seed}")

def load_graph(fullpath, train_ratio=0.6, val_ratio=0.1, seed=42):
    with open(fullpath, "rb") as f:
        data = pickle.load(f)
    g = data["graph"]
    g = g.to('cpu')
    labels = g.nodes['movie'].data['label']
    target = 'movie'
    
    num_classes = torch.unique(labels).shape[0]
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)

    for etype in g.etypes:
        edge_dict[etype] = len(edge_dict)
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

    idx = np.arange(len(labels))

    # Train vs (Val + Test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        idx,
        labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed
    )

    # Val vs Test
    val_size = val_ratio / (1 - train_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=seed
    )

    num_nodes = g.num_nodes('movie')

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    g.nodes['movie'].data['train_mask'] = train_mask
    g.nodes['movie'].data['val_mask']   = val_mask
    g.nodes['movie'].data['test_mask']  = test_mask
    
    return g, node_dict, edge_dict, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target

def load_rating(fullpath, train_idx, val_idx, test_idx):
    ratings = pd.read_csv(fullpath, index_col=0)
    train_ratings = ratings.loc[ratings['movie_node_id'].isin(train_idx)]
    val_ratings = ratings.loc[ratings['movie_node_id'].isin(val_idx)]
    test_ratings = ratings.loc[ratings['movie_node_id'].isin(test_idx)]

    return ratings, train_ratings, val_ratings, test_ratings


def build_user_pos_dict(
    g,
    user_ntype='user',
    movie_ntype='movie',
    rating_etypes=('rate_4', 'rate_5')
):

    user_pos = defaultdict(set)

    for etype in rating_etypes:

        if (user_ntype, etype, movie_ntype) not in g.canonical_etypes:
            continue

        src, dst = g.edges(etype=(user_ntype, etype, movie_ntype))

        for u, m in zip(src.tolist(), dst.tolist()):
            user_pos[u].add(m)

    return {u: list(movies) for u, movies in user_pos.items()}

def build_train_graph(g, train_idx, movie_ntype='movie'):

    train_g = g.clone()

    movie_mask = torch.zeros(g.num_nodes(movie_ntype), dtype=torch.bool)
    movie_mask[train_idx] = True

    for canonical_etype in g.canonical_etypes:

        src_type, etype, dst_type = canonical_etype

        # فقط edge هایی که به movie می‌روند
        if dst_type != movie_ntype:
            continue

        src, dst = g.edges(etype=canonical_etype)

        keep = movie_mask[dst]

        remove_eids = (~keep).nonzero(as_tuple=True)[0]

        if len(remove_eids) > 0:
            train_g.remove_edges(remove_eids, etype=canonical_etype)

    return train_g

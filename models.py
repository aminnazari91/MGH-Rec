import os
import random

import numpy as np
import pickle
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn import GraphConv, HeteroGraphConv, GATConv, GATv2Conv

import dgl.function as fn
from dgl.nn.functional import edge_softmax

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import utils


def init_feat(G, n_inp, features):
    input_dims = {}
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_inp), requires_grad=True)
        nn.init.xavier_uniform_(emb)
        feats = features.get(ntype, emb)
        G.nodes[ntype].data['x'] = feats
        input_dims[ntype] = feats.shape[1]
    return G, input_dims


########## Baselines ##########
class GCNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        node_dict,
        g,
        num_heads=2,
        dropout=0.5
    ):
        super().__init__()
        assert output_dim % num_heads == 0

        self.node_dict = node_dict
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # ===== Heterogeneous GCN =====
        self.gcn = HeteroGraphConv(
            {
                etype: GraphConv(hidden_dim, output_dim)
                for etype in g.canonical_etypes
            },
            aggregate='sum'
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        """
        g: DGLHeteroGraph
        h: dict {ntype: [N, hidden_dim]}
        """

        # ----- GCN message passing -----
        h_new = self.gcn(g, h)

        # ----- activation + dropout -----
        h_out = {}
        for ntype, feat in h_new.items():
            feat = F.relu(feat)
            feat = self.dropout(feat)
            h_out[ntype] = feat

        return h_out

class GCN_MTL(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        input_dims,
        hidden_dim,
        num_classes,
        num_layers=2,
        num_heads=2
    ):
        super().__init__()
        self.node_dict = node_dict

        # ===== Input Projection =====
        self.pre = nn.ModuleList([
            nn.Linear(input_dims[nt], hidden_dim)
            for nt in node_dict
        ])

        # ===== Layers =====
        self.layers = nn.ModuleList([
            GCNLayer(
                hidden_dim,
                hidden_dim,
                node_dict,
                G,
                num_heads
            )
            for _ in range(num_layers)
        ])


        # ===== Classification Head =====
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # ===== Rating Head (GMF + MLP style) =====
        # input: user_emb + movie_emb + dot_product
        self.rating_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),   # +1 for dot product
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 
        )


    def predict_rating(self, user_emb, movie_emb):
        dot = (user_emb * movie_emb).sum(dim=1, keepdim=True)  # GMF component
        concat = torch.cat([user_emb, movie_emb, dot], dim=1)
        return self.rating_head(concat).squeeze(-1)

    def forward(self, G, return_emb=True):
        device = next(self.parameters()).device

        h = {}
        for nt in G.ntypes:
            idx = self.node_dict[nt]
            x = G.nodes[nt].data['x'].to(device)
            h[nt] = F.gelu(self.pre[idx](x))

        for layer in self.layers:
            h = layer(G, h)

        logits = self.cls_head(h['movie'])

        if return_emb:
            return logits, h

        return logits
    
# ===== GAT Layer (multi-head, hetero) =====
class GATLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim, node_dict, g, num_heads=4, dropout=0.2, use_v2=False):
        super().__init__()
        assert output_dim % num_heads == 0

        self.node_dict = node_dict
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_v2 = use_v2

        conv_cls = GATv2Conv if use_v2 else GATConv
        self.gat = HeteroGraphConv(
            {etype: conv_cls(hidden_dim, self.head_dim, num_heads, allow_zero_in_degree=True)
             for etype in g.etypes},
            aggregate='sum'
        )

    def forward(self, g, h):
        h_new = self.gat(g, h)

        # flatten / squeeze + activation + dropout
        h_out = {}
        for ntype, feat in h_new.items():
            if feat.dim() == 3:
                if feat.shape[2] == 1:
                    feat = feat.squeeze(2)
                else:
                    feat = feat.flatten(1)
            h_out[ntype] = self.dropout(F.elu(feat))
        return h_out

# ===== GAT Multi-task Model =====
class GAT_MTL(nn.Module):
    def __init__(self, G, node_dict, input_dims, hidden_dim, num_classes,
                 num_layers=2, num_heads=4, dropout=0.2, use_v2=False):
        super().__init__()
        self.node_dict = node_dict
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ===== Input projection =====
        self.pre = nn.ModuleList([
            nn.Linear(input_dims[nt], hidden_dim) for nt in node_dict
        ])

        # ===== GAT / GATv2 layers =====
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, node_dict, G, num_heads, dropout, use_v2)
            for _ in range(num_layers)
        ])

        # ===== Classification head =====
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # ===== Rating head (GMF + MLP style) =====
        self.rating_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def predict_rating(self, user_emb, movie_emb):
        dot = (user_emb * movie_emb).sum(dim=1, keepdim=True)
        concat = torch.cat([user_emb, movie_emb, dot], dim=1)
        return self.rating_head(concat).squeeze(-1)

    def forward(self, G, return_emb=True):
        device = next(self.parameters()).device

        # ===== Input projection =====
        h = {}
        for nt in G.ntypes:
            idx = self.node_dict[nt]
            x = G.nodes[nt].data['x'].to(device)
            h[nt] = F.gelu(self.pre[idx](x))

        # ===== GAT layers =====
        for layer in self.layers:
            h = layer(G, h)

        # ===== Classification =====
        logits = self.cls_head(h['movie'])

        if return_emb:
            return logits, h
        return logits

########### HGT ############
class HGT_MTL(nn.Module):
    def __init__(
        self,
        g,
        node_dict,
        input_dims,
        hidden_dim,
        num_classes,
        num_layers=2,          
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()

        self.ntypes = g.ntypes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ===== Input projection =====
        self.pre = nn.ModuleDict({
            nt: nn.Linear(input_dims[nt], hidden_dim)
            for nt in node_dict
        })

        # ===== homogeneous graph =====
        self.homo_g = dgl.to_homogeneous(g)
        self.num_ntypes = len(g.ntypes)
        self.num_etypes = len(g.etypes)

        # ===== HGT layers =====
        self.hgt_layers = nn.ModuleList([
            dglnn.HGTConv(
                in_size=hidden_dim,
                head_size=hidden_dim // num_heads,
                num_heads=num_heads,
                num_ntypes=self.num_ntypes,
                num_etypes=self.num_etypes,
                dropout=dropout,
                use_norm=True
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # ===== Classification head =====
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # ===== Rating head =====
        self.rating_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, g):
        device = next(self.parameters()).device
        self.homo_g = self.homo_g.to(device)

        # ===== input projection =====
        feat_dict = {
            nt: self.pre[nt](g.nodes[nt].data['x'])
            for nt in self.pre
        }

        # ===== homogeneous feature tensor =====
        node_types = self.homo_g.ndata[dgl.NTYPE]
        node_ids = self.homo_g.ndata[dgl.NID]

        h = torch.zeros(
            self.homo_g.num_nodes(),
            self.hidden_dim,
            device=device
        )

        for tid, ntype in enumerate(self.ntypes):
            idx = (node_types == tid).nonzero(as_tuple=True)[0]
            h[idx] = feat_dict[ntype][node_ids[idx]]

        ntype = self.homo_g.ndata[dgl.NTYPE].to(device)
        etype = self.homo_g.edata[dgl.ETYPE].to(device)

        # ===== HGT encoder  =====
        for layer in self.hgt_layers:
            h = layer(self.homo_g, h, ntype, etype)
            h = F.elu(h)

        h_drop = self.dropout(h)

        # ===== split embeddings =====
        emb = {}
        for tid, ntype_name in enumerate(self.ntypes):
            idx = (node_types == tid).nonzero(as_tuple=True)[0]
            emb[ntype_name] = h[idx]

        # ===== heads =====
        movie_tid = self.ntypes.index('movie')
        movie_idx = (node_types == movie_tid).nonzero(as_tuple=True)[0]

        movie_emb = emb['movie']
        logits = self.cls_head(h_drop[movie_idx])

        return logits, emb
    
    def predict_rating(self, user_emb, movie_emb):
        dot = (user_emb * movie_emb).sum(dim=1, keepdim=True)  # GMF component
        concat = torch.cat([user_emb, movie_emb, dot], dim=1)
        return self.rating_head(concat).squeeze(-1)

# RGAT Multi-Task Learning
class RGATConvHeteroMTL(nn.Module):
    def __init__(self, in_dim, out_dim, canonical_etypes, num_heads=4, dropout=0.0, aggregate='sum'):
        super().__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.aggregate = aggregate

        self.W_q = nn.ModuleDict()
        self.W_k = nn.ModuleDict()
        self.W_v = nn.ModuleDict()

        for srctype, etype, dsttype in canonical_etypes:
            key = f"{srctype}_{etype}_{dsttype}"
            self.W_q[key] = nn.Linear(in_dim, out_dim, bias=False)
            self.W_k[key] = nn.Linear(in_dim, out_dim, bias=False)
            self.W_v[key] = nn.Linear(in_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h_dict):
        device = next(self.parameters()).device
        out_dict = {ntype: [] for ntype in g.ntypes}
        for srctype, etype, dsttype in g.canonical_etypes:
            key = f"{srctype}_{etype}_{dsttype}"
            subg = g[(srctype, etype, dsttype)]
            h_src = h_dict[srctype]
            h_dst = h_dict[dsttype]

            Q = self.W_q[key](h_dst).view(-1, self.num_heads, self.d_k)
            K = self.W_k[key](h_src).view(-1, self.num_heads, self.d_k)
            V = self.W_v[key](h_src).view(-1, self.num_heads, self.d_k)

            subg.srcdata['K'] = K
            subg.srcdata['V'] = V
            subg.dstdata['Q'] = Q

            subg.apply_edges(lambda edges: {'e': (edges.dst['Q'] * edges.src['K']).sum(-1) / (self.d_k ** 0.5)})
            subg.edata['a'] = dgl.nn.functional.edge_softmax(subg, subg.edata['e'])
            subg.edata['a'] = self.dropout(subg.edata['a'])

            subg.update_all(lambda e: {'m': e.data['a'].unsqueeze(-1) * e.src['V']}, dgl.function.sum('m', 'h_out'))
            out = subg.dstdata['h_out'].reshape(-1, self.out_dim)
            out_dict[dsttype].append(out)

        final_out = {}
        for ntype, outs in out_dict.items():
            if len(outs) == 0:
                continue
            if self.aggregate == 'sum':
                final_out[ntype] = torch.stack(outs).sum(0)
            else:
                final_out[ntype] = torch.stack(outs).mean(0)
        return final_out

class RGAT_MTL(nn.Module):
    def __init__(self, g, node_dict, input_dims, hidden_dim, num_classes, num_heads=4, dropout=0.2):
        super().__init__()
        self.ntypes = g.ntypes
        self.hidden_dim = hidden_dim

        # ===== input projection =====
        self.pre = nn.ModuleDict({nt: nn.Linear(input_dims[nt], hidden_dim) for nt in node_dict})

        # ===== RGAT layers =====
        self.layer1 = RGATConvHeteroMTL(hidden_dim, hidden_dim, g.canonical_etypes, num_heads=num_heads, dropout=dropout)
        self.layer2 = RGATConvHeteroMTL(hidden_dim, hidden_dim, g.canonical_etypes, num_heads=1, dropout=0.0)

        self.dropout = nn.Dropout(dropout)

        # ===== classification head =====
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # ===== rating head =====
        self.rating_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, g):
        device = next(self.parameters()).device
        h = {nt: self.pre[nt](g.nodes[nt].data['x'].to(device)) for nt in self.pre}

        h = self.layer1(g, h)
        h = {nt: F.elu(v) for nt, v in h.items()}
        h = self.layer2(g, h)
        h = {nt: F.elu(v) for nt, v in h.items()}

        h_drop = {nt: self.dropout(v) for nt, v in h.items()}
        logits = self.cls_head(h_drop['movie'])

        return logits, h

    def predict_rating(self, user_emb, movie_emb):
        dot = (user_emb * movie_emb).sum(dim=1, keepdim=True)
        concat = torch.cat([user_emb, movie_emb, dot], dim=1)
        return self.rating_head(concat).squeeze(-1)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # q, kv: [N, D]
        q_ = q.unsqueeze(1)
        kv_ = kv.unsqueeze(1)
        out, _ = self.attn(q_, kv_, kv_)
        out = out.squeeze(1)
        return self.norm(out + q)


class DualBranchCrossAttnModel(nn.Module):
    def __init__(
        self,
        model_gcn,
        model_hgt,
        hidden_dim,
        num_classes,
        alpha_cls=0.9,   #  HGT  classification w
        beta_rec=0.9     #  GCN  recommendation w
    ):
        super().__init__()

        self.branch_gcn = model_gcn
        self.branch_hgt = model_hgt

        # disable old heads
        for name in ['cls_head', 'cluster_head', 'rating_head']:
            setattr(self.branch_gcn, name, Identity())
            setattr(self.branch_hgt, name, Identity())

        # cross attention
        self.cross_cls = CrossAttention(hidden_dim)
        self.cross_rec = CrossAttention(hidden_dim)

        # manual weights
        self.alpha_cls = alpha_cls
        self.beta_rec = beta_rec

        # classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # rating head
        self.rating_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, g):
        _, h_gcn = self.branch_gcn(g)
        _, h_hgt = self.branch_hgt(g)
        return h_gcn, h_hgt

    # ---------- classification (HGT dominant) ----------
    def classify(self, h_gcn, h_hgt):
        movie_hgt = h_hgt['movie']
        movie_gcn = h_gcn['movie']

        cross = self.cross_cls(movie_hgt, movie_gcn)

        movie_final = (
            self.alpha_cls * movie_hgt +
            (1.0 - self.alpha_cls) * cross
        )

        return self.cls_head(movie_final)

    # ---------- recommendation (GCN dominant) ----------
    def predict_rating(self, h_gcn, h_hgt, u_idx, m_idx):
        user_gcn = h_gcn['user'][u_idx]
        movie_gcn = h_gcn['movie'][m_idx]

        user_hgt = h_hgt['user'][u_idx]
        movie_hgt = h_hgt['movie'][m_idx]

        user_cross = self.cross_rec(user_gcn, user_hgt)
        movie_cross = self.cross_rec(movie_gcn, movie_hgt)

        user_final = (
            self.beta_rec * user_gcn +
            (1.0 - self.beta_rec) * user_cross
        )
        movie_final = (
            self.beta_rec * movie_gcn +
            (1.0 - self.beta_rec) * movie_cross
        )

        dot = (user_final * movie_final).sum(dim=1, keepdim=True)
        concat = torch.cat([user_final, movie_final, dot], dim=1)

        return self.rating_head(concat).squeeze(-1)

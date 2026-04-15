import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
import constants
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
import numpy as np

# code adapted from dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py 
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, sem_att_size, layer_num_heads, dropout, 
                    add_self_embeddings=True,
                    add_self_loops = False):
        super().__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv( 
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.relu,
                    allow_zero_in_degree=True,
                    bias=False
                )
            )
        
        self.p_semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads,
            hidden_size=sem_att_size
        )
        self.g_semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads,
            hidden_size=sem_att_size
        )
        # This layer will drop entire metapaths (the 'M' dimension)
        # by treating it as a "channel" dimension.
        self.metapath_dropout = nn.Dropout2d(p=dropout)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        
        self.WG = nn.Linear(in_size, out_size*layer_num_heads)
        self.WP = nn.Linear(in_size, out_size*layer_num_heads)

        self.add_self_embeddings = add_self_embeddings
        self.add_self_loops = add_self_loops

    def forward(self, g, ha, ht):
        p_semantic_embeddings = []
        g_semantic_embeddings = []

        for i, meta_path in enumerate(self.meta_paths):
            new_g = dgl.metapath_reachable_graph(g, meta_path)

            if meta_path[0] in constants.FROM_AMP and meta_path[-1] in constants.TO_TARGET:
                g_semantic_embeddings.append(
                    self.gat_layers[i](
                        new_g, (ha, ht)
                    ).flatten(1)
                )
            elif meta_path[0] in constants.FROM_TARGET and meta_path[-1] in constants.TO_AMP:
                p_semantic_embeddings.append(
                    self.gat_layers[i](
                        new_g, (ht, ha)
                    ).flatten(1)
                )
            elif meta_path[0] in constants.FROM_AMP and meta_path[-1] in constants.TO_AMP:
                new_g = new_g.remove_self_loop()
                if self.add_self_loops:
                    new_g = dgl.add_self_loop(new_g)
                    
                p_semantic_embeddings.append(
                    self.gat_layers[i](
                        new_g, ha
                    ).flatten(1)
                )
            else: # Target --> Target
                new_g = new_g.remove_self_loop()
                if self.add_self_loops:
                    new_g = dgl.add_self_loop(new_g)

                g_semantic_embeddings.append(
                    self.gat_layers[i](
                        new_g, ht
                    ).flatten(1)
                )

        if self.add_self_embeddings:
            g_semantic_embeddings.append(self.WG(ht))
            p_semantic_embeddings.append(self.WP(ha))

        g_semantic_embeddings = torch.stack(
            g_semantic_embeddings, dim=1
        )  # (N, M+1, D * K)

        p_semantic_embeddings = torch.stack(
            p_semantic_embeddings, dim=1
        )  # (N, M+1, D * K)

        # treat the M (metapath) dimension as the "channel" dimension.
        p_sem_4d = p_semantic_embeddings.unsqueeze(-1)
        g_sem_4d = g_semantic_embeddings.unsqueeze(-1)
        p_sem_dropped = self.metapath_dropout(p_sem_4d)
        g_sem_dropped = self.metapath_dropout(g_sem_4d)
        p_out = p_sem_dropped.squeeze(-1)
        g_out = g_sem_dropped.squeeze(-1)

        return self.p_semantic_attention(p_out),  self.g_semantic_attention(g_out) # (N, D * K) in total

# code adapted from https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html
class HeteroLinkPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['ha'], edges.dst['ht']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, ha, ht, etype):
        with g.local_scope():
            g.nodes["AMP"].data["ha"] = ha
            g.nodes["Target"].data["ht"] = ht
            g.apply_edges(self.apply_edges, etype=etype)
            return g.edges[etype].data['score']


class Projector(nn.Module):
    def __init__(self,
                 proteomic_emb_size,
                 genomic_emb_size,
                 han_in_size):
        
        super().__init__()
        self.W1 = nn.Linear(proteomic_emb_size, han_in_size)
        self.W2 = nn.Linear(genomic_emb_size, han_in_size)

    def forward(self, proteomic_embeddings, genomic_embeddings, activation=False):
        p_emb = self.W1(proteomic_embeddings)
        g_emb = self.W2(genomic_embeddings)

        if activation:
            p_emb = F.relu(p_emb)
            g_emb = F.relu(g_emb)
        
        return p_emb, g_emb

class triAMPh(nn.Module):
    def __init__(
        self, 
        meta_paths, 
        han_in_size, 
        han_hidden_size, 
        han_num_heads, 
        han_dropout,
        sem_att_size = 128,
        p_percentile = 0.1,
        g_percentile = 30,
        add_self_embeddings = True, 
        add_self_loops = False
    ):
        super().__init__()

        self.graph_attention = HANLayer(meta_paths, han_in_size, han_hidden_size, sem_att_size, han_num_heads, han_dropout, add_self_embeddings, add_self_loops)      
        self.predictor = HeteroLinkPredictor(han_hidden_size*han_num_heads)
        self.relu = nn.ReLU()
        self.p_percentile = p_percentile
        self.g_percentile = g_percentile

    def complete_graph(self, g, amp_emb_arr, target_emb_arr, cutoff, cutoff_g, device):
        # AMPs
        if torch.cuda.is_available():
            amp_pdist = pairwise_distances(amp_emb_arr.detach().cpu().numpy(), metric="euclidean", n_jobs = 2)
        else:
            amp_pdist = pairwise_distances(amp_emb_arr.detach().numpy(), metric="euclidean", n_jobs = 2)
        amp_pdist_flat = amp_pdist.flatten()
        amp_cutoff = np.percentile(amp_pdist_flat[amp_pdist_flat!=0], cutoff) # TODO: this is really a good hyperparameter!
        amp_adj = (amp_cutoff >= amp_pdist).astype(int)
        np.fill_diagonal(amp_adj, 0) # no self loops
        amp_adj = csr_matrix(amp_adj)
       
        # Targets
        if torch.cuda.is_available():
            target_pdist = pairwise_distances(target_emb_arr.detach().cpu().numpy(),  metric="euclidean", n_jobs = 8)
        else:
            target_pdist = pairwise_distances(target_emb_arr.detach().numpy(),  metric="euclidean", n_jobs = 8)
        target_pdist_flat = target_pdist.flatten()
        target_cutoff = np.percentile(target_pdist_flat[target_pdist_flat!=0], cutoff_g) # TODO: this is really a good hyperparameter!
        target_adj = (target_cutoff >= target_pdist).astype(int)
        
        np.fill_diagonal(target_adj, 0) # no self loops
        target_adj = csr_matrix(target_adj)

        # remove all the edges for is_similar_p and is_similar_g
        g = dgl.remove_edges(g, torch.arange(start=0, end=g.num_edges('is_similar_p')).to(device), 'is_similar_p')
        g = dgl.remove_edges(g, torch.arange(start=0, end=g.num_edges('is_similar_g')).to(device), 'is_similar_g')
        (u,v)=amp_adj.nonzero()
        g.add_edges(u, v, etype="is_similar_p")
        (u,v)=target_adj.nonzero()

        g.add_edges(u, v, etype="is_similar_g")
        g = g.to(device)
        return g
        

    def forward(self, p_emb, g_emb, g, supervision_g, neg_g, device):
        g = self.complete_graph(g, p_emb, g_emb, self.p_percentile, self.g_percentile, device)
        ha, ht = self.graph_attention(g, p_emb, g_emb)
        
        return self.predictor(supervision_g, ha, ht, "is_active"), self.predictor(neg_g, ha, ht, "is_active")

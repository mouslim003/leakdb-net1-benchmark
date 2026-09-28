"""Lightweight DL architectures for pressure time-series leak detection/localization.

All models consume a windowed pressure tensor (B, T=W, F=n_sensors) and produce
a latent vector, followed by task heads. Architectures are kept small
("lightweight") with comparable parameter budgets for a fair benchmark.

Common heads:
  - detection: binary leak-present (shared)
  - localization: 10-class (9 leak nodes + no-leak) -- trained jointly
"""
import numpy as np
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Compact 1D convolutional encoder for time-series windows."""

    def __init__(self, in_ch, hidden=64, kernel=7, n_blocks=2):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_blocks):
            layers += [
                nn.Conv1d(ch, hidden, kernel, padding=kernel // 2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            ch = hidden
        self.blocks = nn.Sequential(*layers)
        self.repr_dim = hidden  # after global avg pool

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(dim=-1)  # global average pool -> (B, hidden)
        return x


class RNNBase(nn.Module):
    """LSTM or GRU encoder taking the last hidden state."""

    def __init__(self, kind, in_ch, hidden=64, n_layers=1, dropout=0.1):
        super().__init__()
        assert kind in ("lstm", "gru")
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(input_size=in_ch, hidden_size=hidden,
                       num_layers=n_layers, batch_first=True,
                       dropout=dropout if n_layers > 1 else 0.0)
        self.repr_dim = hidden

    def forward(self, x):
        out, _ = self.rnn(x)          # out: (B, T, hidden)
        return out[:, -1]             # last step


class TransformerEnc(nn.Module):
    """Small transformer encoder with learned CLS token + mean over time."""

    def __init__(self, in_ch, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 200, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.repr_dim = d_model

    def forward(self, x):
        T = x.size(1)
        h = self.in_proj(x) + self.pos[:, :T]
        h = self.enc(h)
        return self.norm(h.mean(dim=1))   # mean-pool over time


def _num_params(m):
    return sum(p.numel() for p in m.parameters())


class SetInputEncoder(nn.Module):
    """Permutation-invariant sensor-set input encoder with explicit node identity.

    h(t) = sum_c [ MLP(x(t, c)) + E[node_id(c) ]

    Each channel is a (pressure value, physical node id) pair; the node embedding
    E makes the model aware of WHICH node each channel belongs to, and the
    sum-pool makes the representation invariant to channel order/placement.
    """

    def __init__(self, n_nodes, d_model=64, hidden=64):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, d_model))
        self.node_emb = nn.Embedding(n_nodes, d_model)

    def forward(self, x, node_ids):
        # x: (B, T, k) values; node_ids: (k,) physical node indices
        v = self.value_mlp(x.unsqueeze(-1))            # (B, T, k, d)
        e = self.node_emb(node_ids)                    # (k, d)
        return (v + e).sum(dim=2)                      # (B, T, d)


class GNNLayer(nn.Module):
    """Kipf GCN layer: H' = act( D^-1/2 (A+I) D^-1/2 H W )."""

    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d)

    def forward(self, h, adj_norm):
        return torch.relu(torch.matmul(adj_norm, self.lin(h)))


class GNNLeakModel(nn.Module):
    """Lightweight spatio-temporal GCN over the Net1 junction graph.

    Per timestep: node features = W(pressure) at observed nodes, a learned
    'missing' embedding at unobserved nodes; 2 GCN layers propagate over the
    physical adjacency. Temporal mean-pool per node -> per-node localization
    logits (9 nodes) + global-pooled detection logit + no-leak logit.
    Placement-agnostic by construction: any subset of observed nodes is valid.
    """

    def __init__(self, adj, d=64):
        super().__init__()
        n = adj.shape[0]
        adj = np.asarray(adj, dtype=np.float32)
        ahat = adj + np.eye(n, dtype=np.float32)
        deg = ahat.sum(1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        adj_norm = d_inv_sqrt @ ahat @ d_inv_sqrt
        self.register_buffer("adj_norm", torch.tensor(adj_norm, dtype=torch.float32))
        self.value_lin = nn.Linear(1, d)
        self.missing_emb = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.missing_emb, std=0.02)
        self.g1 = GNNLayer(d, d)
        self.g2 = GNNLayer(d, d)
        self.loc_head = nn.Linear(d, 1)   # shared per-node localization score
        self.glob_head = nn.Linear(d, 1)  # global detection logit
        self.none_head = nn.Linear(d, 1)  # no-leak logit for the loc head
        self.repr_dim = d
        self.n_nodes = n
        self._node_ids = None

    def set_sensors(self, node_ids):
        self._node_ids = torch.tensor(np.asarray(node_ids), dtype=torch.long)

    def param_count(self):
        return _num_params(self)

    def forward(self, x):
        # x: (B, T, k) observed pressures; node_ids: which nodes are observed
        B, T, k = x.shape
        dev = x.device
        node_ids = self._node_ids.to(dev)
        obs = torch.zeros(B * T, self.n_nodes, 1, device=dev)
        feats = torch.zeros(B * T, self.n_nodes, self.g1.lin.in_features, device=dev)
        v = self.value_lin(x.reshape(B * T, k, 1))     # (BT, k, d)
        feats[:, node_ids, :] = v
        miss = self.missing_emb.expand(B * T, self.n_nodes, feats.shape[-1])
        mask = torch.zeros(B * T, self.n_nodes, 1, device=dev)
        mask[:, node_ids, :] = 1.0
        h = feats * mask + miss * (1.0 - mask)
        h = self.g1(h, self.adj_norm)
        h = self.g2(h, self.adj_norm)
        # temporal mean-pool per node -> (B, N, d)
        h = h.reshape(B, T, self.n_nodes, -1).mean(dim=1)
        node_scores = self.loc_head(h).squeeze(-1)     # (B, 9)
        glob = h.mean(dim=1)                           # (B, d)
        det_logit = self.glob_head(glob).squeeze(-1)   # (B,)
        none_logit = self.none_head(glob)              # (B, 1)
        loc_logits = torch.cat([none_logit, node_scores], dim=1)  # (B, 10)
        return det_logit, loc_logits


class SetLeakModel(nn.Module):
    """Set-encoder (sensor-identity) variant: input encoder + temporal encoder + heads."""

    def __init__(self, temporal_kind, n_nodes=9, d_model=64, hidden=64):
        super().__init__()
        self.input = SetInputEncoder(n_nodes, d_model=d_model, hidden=hidden)
        if temporal_kind == "cnn":
            self.temporal = CNN1D(d_model, hidden=hidden)
        elif temporal_kind == "lstm":
            self.temporal = RNNBase("lstm", d_model, hidden=hidden)
        elif temporal_kind == "gru":
            self.temporal = RNNBase("gru", d_model, hidden=hidden)
        elif temporal_kind == "transformer":
            self.temporal = TransformerEnc(d_model, d_model=hidden)
        else:
            raise ValueError(temporal_kind)
        d = self.temporal.repr_dim
        self.det_head = nn.Linear(d, 1)
        self.loc_head = nn.Linear(d, n_nodes + 1)
        self.repr_dim = d
        self._node_ids = None

    def set_sensors(self, node_ids):
        self._node_ids = torch.tensor(np.asarray(node_ids), dtype=torch.long)

    def param_count(self):
        return _num_params(self)

    def forward(self, x):
        node_ids = self._node_ids.to(x.device)
        h = self.input(x, node_ids)      # (B, T, d)
        z = self.temporal(h)
        return self.det_head(z).squeeze(-1), self.loc_head(z)


def build_set_model(kind, n_nodes=9, hidden=64):
    return SetLeakModel(kind, n_nodes=n_nodes, d_model=hidden, hidden=hidden)


def get_net1_adjacency(inp_path="datasets/LeakDB/CCWI-WDSA2018/Dataset_Generator_Py3/networks/Net1.inp"):
    """9x9 junction adjacency of LeakDB Net1 from pipes (tank/reservoir excluded)."""
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    junctions = sorted(wn.junction_name_list)
    j_idx = {j: i for i, j in enumerate(junctions)}
    n = len(junctions)
    adj = np.zeros((n, n), dtype=np.float32)
    for name, link in wn.pipes():
        a, b = link.start_node_name, link.end_node_name
        if a in j_idx and b in j_idx:
            adj[j_idx[a], j_idx[b]] = 1.0
            adj[j_idx[b], j_idx[a]] = 1.0
    return adj, junctions


def build_gnn_model(adj, hidden=64):
    return GNNLeakModel(adj, d=hidden)


def leak_model_param_count(m):
    return sum(p.numel() for p in m.parameters())


class LeakModel(nn.Module):
    """Encoder + detection & localization heads."""

    def __init__(self, encoder, n_loc=10):
        super().__init__()
        self.encoder = encoder
        d = encoder.repr_dim
        self.det_head = nn.Linear(d, 1)
        self.loc_head = nn.Linear(d, n_loc)

    def forward(self, x):
        z = self.encoder(x)
        return self.det_head(z).squeeze(-1), self.loc_head(z)

    def param_count(self):
        return _num_params(self)


def build_model(kind, in_ch, hidden=64):
    kind = kind.lower()
    if kind == "cnn":
        return LeakModel(CNN1D(in_ch, hidden=hidden))
    if kind == "lstm":
        return LeakModel(RNNBase("lstm", in_ch, hidden=hidden))
    if kind == "gru":
        return LeakModel(RNNBase("gru", in_ch, hidden=hidden))
    if kind == "transformer":
        return LeakModel(TransformerEnc(in_ch, d_model=hidden))
    raise ValueError(kind)


if __name__ == "__main__":
    # param report for a fair comparison
    for kind in ["cnn", "lstm", "gru", "transformer"]:
        m = build_model(kind, in_ch=9, hidden=64)
        n = m.param_count()
        print(f"{kind:12s} total params: {n:8d}  ({n/1e3:.1f}k)")

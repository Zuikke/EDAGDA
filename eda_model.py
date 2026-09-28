import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def compute_mmd(x, y, bandwidth=None):
    if bandwidth is None:
        z = torch.cat([x, y], dim=0)
        n_med = min(z.size(0), 1024)
        idx = torch.randperm(z.size(0), device=z.device)[:n_med]
        bandwidth = torch.pdist(z[idx]).median().clamp_min(1e-3)
    gamma = 0.5 / bandwidth.pow(2)

    kxx = torch.exp(-gamma * torch.pdist(x).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.pdist(y).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return kxx + kyy - 2.0 * kxy


class SharedGCNVAE(nn.Module):
    def __init__(self, in_dim, hidden=64, latent=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Returns (mu, logvar) of the approximate posterior q(Z | X, A)."""
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def encode(self, x, edge_index):
        mu, logvar = self.forward(x, edge_index)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()


class EdgeFeatureEncoder(nn.Module):

    def __init__(self, latent_dim, hid_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hid_dim),
            nn.ReLU(),
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        return self.mlp(torch.cat([z[src], z[dst]], dim=-1))


class EdgePredictor(nn.Module):

    def __init__(self, shared_vgae, latent_dim=64, edge_hid=32):
        super().__init__()
        self.shared_vgae = shared_vgae
        self.edge_enc = EdgeFeatureEncoder(latent_dim, edge_hid)

    def encode(self, x, edge_index):
        return self.shared_vgae.encode(x, edge_index)

    def edge_latents(self, z, edge_index):
        return self.edge_enc(z, edge_index)

    @staticmethod
    def inner_product(z):
        return z @ z.t()

    @staticmethod
    def topk_new_edges(z, edge_index, num_edges, ratio, num_nodes):

        scores = EdgePredictor.inner_product(z)
        diag = torch.arange(num_nodes, device=z.device)
        scores[diag, diag] = -float("inf")
        flat = scores.view(-1)
        flat[edge_index[0] * num_nodes + edge_index[1]] = -float("inf")
        k = max(1, int(ratio * num_edges))
        _, idx = torch.topk(flat, k)
        src = idx // num_nodes
        dst = idx % num_nodes
        return torch.stack([src, dst], dim=0)


class ClassifierGNN(nn.Module):

    def __init__(self, in_dim, hidden=64, num_classes=None, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h)

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from eda_model import ClassifierGNN, EdgePredictor, SharedGCNVAE, compute_mmd


def arg_parse():
    parser = argparse.ArgumentParser(
        description="EDA-GDA: Edge Distribution Alignment for Graph Domain Adaptation")

    # dataset
    parser.add_argument("--dataset", type=str, default="DBLP_ACM",
                        choices=["DBLP_ACM", "Airports", "Twitch"])
    parser.add_argument("--src_name", type=str, default="dblp")
    parser.add_argument("--tgt_name", type=str, default="acm")
    parser.add_argument("--data_dir", type=str, default="./data_files")

    # training schedule
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr1", type=float, default=0.01, help="stage-1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-3, help="stage-2 learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5)

    # stage 1 (edge distribution alignment)
    parser.add_argument("--hidden", type=int, default=64, help="shared GCN encoder hidden dim")
    parser.add_argument("--latent", type=int, default=64, help="node latent dim d_z")
    parser.add_argument("--edge_hid", type=int, default=32, help="edge encoder output dim")
    parser.add_argument("--alpha", type=float, default=0.01, help="edge-prediction loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="MMD loss weight")
    parser.add_argument("--edge_ratio", type=float, default=0.1,
                        help="ratio r of added edges, K = r * |E|")
    parser.add_argument("--reset_frequency", type=int, default=20,
                        help="frequency of refreshing the aligned graphs (epochs)")

    # stage 2 (attribute alignment)
    parser.add_argument("--gnn_hidden", type=int, default=64, help="stage-2 GNN hidden dim")
    parser.add_argument("--start_pseudo_epoch", type=int, default=150,
                        help="epoch from which the pseudo-label loss is active")
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=0.8,
                        help="confidence threshold of high-confidence target nodes")

    return parser.parse_args()


def assign_masks(graph, seed=0, src_ratios=(0.5, 0.3, 0.2), tgt_ratios=(0.2, 0.8)):
    """Class-stratified random split (paper protocol).

    Source graph: 50/30/20 train/val/test. Target graph: 20/80 val/test,
    where the validation labels are used solely for model selection.
    Overrides any loader-internal split.
    """
    rng = np.random.RandomState(seed)
    y = graph.y.cpu().numpy()
    train_idx, val_idx, test_idx, tgt_val_idx, tgt_test_idx = [], [], [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        t1 = int(n * src_ratios[0])
        t2 = int(n * (src_ratios[0] + src_ratios[1]))
        v1 = int(n * tgt_ratios[0])
        train_idx.append(idx[:t1])
        val_idx.append(idx[t1:t2])
        test_idx.append(idx[t2:])
        tgt_val_idx.append(idx[:v1])
        tgt_test_idx.append(idx[v1:])

    def to_tensor(lst):
        return torch.from_numpy(np.concatenate(lst)).long()

    graph.source_training_mask = to_tensor(train_idx)
    graph.source_validation_mask = to_tensor(val_idx)
    graph.source_testing_mask = to_tensor(test_idx)
    graph.target_validation_mask = to_tensor(tgt_val_idx)
    graph.target_testing_mask = to_tensor(tgt_test_idx)
    return graph


def bce_recon(z, edge_index):
    """Edge reconstruction loss on 1:1 positive/negative samples.

    a_hat_ij = sigma(z_i^T z_j) is the inner-product decoder of Stage 1.
    """
    pos = edge_index
    neg = negative_sampling(edge_index, num_nodes=z.size(0),
                            num_neg_samples=edge_index.size(1))
    pos_score = (z[pos[0]] * z[pos[1]]).sum(dim=-1)
    neg_score = (z[neg[0]] * z[neg[1]]).sum(dim=-1)
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([
        torch.ones(pos_score.size(0), device=z.device),
        torch.zeros(neg_score.size(0), device=z.device),
    ])
    return F.binary_cross_entropy_with_logits(scores, labels)


def evaluate(gnn, src_dataset, tgt_dataset, A_s, A_t):
    gnn.eval()
    with torch.no_grad():
        logits_s = gnn(src_dataset.x, A_s)
        logits_t = gnn(tgt_dataset.x, A_t)
        pred_s = logits_s.argmax(dim=-1)
        pred_t = logits_t.argmax(dim=-1)

    def acc(pred, mask, y):
        return (pred[mask] == y[mask]).float().mean().item()

    return [
        acc(pred_s, src_dataset.source_training_mask, src_dataset.y),
        acc(pred_s, src_dataset.source_validation_mask, src_dataset.y),
        acc(pred_s, src_dataset.source_testing_mask, src_dataset.y),
        acc(pred_t, tgt_dataset.target_validation_mask, tgt_dataset.y),
        acc(pred_t, tgt_dataset.target_testing_mask, tgt_dataset.y),
    ]


def train(args, src_dataset, tgt_dataset, device):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    src_dataset = src_dataset.to(device)
    tgt_dataset = tgt_dataset.to(device)

    # Stage 1 encodes on the original adjacencies; the aligned graphs A'
    # start as the originals and are refreshed every reset_frequency epochs.
    A_s_orig = src_dataset.edge_index.clone()
    A_t_orig = tgt_dataset.edge_index.clone()
    A_s = A_s_orig.clone()
    A_t = A_t_orig.clone()

    in_dim = src_dataset.x.size(1)
    num_classes = int(src_dataset.y.unique().size(0))

    shared_vgae = SharedGCNVAE(in_dim, hidden=args.hidden, latent=args.latent,
                               dropout=args.dropout).to(device)
    # EP_S and EP_T share the variational encoder; only the edge encoders differ.
    ep_s = EdgePredictor(shared_vgae, latent_dim=args.latent, edge_hid=args.edge_hid)
    ep_t = EdgePredictor(shared_vgae, latent_dim=args.latent, edge_hid=args.edge_hid)
    gnn = ClassifierGNN(in_dim, hidden=args.gnn_hidden, num_classes=num_classes,
                        dropout=args.dropout).to(device)

    # collect each parameter once (the shared encoder is common to EP_S and EP_T)
    stage1_params = (list(shared_vgae.parameters())
                     + list(ep_s.edge_enc.parameters())
                     + list(ep_t.edge_enc.parameters()))
    opt1 = torch.optim.Adam(stage1_params, lr=args.lr1, weight_decay=args.weight_decay)
    opt2 = torch.optim.Adam(gnn.parameters(), lr=args.lr2, weight_decay=args.weight_decay)

    list_acc = [[] for _ in range(5)]       # src train/val/test, tgt val/test
    list_losses = [[] for _ in range(5)]    # stage1, ep, mmd, source, pseudo
    best_tgt_val = -1.0
    best_accs = None

    for e in range(1, args.epochs + 1):
        shared_vgae.train()
        gnn.train()

        # ---------- Stage 1: edge distribution alignment ----------
        z_s, mu_s, logvar_s = shared_vgae.encode(src_dataset.x, A_s_orig)
        z_t, mu_t, logvar_t = shared_vgae.encode(tgt_dataset.x, A_t_orig)

        recon_s = bce_recon(z_s, A_s_orig)
        recon_t = bce_recon(z_t, A_t_orig)
        kl_s = SharedGCNVAE.kl_loss(mu_s, logvar_s)
        kl_t = SharedGCNVAE.kl_loss(mu_t, logvar_t)

        # cross-domain prediction: top-K missing edges of each domain
        hat_E_s = EdgePredictor.topk_new_edges(z_s, A_s_orig, A_s_orig.size(1),
                                               args.edge_ratio, src_dataset.num_nodes)
        hat_E_t = EdgePredictor.topk_new_edges(z_t, A_t_orig, A_t_orig.size(1),
                                               args.edge_ratio, tgt_dataset.num_nodes)

        # distributional consistency constraint: the domain-D edge encoder is
        # cross-applied to the predicted edges of the opposite domain, and MMD
        # pulls their edge-latent distribution toward that of the real edges
        # of domain D.
        z_s_real = ep_s.edge_latents(z_s, A_s_orig)
        z_s_cross = ep_s.edge_latents(z_t, hat_E_t)
        z_t_real = ep_t.edge_latents(z_t, A_t_orig)
        z_t_cross = ep_t.edge_latents(z_s, hat_E_s)
        mmd = compute_mmd(z_s_real, z_s_cross) + compute_mmd(z_t_real, z_t_cross)

        ep_loss = recon_s + recon_t + kl_s + kl_t
        loss_stage1 = args.alpha * ep_loss + args.beta * mmd
        opt1.zero_grad()
        loss_stage1.backward()
        opt1.step()

        if e % args.reset_frequency == 0:
            A_s = torch.cat([A_s_orig, hat_E_s], dim=1)
            A_t = torch.cat([A_t_orig, hat_E_t], dim=1)

        # ---------- Stage 2: attribute alignment with pseudo-label enhancement ----------
        logits_s = gnn(src_dataset.x, A_s)
        logits_t = gnn(tgt_dataset.x, A_t)

        loss_source = F.cross_entropy(
            logits_s[src_dataset.source_training_mask],
            src_dataset.y[src_dataset.source_training_mask])

        if e >= args.start_pseudo_epoch:
            probs = F.softmax(logits_t, dim=-1)
            confidence, _ = probs.max(dim=-1)
            high_mask = confidence > args.conf_threshold
            if high_mask.any():
                logits_high = logits_t[high_mask]
                # differentiable "soft" pseudo-labels via Gumbel-Softmax
                pseudo = F.gumbel_softmax(logits_high, tau=args.gumbel_tau, hard=False)
                loss_pseudo = -(pseudo * F.log_softmax(logits_high, dim=-1)).sum(dim=-1).mean()
            else:
                loss_pseudo = torch.zeros((), device=device)
        else:
            loss_pseudo = torch.zeros((), device=device)

        loss_stage2 = loss_source + loss_pseudo
        opt2.zero_grad()
        loss_stage2.backward()
        opt2.step()

        # ---------- evaluation ----------
        accs = evaluate(gnn, src_dataset, tgt_dataset, A_s, A_t)
        if accs[3] > best_tgt_val:
            best_tgt_val = accs[3]
            best_accs = accs

        losses = [loss_stage1.item(), ep_loss.item(), mmd.item(),
                  loss_source.item(), loss_pseudo.item()]
        for i in range(5):
            list_acc[i].append(accs[i])
            list_losses[i].append(losses[i])

        print(f"epoch: {e:3d} | stage1: {loss_stage1.item():.4f} "
              f"(ep {ep_loss.item():.4f}, mmd {mmd.item():.4f}) | "
              f"stage2: {loss_stage2.item():.4f} "
              f"(src {loss_source.item():.4f}, pseudo {loss_pseudo.item():.4f}) | "
              f"accs: src_tr {accs[0]:.4f} src_va {accs[1]:.4f} "
              f"tgt_va {accs[3]:.4f} tgt_te {accs[4]:.4f}")

    os.makedirs("./eda_gda_save", exist_ok=True)
    np.save("./eda_gda_save/accs.npy", np.asarray(list_acc))
    np.save("./eda_gda_save/losses.npy", np.asarray(list_losses))

    print("Best ACC (checkpoint selected by target validation accuracy):")
    print(f"  src_train={best_accs[0]:.4f}  src_val={best_accs[1]:.4f}  "
          f"src_test={best_accs[2]:.4f}  "
          f"tgt_val={best_accs[3]:.4f}  tgt_test={best_accs[4]:.4f}")
    return best_accs


def load_datasets(args):
    # Lazy import: data_process/pre_datasets.py pulls in pre_arxiv, which is
    # not shipped in this folder; importing here keeps synthetic/other usage
    # of this script independent of that module.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_process import pre_datasets

    if args.dataset == "DBLP_ACM":
        src = pre_datasets.prepare_dblp_acm(args.data_dir, args.src_name)
        tgt = pre_datasets.prepare_dblp_acm(args.data_dir, args.tgt_name)
    elif args.dataset == "Airports":
        src = pre_datasets.prepare_airports(os.path.join(args.data_dir, "airports"),
                                            args.src_name)
        tgt = pre_datasets.prepare_airports(os.path.join(args.data_dir, "airports"),
                                            args.tgt_name)
    elif args.dataset == "Twitch":
        src = pre_datasets.load_twitch(args.data_dir, args.src_name)
        tgt = pre_datasets.load_twitch(args.data_dir, args.tgt_name)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # paper protocol: source 50/30/20, target 20/80 (overrides loader splits)
    assign_masks(src, seed=args.seed)
    assign_masks(tgt, seed=args.seed)
    return src, tgt


if __name__ == "__main__":
    args = arg_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    src_dataset, tgt_dataset = load_datasets(args)
    train(args, src_dataset, tgt_dataset, device)

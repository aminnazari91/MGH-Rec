import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


def normalize_rating(r):
    return (r - 1.0) / 4.0

def denormalize_rating(r):
    return r * 4.0 + 1.0

def bpr_loss(user_emb, pos_movie_emb, neg_movie_emb):
    pos = (user_emb * pos_movie_emb).sum(dim=1)
    neg = (user_emb * neg_movie_emb).sum(dim=1)
    return -torch.mean(F.logsigmoid(pos - neg))

def train(model, 
          g, 
          train_idx, 
          val_idx,
          val_ratings, 
          labels, 
          user_pos_dict, 
          train_ratings,
          epochs=200, 
          lr=1e-3, 
          lambda_cls=1.0, 
          lambda_bpr=0.5, 
          lambda_mse=0.8,
          patience=10, 
          device='cuda'):
    
    model.to(device)
    g = g.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    ce = nn.CrossEntropyLoss()

    best_val = float('inf')
    wait = 0
    best_state = None

    bpr_users = torch.tensor(list(user_pos_dict.keys()), device=device)

    for epoch in range(1, epochs + 1):
        model.train()

        h_gcn, h_hgt = model(g)

        # --- classification ---
        logits = model.classify(h_gcn, h_hgt)
        loss_cls = ce(logits[train_idx], labels[train_idx])

        # --- BPR (GCN) ---
        u = bpr_users[torch.randint(0, len(bpr_users), (len(train_idx),))]
        pos_m = torch.tensor(
            [random.choice(user_pos_dict[int(x)]) for x in u],
            device=device
        )

        neg_m = []
        for ui in u.tolist():
            neg = random.randint(0, g.num_nodes('movie') - 1)
            while neg in user_pos_dict[int(ui)]:
                neg = random.randint(0, g.num_nodes('movie') - 1)
            neg_m.append(neg)
        neg_m = torch.tensor(neg_m, device=device)

        loss_bpr = bpr_loss(
            h_gcn['user'][u],
            h_gcn['movie'][pos_m],
            h_gcn['movie'][neg_m]
        )

        # --- rating MSE ---
        batch = torch.randint(0, len(train_ratings), (512,))
        rows = train_ratings.iloc[batch.cpu().tolist()]

        u_r = torch.tensor(rows['user_node_id'].values, device=device)
        m_r = torch.tensor(rows['movie_node_id'].values, device=device)
        r_norm = torch.tensor(
            (rows['rate'].values - 1.0) / 4.0,
            device=device,
            dtype=torch.float
        )

        preds = model.predict_rating(h_gcn, h_hgt, u_r, m_r)
        loss_mse = F.mse_loss(preds, r_norm)

        loss = (
            lambda_cls * loss_cls +
            lambda_bpr * loss_bpr +
            lambda_mse * loss_mse
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- validation ---
        # model.eval()
        # with torch.no_grad():
        #     h_gcn, h_hgt = model(g)
        #     val_logits = model.classify(h_gcn, h_hgt)
        #     val_loss = ce(val_logits[val_idx], labels[val_idx])

        # if val_loss < best_val:
        #     best_val = val_loss
        #     best_state = model.state_dict()
        #     wait = 0
        # else:
        #     wait += 1


        # --- validation multi-task ---
        model.eval()
        with torch.no_grad():
            h_gcn, h_hgt = model(g)

            # classification loss
            val_logits = model.classify(h_gcn, h_hgt)
            val_loss_cls = ce(val_logits[val_idx], labels[val_idx])

            val_rows = val_ratings[val_ratings['movie_node_id'].isin(val_idx)]

            u_r = torch.tensor(val_rows['user_node_id'].values, device=device)
            m_r = torch.tensor(val_rows['movie_node_id'].values, device=device)
            r_norm = torch.tensor((val_rows['rate'].values - 1) / 4, device=device, dtype=torch.float)

            preds_r = model.predict_rating(h_gcn, h_hgt, u_r, m_r)
            val_loss_rec = F.mse_loss(preds_r, r_norm)


            # early stopping
            val_loss = lambda_cls * val_loss_cls + lambda_mse * val_loss_rec

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"CLS: {loss_cls.item():.3f} | "
                f"MSE: {loss_mse.item():.3f} | "
                f"VAL: {val_loss.item():.4f}"
            )

        if wait >= patience:
            print("Early stopping!")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def test(
    model,
    g,
    test_idx,
    labels,
    test_ratings,
    k=10,
    device='cuda'
):

    model.eval()
    device = torch.device(device)

    with torch.no_grad():
        h_gcn, h_hgt = model(g)

        # -------------------------------
        #  Classification Metrics
        # -------------------------------
        logits = model.classify(h_gcn, h_hgt)

        # test_idx به tensor روی همان device
        if isinstance(test_idx, np.ndarray):
            test_idx_tensor = torch.tensor(test_idx, dtype=torch.long, device=logits.device)
        else:
            test_idx_tensor = test_idx

        preds = logits[test_idx_tensor].argmax(dim=1)

        acc = accuracy_score(labels[test_idx_tensor].cpu(), preds.cpu())
        f1_micro = f1_score(labels[test_idx_tensor].cpu(), preds.cpu(), average='micro')
        f1_macro = f1_score(labels[test_idx_tensor].cpu(), preds.cpu(), average='macro')

        # -------------------------------
        #  Clustering Metrics (Movie embeddings)
        # -------------------------------
        X = h_hgt['movie'][test_idx_tensor].cpu().numpy()  # HGT dominant embeddings
        y = labels[test_idx_tensor].cpu().numpy()

        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42).fit(X)
        nmi = normalized_mutual_info_score(y, kmeans.labels_)
        ari = adjusted_rand_score(y, kmeans.labels_)

        # -------------------------------
        #  Rating Prediction Metrics
        # -------------------------------
        preds_r, trues_r = [], []
        for row in test_ratings.itertuples(index=False):
            u = row.user_node_id
            m = row.movie_node_id
            true_rating = row.rate

            score_norm = model.predict_rating(
                h_gcn, h_hgt,
                u_idx=torch.tensor([u], device=h_gcn['user'].device),
                m_idx=torch.tensor([m], device=h_gcn['movie'].device)
            ).item()

            pred_rating = 1 + 4 * score_norm  # denormalize
            preds_r.append(pred_rating)
            trues_r.append(true_rating)

        preds_r = np.array(preds_r)
        trues_r = np.array(trues_r)
        rmse = np.sqrt(np.mean((preds_r - trues_r) ** 2))
        mae = np.mean(np.abs(preds_r - trues_r))


        # -------------------------------
        # 5️⃣ Return final dictionary
        # -------------------------------
        return {
            'Name': model.__class__.__name__,
            'Accuracy': acc,
            'Micro-F1': f1_micro,
            'Macro-F1': f1_macro,
            'NMI': nmi,
            'ARI': ari,
            'RMSE': rmse,
            'MAE': mae,
        }

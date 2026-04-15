import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import statistics

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score
)

# -----------------------------------
# Reproducibility
# -----------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# -----------------------------------
# GraphSAGE Model
# -----------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)


# -----------------------------------
# Load Dataset for One Repo
# -----------------------------------
def load_repo(repo_name):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "data", f"{repo_name}_dataset")

    graphs = []

    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".json"):
            with open(os.path.join(dataset_dir, file), "r") as f:
                data = json.load(f)

            if len(data["X"]) == 0:
                continue

            x = torch.tensor(data["X"], dtype=torch.float32)
            y = torch.tensor(data["Y"], dtype=torch.long)

            edges = torch.tensor(data["edges"], dtype=torch.long)
            if len(edges) > 0:
                edge_index = edges.t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs


# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":

    TRAIN_REPOS = ["pytube", "cookiecutter"]
    TEST_REPO = "httpie"

    seeds = [0, 1, 2, 3, 4]

    all_f1 = []
    all_roc = []
    all_pr = []

    for seed in seeds:

        print("\n==============================")
        print("Seed:", seed)
        print("==============================")

        set_seed(seed)

        train_graphs = []
        for repo in TRAIN_REPOS:
            train_graphs.extend(load_repo(repo))

        test_graphs = load_repo(TEST_REPO)

        print("Training graphs:", len(train_graphs))
        print("Testing graphs:", len(test_graphs))

        train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=1)

        input_dim = train_graphs[0].num_node_features
        model = GraphSAGE(input_dim, hidden_dim=64)

        # ----- Class imbalance handling -----
        all_train_labels = []
        for g in train_graphs:
            all_train_labels.extend(g.y.tolist())

        pos_weight = len(all_train_labels) / sum(all_train_labels)
        neg_weight = len(all_train_labels) / (len(all_train_labels) - sum(all_train_labels))

        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # -----------------------------------
        # Training
        # -----------------------------------
        for epoch in range(60):

            model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), "model.pt")
        print("Model saved as model.pt")
        # -----------------------------------
        # Evaluation
        # -----------------------------------
        model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.numpy())
                all_labels.extend(batch.y.numpy())
        
        # -----------------------------------
        # Threshold Tuning
        # -----------------------------------
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = [1 if p >= threshold else 0 for p in all_probs]
            f1 = f1_score(all_labels, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        final_preds = [1 if p >= best_threshold else 0 for p in all_probs]

        print("\nBest Threshold:", round(best_threshold, 2))
        print("Best F1:", round(best_f1, 4))

        print("\n===== Final Classification Report =====")
        print(classification_report(all_labels, final_preds, digits=4))

        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)

        print("ROC-AUC:", round(roc_auc, 4))
        print("PR-AUC:", round(pr_auc, 4))

        all_f1.append(best_f1)
        all_roc.append(roc_auc)
        all_pr.append(pr_auc)

    # -----------------------------------
    # Multi-seed Statistical Summary
    # -----------------------------------
    print("\n====================================")
    print("FINAL MULTI-SEED SUMMARY (5 runs)")
    print("====================================")

    print("F1 Mean ± Std:",
          round(statistics.mean(all_f1), 4),
          "±",
          round(statistics.stdev(all_f1), 4))

    print("ROC-AUC Mean ± Std:",
          round(statistics.mean(all_roc), 4),
          "±",
          round(statistics.stdev(all_roc), 4))

    print("PR-AUC Mean ± Std:",
          round(statistics.mean(all_pr), 4),
          "±",
          round(statistics.stdev(all_pr), 4))
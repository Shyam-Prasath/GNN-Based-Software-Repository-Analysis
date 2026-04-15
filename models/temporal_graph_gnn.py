import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.metrics import classification_report
import numpy as np


# ----------------------------
# Graph Encoder (GraphSAGE)
# ----------------------------
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# ----------------------------
# Temporal LSTM Model
# ----------------------------
class TemporalModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ----------------------------
# Load dataset
# ----------------------------
def load_datasets(repo_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "data", f"{repo_name}_dataset")

    datasets = []

    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".json"):
            with open(os.path.join(dataset_dir, file), "r") as f:
                data = json.load(f)
                datasets.append(data)

    return datasets


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    repo_name = "pytube"  # change for others

    datasets = load_datasets(repo_name)

    print("Snapshots loaded:", len(datasets))

    input_dim = len(datasets[0]["X"][0])
    hidden_dim = 64

    graph_encoder = GraphEncoder(input_dim, hidden_dim)
    temporal_model = TemporalModel(hidden_dim, 64)

    optimizer = torch.optim.Adam(
        list(graph_encoder.parameters()) +
        list(temporal_model.parameters()),
        lr=0.001
    )

    # ----------------------------
    # Build embeddings per snapshot
    # ----------------------------
    snapshot_embeddings = []
    snapshot_labels = []
    snapshot_files = []

    for t, data in enumerate(datasets):

        X = torch.tensor(data["X"], dtype=torch.float32)
        Y = torch.tensor(data["Y"], dtype=torch.long)
        files = data["files"]

        edges = torch.tensor(data["edges"], dtype=torch.long)

        if len(edges) > 0:
            edge_index = edges.t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        embeddings = graph_encoder(X, edge_index)

        snapshot_embeddings.append(embeddings.detach())
        snapshot_labels.append(Y)
        snapshot_files.append(files)

    # ----------------------------
    # Build temporal sequences
    # ----------------------------
    file_history = defaultdict(list)

    for t in range(len(snapshot_embeddings)):

        embeddings = snapshot_embeddings[t]
        labels = snapshot_labels[t]
        files = snapshot_files[t]

        for i, file in enumerate(files):
            file_history[file].append((t, embeddings[i], labels[i]))

    sequences = []
    targets = []

    sequence_length = 3

    for file, history in file_history.items():

        history = sorted(history, key=lambda x: x[0])

        if len(history) >= sequence_length + 1:

            for i in range(len(history) - sequence_length):

                seq = []
                for j in range(sequence_length):
                    seq.append(history[i + j][1].numpy())

                label = history[i + sequence_length][2].item()

                sequences.append(seq)
                targets.append(label)

    sequences = torch.tensor(sequences, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    print("Total sequences:", len(sequences))

    if len(sequences) == 0:
        print("Not enough data for temporal training.")
        exit()

    # Train/test split
    split = int(0.8 * len(sequences))
    X_train, X_test = sequences[:split], sequences[split:]
    Y_train, Y_test = targets[:split], targets[split:]

    # Class weighting
    class_counts = torch.bincount(Y_train)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ----------------------------
    # Training
    # ----------------------------
    for epoch in range(60):

        graph_encoder.train()
        temporal_model.train()

        optimizer.zero_grad()

        outputs = temporal_model(X_train)
        loss = criterion(outputs, Y_train)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # ----------------------------
    # Evaluation
    # ----------------------------
    graph_encoder.eval()
    temporal_model.eval()

    with torch.no_grad():
        preds = torch.argmax(temporal_model(X_test), dim=1)

    print("\n===== Temporal Graph GNN Results =====")
    print(classification_report(
        Y_test.numpy(),
        preds.numpy(),
        digits=4,
        zero_division=0
    ))

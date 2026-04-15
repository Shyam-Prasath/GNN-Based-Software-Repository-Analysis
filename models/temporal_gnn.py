import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report


# ------------------------
# Temporal Model (LSTM)
# ------------------------
class TemporalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


# ------------------------
# Load datasets
# ------------------------
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


# ------------------------
# Build sequences
# ------------------------
def build_sequences(datasets, sequence_length=3):

    file_history = {}

    for t, data in enumerate(datasets):
        files = data["files"]
        X = data["X"]
        Y = data["Y"]

        for i, file in enumerate(files):
            if file not in file_history:
                file_history[file] = []

            file_history[file].append((t, X[i], Y[i]))

    sequences = []
    labels = []

    for file, history in file_history.items():
        history = sorted(history, key=lambda x: x[0])

        if len(history) >= sequence_length + 1:
            for i in range(len(history) - sequence_length):
                seq_features = []
                for j in range(sequence_length):
                    seq_features.append(history[i + j][1])

                label = history[i + sequence_length][2]

                sequences.append(seq_features)
                labels.append(label)

    return np.array(sequences), np.array(labels)


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":

    repo_name = "pytube"  # change per repo

    datasets = load_datasets(repo_name)

    print("Total snapshots loaded:", len(datasets))

    sequences, labels = build_sequences(datasets)

    print("Total sequences:", len(sequences))

    if len(sequences) == 0:
        print("Not enough temporal data.")
        exit()

    X = torch.tensor(sequences, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)

    # Train/Test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ------------------------
    # CLASS WEIGHTING (FIX)
    # ------------------------
    counter = Counter(Y_train.numpy())
    total = len(Y_train)

    weight_0 = total / counter[0]
    weight_1 = total / counter[1]

    print("Class distribution:", counter)
    print("Class weights:", weight_0, weight_1)

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)

    model = TemporalModel(input_dim=X.shape[2], hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------
    # Training
    # ------------------------
    for epoch in range(60):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, Y_train)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # ------------------------
    # Evaluation
    # ------------------------
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)

    print("\n===== Temporal Model Results =====")
    print(classification_report(
        Y_test.numpy(),
        preds.numpy(),
        digits=4,
        zero_division=0
    ))

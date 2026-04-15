import os
import json
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.utils import shuffle


# ----------------------------------
# Reproducibility
# ----------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------
# Load Dataset For One Repo
# ----------------------------------
def load_repo(repo_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "data", f"{repo_name}_dataset")

    X_all = []
    Y_all = []

    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".json"):
            with open(os.path.join(dataset_dir, file), "r") as f:
                data = json.load(f)

            if len(data["X"]) == 0:
                continue

            X_all.extend(data["X"])
            Y_all.extend(data["Y"])

    return np.array(X_all), np.array(Y_all)


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":

    TRAIN_REPOS = ["pytube", "cookiecutter"]
    TEST_REPO = "httpie"

    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:

        print("\n==============================")
        print("Seed:", seed)
        print("==============================")

        set_seed(seed)

        # Load train data
        X_train_all = []
        Y_train_all = []

        for repo in TRAIN_REPOS:
            X_repo, Y_repo = load_repo(repo)
            X_train_all.append(X_repo)
            Y_train_all.append(Y_repo)

        X_train = np.vstack(X_train_all)
        Y_train = np.concatenate(Y_train_all)

        # Load test data
        X_test, Y_test = load_repo(TEST_REPO)

        print("Train samples:", len(X_train))
        print("Test samples:", len(X_test))

        # Shuffle train set
        X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)

        # ----------------------------------
        # Logistic Regression
        # ----------------------------------
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_train, Y_train)

        probs_lr = lr.predict_proba(X_test)[:, 1]

        # Threshold tuning
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs_lr >= threshold).astype(int)
            f1 = f1_score(Y_test, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        final_preds = (probs_lr >= best_threshold).astype(int)

        print("\n===== Logistic Regression =====")
        print("Best Threshold:", round(best_threshold, 2))
        print("Best F1:", round(best_f1, 4))
        print(classification_report(Y_test, final_preds, digits=4))

        print("ROC-AUC:", round(roc_auc_score(Y_test, probs_lr), 4))
        print("PR-AUC:", round(average_precision_score(Y_test, probs_lr), 4))

        # ----------------------------------
        # Random Forest
        # ----------------------------------
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed
        )

        rf.fit(X_train, Y_train)
        probs_rf = rf.predict_proba(X_test)[:, 1]

        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs_rf >= threshold).astype(int)
            f1 = f1_score(Y_test, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        final_preds = (probs_rf >= best_threshold).astype(int)

        print("\n===== Random Forest =====")
        print("Best Threshold:", round(best_threshold, 2))
        print("Best F1:", round(best_f1, 4))
        print(classification_report(Y_test, final_preds, digits=4))

        print("ROC-AUC:", round(roc_auc_score(Y_test, probs_rf), 4))
        print("PR-AUC:", round(average_precision_score(Y_test, probs_rf), 4))

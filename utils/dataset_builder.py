import os
import json


def build_dataset(repo_name):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    features_dir = os.path.join(project_root, "data", f"{repo_name}_features")
    labels_dir = os.path.join(project_root, "data", f"{repo_name}_labels")
    graphs_dir = os.path.join(project_root, "data", f"{repo_name}_graphs")

    output_dir = os.path.join(project_root, "data", f"{repo_name}_dataset")
    os.makedirs(output_dir, exist_ok=True)

    feature_files = sorted(os.listdir(features_dir))
    label_files = sorted(os.listdir(labels_dir))
    graph_files = sorted(os.listdir(graphs_dir))

    total_snapshots = min(len(feature_files), len(label_files), len(graph_files))

    print(f"Total usable snapshots: {total_snapshots}")

    for i in range(total_snapshots):

        print(f"Building dataset for snapshot {i}")

        # Load features
        with open(os.path.join(features_dir, feature_files[i]), "r") as f:
            features = json.load(f)

        # Load labels
        with open(os.path.join(labels_dir, label_files[i]), "r") as f:
            bug_files = set(json.load(f))

        # Load graph
        with open(os.path.join(graphs_dir, graph_files[i]), "r") as f:
            graph = json.load(f)

        X = []
        Y = []
        file_names = []

        file_to_index = {}
        index = 0

        for file, feature_vector in features.items():
            file_to_index[file] = index
            X.append(feature_vector)
            Y.append(1 if file in bug_files else 0)
            file_names.append(file)
            index += 1

        # Build edge list (index-based)
        edges = []

        for src, targets in graph.items():
            if src in file_to_index:
                src_idx = file_to_index[src]
                for tgt in targets:
                    if tgt in file_to_index:
                        tgt_idx = file_to_index[tgt]
                        edges.append([src_idx, tgt_idx])

        dataset = {
            "X": X,
            "Y": Y,
            "files": file_names,
            "edges": edges
        }

        output_file = os.path.join(output_dir, f"dataset_{i}.json")

        with open(output_file, "w") as f:
            json.dump(dataset, f)

        print(f"Saved dataset_{i}.json")

    print("Datasets rebuilt with graph structure.")


if __name__ == "__main__":

    repo_name = "httpie"  # change per repo
    build_dataset(repo_name)

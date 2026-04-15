import os
import ast
import json
import subprocess
import networkx as nx
from collections import defaultdict


def count_loc(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except:
        return 0


def count_functions_classes(file_path):
    functions = 0
    classes = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
    except:
        pass

    return functions, classes


def compute_degrees(graph):
    in_degree = defaultdict(int)
    out_degree = {}

    for node, imports in graph.items():
        out_degree[node] = len(imports)
        for imp in imports:
            in_degree[imp] += 1

    return in_degree, out_degree


def compute_graph_metrics(graph):
    G = nx.DiGraph()

    for node, imports in graph.items():
        G.add_node(node)
        for imp in imports:
            G.add_edge(node, imp)

    # Convert to undirected for centrality metrics
    G_undirected = G.to_undirected()

    pagerank = nx.pagerank(G_undirected)
    betweenness = nx.betweenness_centrality(G_undirected)
    closeness = nx.closeness_centrality(G_undirected)
    clustering = nx.clustering(G_undirected)

    return pagerank, betweenness, closeness, clustering


if __name__ == "__main__":

    repo_name = "cookiecutter"  # change for other repos

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_path = os.path.join(project_root, "repos", repo_name)

    graph_dir = os.path.join(project_root, "data", f"{repo_name}_graphs")
    snapshot_commit_file = os.path.join(project_root, "data", f"{repo_name}_snapshots.txt")

    output_dir = os.path.join(project_root, "data", f"{repo_name}_features")
    os.makedirs(output_dir, exist_ok=True)

    with open(snapshot_commit_file, "r") as f:
        snapshot_commits = [line.strip() for line in f.readlines()]

    graph_files = sorted(os.listdir(graph_dir))

    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    default_branch = branch_result.stdout.strip()

    for i, graph_file in enumerate(graph_files):

        print(f"Processing snapshot {i}")

        subprocess.run(
            ["git", "checkout", snapshot_commits[i]],
            cwd=repo_path,
            capture_output=True
        )

        with open(os.path.join(graph_dir, graph_file), "r") as f:
            graph = json.load(f)

        in_degree, out_degree = compute_degrees(graph)

        # NEW STRUCTURAL FEATURES
        pagerank, betweenness, closeness, clustering = compute_graph_metrics(graph)

        snapshot_features = {}

        for file in graph.keys():

            full_path = os.path.join(repo_path, file)

            loc = count_loc(full_path)
            functions, classes = count_functions_classes(full_path)
            imports_count = len(graph[file])
            indeg = in_degree[file]
            outdeg = out_degree[file]

            pr = pagerank.get(file, 0)
            btw = betweenness.get(file, 0)
            clo = closeness.get(file, 0)
            clu = clustering.get(file, 0)

            snapshot_features[file] = [
                loc,
                functions,
                classes,
                imports_count,
                indeg,
                outdeg,
                pr,
                btw,
                clo,
                clu
            ]

        output_file = os.path.join(output_dir, f"features_{i}.json")

        with open(output_file, "w") as f:
            json.dump(snapshot_features, f)

        print(f"Features saved for snapshot {i}")

    subprocess.run(
        ["git", "checkout", default_branch],
        cwd=repo_path
    )

    print("All features extracted and branch restored.")
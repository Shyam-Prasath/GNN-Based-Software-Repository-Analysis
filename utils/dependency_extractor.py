import os
import ast
import json
import subprocess

def extract_imports(file_path):
    imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

    except Exception:
        pass

    return imports


def build_dependency_graph(repo_path, file_list):
    graph = {}

    file_set = set(file_list)

    for file in file_list:

        full_path = os.path.join(repo_path, file)
        graph[file] = []

        if not os.path.exists(full_path):
            continue

        imports = extract_imports(full_path)

        for imp in imports:

            # Convert module format to file path
            candidate = imp.replace(".", "/") + ".py"

            if candidate in file_set:
                graph[file].append(candidate)

    return graph


if __name__ == "__main__":

    repo_name = "pytube"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_path = os.path.join(project_root, "repos", repo_name)

    snapshot_commit_file = os.path.join(project_root, "data", f"{repo_name}_snapshots.txt")
    with open(snapshot_commit_file, "r") as f:
        snapshot_commits = [line.strip() for line in f.readlines()]

    snapshot_files_dir = os.path.join(project_root, "data", f"{repo_name}_files")
    snapshot_files = sorted(os.listdir(snapshot_files_dir))

    output_dir = os.path.join(project_root, "data", f"{repo_name}_graphs")
    os.makedirs(output_dir, exist_ok=True)

    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    default_branch = branch_result.stdout.strip()

    print(f"Default branch detected: {default_branch}")

    for i, snapshot_file in enumerate(snapshot_files):

        print(f"Processing snapshot {i+1}/{len(snapshot_files)}")

        commit_hash = snapshot_commits[i]

        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo_path,
            capture_output=True
        )

        file_path = os.path.join(snapshot_files_dir, snapshot_file)
        with open(file_path, "r") as f:
            file_list = [line.strip() for line in f.readlines()]

        graph = build_dependency_graph(repo_path, file_list)

        output_file = os.path.join(output_dir, f"graph_{i}.json")
        with open(output_file, "w") as f:
            json.dump(graph, f)

        print(f"Graph saved for snapshot {i}")

    subprocess.run(
        ["git", "checkout", default_branch],
        cwd=repo_path
    )

    print("All dependency graphs built and branch restored.")

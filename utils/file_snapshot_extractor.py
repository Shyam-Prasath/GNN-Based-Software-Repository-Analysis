import subprocess
import os

def get_snapshot_commits(repo_name, project_root):
    snapshot_file = os.path.join(project_root, "data", f"{repo_name}_snapshots.txt")
    with open(snapshot_file, "r") as f:
        commits = [line.strip() for line in f.readlines()]
    return commits

def checkout_commit(repo_path, commit_hash):
    subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=repo_path,
        capture_output=True
    )

def get_python_files(repo_path):
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    files = result.stdout.strip().split("\n")
    return files

if __name__ == "__main__":
    repo_name = "pytube"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_path = os.path.join(project_root, "repos", repo_name)

    commits = get_snapshot_commits(repo_name, project_root)

    output_dir = os.path.join(project_root, "data", f"{repo_name}_files")
    os.makedirs(output_dir, exist_ok=True)

    for i, commit in enumerate(commits):
        print(f"Processing snapshot {i+1}/{len(commits)}")

        checkout_commit(repo_path, commit)
        files = get_python_files(repo_path)

        output_file = os.path.join(output_dir, f"snapshot_{i}.txt")
        with open(output_file, "w") as f:
            for file in files:
                f.write(file + "\n")

    print("File extraction complete.")

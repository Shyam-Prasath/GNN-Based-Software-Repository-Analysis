import subprocess
import os

def get_commit_list(repo_path):
    result = subprocess.run(
        ["git", "rev-list", "--reverse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    commits = result.stdout.strip().split("\n")
    return commits

def select_snapshots(commits, interval=50):
    return commits[::interval]

if __name__ == "__main__":
    repo_name = "cookiecutter"  # change later
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_path = os.path.join(project_root, "repos", repo_name)

    commits = get_commit_list(repo_path)
    snapshots = select_snapshots(commits, interval=50)

    print(f"Total commits: {len(commits)}")
    print(f"Selected snapshots: {len(snapshots)}")

    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"{repo_name}_snapshots.txt")

    with open(output_file, "w") as f:
        for s in snapshots:
            f.write(s + "\n")

    print(f"Snapshots saved to {output_file}")

import os
import subprocess
import json

BUG_KEYWORDS = ["fix", "bug", "patch", "resolve", "error"]

def get_snapshot_commits(project_root, repo_name):
    file_path = os.path.join(project_root, "data", f"{repo_name}_snapshots.txt")
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def is_bug_commit(message):
    message = message.lower()
    return any(keyword in message for keyword in BUG_KEYWORDS)

def get_bug_files_between(repo_path, start_commit, end_commit):

    result = subprocess.run(
        ["git", "log", f"{start_commit}..{end_commit}", "--pretty=format:%H"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    commits = result.stdout.strip().split("\n")
    bug_files = set()

    for commit in commits:
        if not commit.strip():
            continue

        msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%B", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        ).stdout

        if not msg:
            continue

        if is_bug_commit(msg):

            files = subprocess.run(
                ["git", "show", "--name-only", "--pretty=", commit],
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            ).stdout.split("\n")

            for file in files:
                if file.endswith(".py"):
                    bug_files.add(file.strip())

    return bug_files

if __name__ == "__main__":

    repo_name = "pytube"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_path = os.path.join(project_root, "repos", repo_name)

    snapshot_commits = get_snapshot_commits(project_root, repo_name)

    output_dir = os.path.join(project_root, "data", f"{repo_name}_labels")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(snapshot_commits) - 1):

        print(f"Processing label for snapshot {i}")

        start_commit = snapshot_commits[i]
        end_commit = snapshot_commits[i + 1]

        bug_files = get_bug_files_between(repo_path, start_commit, end_commit)

        output_file = os.path.join(output_dir, f"labels_{i}.json")

        with open(output_file, "w") as f:
            json.dump(list(bug_files), f)

        print(f"Labels saved for snapshot {i}")

    print("All labels generated.")

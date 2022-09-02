import subprocess


def run(lst):
    path = "C:\\Users\\eliai\\Sync\\Programming\\Bracketless"
    print(" ".join(lst))
    return subprocess.run(
        lst, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", cwd=path
    )


def commits_from_branch(branch):
    proc = run(["git", "rev-list", branch])
    commits = []
    i = 0
    while i < len(proc.stdout):
        commits.append(proc.stdout[i : (i + 40)])
        i += 40
        if proc.stdout[i] != "\n":
            raise Exception
        i += 1
    return commits


commits = commits_from_branch("master")

for i in range(len(commits))[::-1]:
    print(f"{len(commits) - i}/{len(commits)}")
    current_commit = commits[i]
    run(["git", "switch", "--detach", current_commit])
    if i == len(commits) - 1:
        pass
    else:
        run(["git", "reset", "--mixed", current_formatted_commit])
    run(["black", "."])
    run(["git", "add", "*"])
    if i == len(commits) - 1:
        run(["git", "commit", "--amend", "--no-edit"])
    else:
        run(["git", "commit", f"--reuse-message={current_commit}"])
    proc = run(["git", "rev-parse", "HEAD"])
    current_formatted_commit = proc.stdout[0:40]
run(["git", "switch", "-c", "formatted-master"])

run(["git", "push", "upstream", "formatted-master"])
formatted_commits = commits_from_branch("formatted-master")
for i in range(len(formatted_commits))[::-1]:
    print(f"{len(formatted_commits) - i}/{len(formatted_commits)}")
    run(
        ["git", "push", "--force", "upstream", f"formatted-master~{i}:formatted-master"]
    )

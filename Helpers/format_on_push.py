import sys
import subprocess


def run(lst):
    print(" ".join(lst))
    return subprocess.run(
        lst, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )


proc = run(["git", "branch", "--show-current"])
branch_name = proc.stdout.splitlines()[0]
if branch_name != "master":
    sys.exit()

run(["git", "fetch", "origin", "master"])
run(["git", "fetch", "origin", "formatted-master"])

run(["pip", "install", "black"])

run(["git", "config", "--global", "user.name", "creator1creeper1"])
run(["git", "config", "--global", "user.email", "creator1creeper1@airmail.cc"])

proc = run(["git", "rev-parse", "origin/master"])
current_master_commit = proc.stdout.splitlines()[0]
proc = run(["git", "rev-parse", "origin/formatted-master"])
current_formatted_master_commit = proc.stdout.splitlines()[0]
run(["git", "switch", "--detach", current_master_commit])
run(["git", "reset", "--mixed", current_formatted_master_commit])
run(["black", "."])
run(["git", "add", "*"])
run(["git", "commit", f"--reuse-message={current_master_commit}"])
proc = run(["git", "rev-parse", "HEAD"])
new_formatted_master_commit = proc.stdout.splitlines()[0]

run(
    [
        "git",
        "push",
        "--force",
        "origin",
        f"{new_formatted_master_commit}:formatted-master",
    ]
)

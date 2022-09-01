import subprocess


class TestType:
    Broken = 0
    Silent = 1
    Visible = 2


with open("visible.runconfig", "rt") as f_in:
    visible_tests = f_in.read().split("\n")

with open("silent.runconfig", "rt") as f_in:
    silent_tests = f_in.read().split("\n")

with open("broken.runconfig", "rt") as f_in:
    broken_tests = f_in.read().split("\n")

tests = []
for (test_type, lst) in [
    (TestType.Visible, visible_tests),
    (TestType.Silent, silent_tests),
    (TestType.Broken, broken_tests),
]:
    tests += [(test_type, test_name) for test_name in lst]

for (test_type, test_name) in tests:
    if test_name == "":
        continue
    if test_type == TestType.Broken:
        continue
    print(f"Running test: {test_name}")
    proc = subprocess.run(
        ["python", "bracketless.py", f"Tests/{test_name}.br"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    with open(f"Tests/{test_name}.out") as f:
        expected_output = f.read()
    did_succeed = proc.returncode == 0 and proc.stdout == expected_output
    if did_succeed:
        print("Success.")
    else:
        print("Failed.")
    if test_type == TestType.Visible or not did_succeed:
        print("stderr:")
        print(proc.stderr)
        print("stdout:")
        print(proc.stdout)
    if not did_succeed:
        raise Exception(f"Test {test_name} failed")
print("All tests succeeded!")

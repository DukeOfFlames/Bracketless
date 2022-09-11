import subprocess
import json


def flatten_list(lst):
    return [inner for outer in lst for inner in outer]


class TestType:
    Broken = 0
    Silent = 1
    Visible = 2


with open('runconfig.json', 'r') as f:
    tests = json.load(f)

tests = flatten_list([[({'visible': TestType.Visible, 'silent': TestType.Silent, 'broken': TestType.Broken}[test_type], test_name) for test_name in tests[test_type]] for test_type in tests.keys()])

for (test_type, test_name) in tests:
    if test_name == '':
        continue
    if test_type == TestType.Broken:
        continue
    print(f"Running test: {test_name}")
    proc = subprocess.run(["python", "bracketless.py", f"Tests/{test_name}.br"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, encoding="utf-8")
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

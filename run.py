import subprocess


class TestType:
    Broken = 0
    Silent = 1
    Visible = 2


visible_tests = []

silent_tests = [
    "double_negation",
    "list_indexing",
    "assignment",
    "scope_test",
    "comments",
    "functions",
    "declarations",
    "dot",
    "factorial",
    "drucke",
    "abc",
    "builtin_functions",
    "negative_numbers",
    "multiple_parameters",
]

broken_tests = [
    "if_test",
    "type_assignment",
    "brack",
    "for_test",
    "try",
    "library_imports",
]

tests = []
for (test_type, lst) in [
    (TestType.Visible, visible_tests),
    (TestType.Silent, silent_tests),
    (TestType.Broken, broken_tests),
]:
    tests += [(test_type, test_name) for test_name in lst]

for (test_type, test_name) in tests:
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
    if test_type == TestType.Silent:
        if did_succeed:
            print("Success.")
        else:
            print("Failed.")
            raise Exception(f"Test {test_name} failed")
    elif test_type == TestType.Visible:
        if did_succeed:
            print("Success.")
            print("stdout:")
            print(proc.stdout)
        else:
            print("Failed.")
            print("stderr:")
            print(proc.stderr)
            print("stdout:")
            print(proc.stdout)
            raise Exception(f"Test {test_name} failed")
    else:
        raise Exception
print("All tests succeeded!")

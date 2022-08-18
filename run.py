import subprocess

class TestType:
    Broken = 0
    Silent = 1
    Visible = 2

tests = [
    (
        TestType.Broken,
        "if_test",
    ),
    (
        TestType.Broken,
        "type_assignment",
    ),
    (
        TestType.Broken,
        "brack",
    ),
    (
        TestType.Broken,
        "for_test",
    ),
    (
        TestType.Broken,
        "try",
    ),
    (
        TestType.Silent,
        "assignment",
    ),
    (
        TestType.Silent,
        "scope_test",
    ),
    (
        TestType.Silent,
        "comments",
    ),
    (
        TestType.Silent,
        "functions",
    ),
    (
        TestType.Silent,
        "declarations",
    ),
    (
        TestType.Silent,
        "dot",
    ),
    (
        TestType.Silent,
        "factorial",
    ),
    (
        TestType.Silent,
        "drucke",
    ),
    (
        TestType.Silent,
        "abc",
    ),
    (
        TestType.Silent,
        "builtin_functions",
    ),
    (
        TestType.Silent,
        "negative_numbers",
    ),
    (
        TestType.Silent,
        "multiple_parameters",
    ),
    (
        TestType.Broken,
        "library_imports"
    )
]

for (test_type, test_name) in tests:
    if test_type == TestType.Broken:
        continue
    print(f"Running test: {test_name}")
    proc = subprocess.run(["python", "bracketless.py", f"Tests/{test_name}.br"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
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

"""
. a b c = a (b c)

. . a b c d = . (a b) c d = (a b) (c d)

. . . a b c d = . (. a) b c d = (. a) (b c) d = . a (b c) d = a ((b c) d) = a (b c d)

. . . . a b c d e = . (. .) a b c d e = (. .) (a b) c d e = . . (a b) c d e = . ((a b) c) d e = . (a b c) d e = (a b c) (d e)
"""

import enum
import string


class Ref:
    def __init__(self, v):
        self.v = v


class NAryExpr:
    class Type(enum.Enum):
        List = 0
        Var = 1

    def __init__(self, type, *args):
        self.type = type
        if self.type == NAryExpr.Type.List:
            (lst,) = args
            self.lst = lst
        if self.type == NAryExpr.Type.Var:
            (num,) = args
            self.num = num

    def __repr__(self):
        if self.type == NAryExpr.Type.List:
            return "[" + " ".join([repr(elem) for elem in self.lst]) + "]"
        if self.type == NAryExpr.Type.Var:
            return string.ascii_lowercase[self.num]
        raise Exception

    def __len__(self):
        if self.type == NAryExpr.Type.List:
            return len(self.lst)
        raise Exception

    def __getitem__(self, i):
        if self.type == NAryExpr.Type.List:
            return self.lst[i]
        raise Exception

    def __add__(self, other):
        if self.type == NAryExpr.Type.List and other.type == NAryExpr.Type.List:
            return NAryExpr(NAryExpr.Type.List, self.lst + other.lst)
        raise Exception


class BinaryExpr:
    class Type(enum.Enum):
        Pair = 0
        Dot = 1
        Var = 2

    def __init__(self, type, *args):
        self.type = type
        if self.type == BinaryExpr.Type.Pair:
            lhs, rhs = args
            self.lhs = Ref(lhs)
            self.rhs = Ref(rhs)
        if self.type == BinaryExpr.Type.Var:
            (num,) = args
            self.num = num

    def __repr__(self):
        if self.type == BinaryExpr.Type.Pair:
            return f"({self.lhs.v} {self.rhs.v})"
        if self.type == BinaryExpr.Type.Dot:
            return "."
        if self.type == BinaryExpr.Type.Var:
            return string.ascii_lowercase[self.num]
        raise Exception

    def left_destructure(self):
        if self.type != BinaryExpr.Type.Pair:
            return [self]
        else:
            return self.lhs.v.left_destructure() + [self.rhs.v]

    def binary_tree_to_n_ary_tree(self):
        if self.type != BinaryExpr.Type.Pair:
            return NAryExpr(NAryExpr.Type.Var, self.num)
        else:
            left_tree = self.lhs.v.binary_tree_to_n_ary_tree()
            right_tree = self.rhs.v.binary_tree_to_n_ary_tree()
            if left_tree.type == NAryExpr.Type.Var:
                left_tree = NAryExpr(NAryExpr.Type.List, [left_tree])
            right_tree = NAryExpr(NAryExpr.Type.List, [right_tree])
            return left_tree + right_tree

    def leftmost_value(self):
        if self.type != BinaryExpr.Type.Pair:
            return self
        else:
            return self.lhs.v.leftmost_value()

    def left_depth(self):
        if self.type != BinaryExpr.Type.Pair:
            return 1
        else:
            return self.lhs.v.left_depth() + 1


def generate_initial_dots(n):
    if n == 1:
        return BinaryExpr(BinaryExpr.Type.Dot)
    else:
        return BinaryExpr(
            BinaryExpr.Type.Pair,
            generate_initial_dots(n - 1),
            BinaryExpr(BinaryExpr.Type.Dot),
        )


def expand_dots_once(tree, next_var_num):
    depth = tree.v.left_depth()
    while tree.v.left_depth() < 4:
        new_var = BinaryExpr(BinaryExpr.Type.Var, next_var_num.v)
        next_var_num.v += 1
        tree.v = BinaryExpr(BinaryExpr.Type.Pair, tree.v, new_var)

    expr = tree
    while expr.v.left_depth() > 4:
        expr = expr.v.lhs
    dot, first, second, third = expr.v.left_destructure()
    if dot.type != BinaryExpr.Type.Dot:
        raise Exception
    expr.v = BinaryExpr(
        BinaryExpr.Type.Pair, first, BinaryExpr(BinaryExpr.Type.Pair, second, third)
    )


def expand_dots(tree):
    tree = Ref(tree)
    next_var_num = Ref(0)
    while tree.v.leftmost_value().type == BinaryExpr.Type.Dot:
        expand_dots_once(tree, next_var_num)
    return tree.v


class VarType(enum.Enum):
    Value = 0
    UnaryFunction = 1
    BinaryFunction = 2
    TernaryFunction = 3
    QuartaryFunction = 4


def calculate_type_for_var(type_for_var, n_ary_tree):
    for i in range(len(n_ary_tree)):
        elem = n_ary_tree[i]
        if elem.type == NAryExpr.Type.List:
            calculate_type_for_var(type_for_var, elem)
        else:
            if i == 0:
                type_for_var[elem.num] = {
                    1: VarType.UnaryFunction,
                    2: VarType.BinaryFunction,
                    3: VarType.TernaryFunction,
                    4: VarType.QuartaryFunction,
                }[len(n_ary_tree) - 1]
            else:
                type_for_var[elem.num] = VarType.Value


def calculate_value_for_var(value_for_var, type_for_var):
    counter = 3
    for var_num in range(max(type_for_var.keys()) + 1):
        var_type = type_for_var[var_num]
        if var_type == VarType.Value:
            value_for_var[var_num] = counter
            counter += 1
        if var_type == VarType.UnaryFunction:
            value_for_var[var_num] = "square"
        if var_type == VarType.BinaryFunction:
            value_for_var[var_num] = "add"
        if var_type == VarType.TernaryFunction:
            value_for_var[var_num] = "permute_3"
        if var_type == VarType.QuartaryFunction:
            value_for_var[var_num] = "permute_4"


def generate_code(value_for_var, n):
    call_str = ""
    call_str += "{dot}" * n
    string_value_for_var = {key: f"{{{value}}}" for key, value in value_for_var.items()}
    counter = 3
    for var_num in range(max(string_value_for_var.keys()) + 1):
        var_string_value = string_value_for_var[var_num]
        call_str += var_string_value
    return (
        """START
°dot = fn {f} { -> fn {g} { -> f . g } }
°square = fn {x} { -> x ^ 2 }
°add = fn {x} { -> fn {y} { -> x + y } }
°permute_3 = fn {x} { -> fn {y} { -> fn {z} { -> {y - x} * {z - y} * {z - x} } } }
°permute_4 = fn {x} { -> fn {y} { -> fn {z} { -> fn {w} { -> {y - x} * {z - y} * {z - x} * {w - z} * {w - y} * {w - x} } } } }
"""
        + "°res = "
        + call_str
        + """
#drucke{res}
END"""
    )


def square(x):
    return x**2


def add(x, y):
    return x + y


def permute_3(x, y, z):
    return (y - x) * (z - y) * (z - x)


def permute_4(x, y, z, w):
    return (y - x) * (z - y) * (z - x) * (w - z) * (w - y) * (w - x)


def calculate_output(python_value_for_var, tree):
    if tree.type == NAryExpr.Type.List:
        return calculate_output(python_value_for_var, tree.lst[0])(
            *[calculate_output(python_value_for_var, elem) for elem in tree.lst[1:]]
        )
    else:
        return python_value_for_var[tree.num]


def generate_output(value_for_var, tree):
    python_value_for_var = {
        key: {int: (lambda v: v), str: (lambda v: eval(v))}[type(value)](value)
        for key, value in value_for_var.items()
    }
    output = calculate_output(python_value_for_var, tree)
    return f"{output}\n"


for n in range(1, 11):
    print(n)
    dot_tree = generate_initial_dots(n)
    print(dot_tree)
    expanded_tree = expand_dots(dot_tree)
    print(expanded_tree)
    n_ary_tree = expanded_tree.binary_tree_to_n_ary_tree()
    print(n_ary_tree)
    type_for_var = dict()
    calculate_type_for_var(type_for_var, n_ary_tree)
    value_for_var = dict()
    calculate_value_for_var(value_for_var, type_for_var)
    code = generate_code(value_for_var, n)
    print(code)
    output = generate_output(value_for_var, n_ary_tree)
    print(output)
    print("")
    with open(f"..\\Tests\\complex_dot_n\\{n}.br", "w", encoding="utf-8") as f:
        f.write(code)
    with open(f"..\\Tests\\complex_dot_n\\{n}.out", "w", encoding="utf-8") as f:
        f.write(output)

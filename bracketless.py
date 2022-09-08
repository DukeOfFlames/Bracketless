# fn: function
# cl: class
# {}
# l = {1, 2, 3} -> l{0} = 1
#
# fn x{a, b}
# {
#   res = a.lol{}
#   > b{res}
# }
# i
# !: factorial
# ?: umgekehrtes factorial
# (.): f . g wie in Haskell
# ^: power
# ==
# %
# § -> kommentar
# §§§ -> multi-line kommentar
# ; um zeilen zu beenden muss noch diskutiert werden
#
# *, /, **,
# { a = {1,2,3} }
# \x ->
# float{}
# while

import string
import sys
import math
import enum
import dis


def flatten_list(lst):
    return [inner for outer in lst for inner in outer]


def factorial(n):
    res = 1
    while n > 0:
        res *= n
        n -= 1
    return res


# https://www.desmos.com/calculator/3y4mi46f1j
# https://oeis.org/A030169
def factorial_approximation(f):
    res = 1.0
    while f > 1:
        res *= f
        f -= 1
    coefs = [
        1.0,
        -0.5717359821489323,
        0.9364531044181281,
        -0.6892160181080689,
        0.4597437410503836,
        -0.15662271468032285,
        0.016194354022299642,
        0.005183515446512647,
    ]
    res *= sum([coefs[exp] * f**exp for exp in range(len(coefs))])
    return res


def inverse_factorial_approximation(f):
    pow = 0
    while factorial_approximation(2**pow) < f:
        pow += 1
    pow -= 1
    res = 2**pow
    while (
        res + 2**pow != res
    ):  # while `2**pow` still has an effect when added to `res`
        while factorial_approximation(res + 2**pow) < f:
            res += 2**pow
        pow -= 1
    return res


def inverse_factorial(f):
    return inverse_factorial_approximation(f)


def python_value_from_bracketless_value(bracketless_value):
    if bracketless_value.type == InterpreterNode.Type.Integer:
        return bracketless_value.value
    if bracketless_value.type == InterpreterNode.Type.Float:
        return bracketless_value.value
    if bracketless_value.type == InterpreterNode.Type.Boolean:
        return bracketless_value.value
    if bracketless_value.type == InterpreterNode.Type.Function:
        bracketless_func = bracketless_value

        def call(*python_params):
            return python_value_from_bracketless_value(
                ParserNode(
                    ParserNode.Type.FunctionCallOrListIndexing,
                    (
                        ParserNode(
                            ParserNode.Type.InternalInterpreterNode, bracketless_func
                        ),
                        [
                            ParserNode(
                                ParserNode.Type.InternalInterpreterNode,
                                bracketless_value_from_python_value(python_param),
                            )
                            for python_param in python_params
                        ],
                    ),
                ).interpret(None)
            )

        return call
    if bracketless_value.type == InterpreterNode.Type.List:
        bracketless_list = bracketless_value
        return [
            python_value_from_bracketless_value(elem) for elem in bracketless_list.value
        ]
    debug_print(rich_repr(bracketless_value))
    raise Exception


def bracketless_value_from_python_value(python_value):
    if type(python_value) == int:
        return InterpreterNode(InterpreterNode.Type.Integer, python_value)
    if type(python_value) == float:
        return InterpreterNode(InterpreterNode.Type.Float, python_value)
    if type(python_value) == bool:
        return InterpreterNode(InterpreterNode.Type.Boolean, python_value)
    if hasattr(python_value, "__call__"):
        python_func = python_value

        def call(current_scope, bracketless_params):
            return bracketless_value_from_python_value(
                python_func(
                    *[
                        python_value_from_bracketless_value(bracketless_param)
                        for bracketless_param in bracketless_params
                    ]
                )
            )

        return InterpreterNode(
            InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": call}
        )
    if hasattr(python_value, "__iter__"):
        python_list = python_value
        return InterpreterNode(
            InterpreterNode.Type.List,
            [bracketless_value_from_python_value(elem) for elem in python_list],
        )
    debug_print(rich_repr(python_value))
    raise Exception


class RichRepr:
    def __init__(self, lst):
        self.lst = lst

    def concatenate(lst):
        return RichRepr(
            [
                (line, indentation)
                for rich_repr in lst
                for (line, indentation) in rich_repr.lst
            ]
        )

    def from_str(s):
        return RichRepr([(s, 0)])

    def __add__(self, other):
        return RichRepr(self.lst + other.lst)

    def indent(self):
        return RichRepr([(line, indentation + 1) for (line, indentation) in self.lst])

    def from_any(v):
        if type(v) in [str, int, bool, float]:
            return RichRepr.from_str(repr(v))
        if type(v) == list:
            return (
                RichRepr.from_str("[")
                + RichRepr.concatenate([RichRepr.from_any(elem) for elem in v]).indent()
                + RichRepr.from_str("]")
            )
        if type(v) == tuple:
            return (
                RichRepr.from_str("(")
                + RichRepr.concatenate([RichRepr.from_any(elem) for elem in v]).indent()
                + RichRepr.from_str(")")
            )
        if type(v) == dict:
            return (
                RichRepr.from_str("{")
                + RichRepr.concatenate(
                    [
                        RichRepr.from_str(f"{key}:") + RichRepr.from_any(value).indent()
                        for key, value in v.items()
                    ]
                ).indent()
                + RichRepr.from_str("}")
            )
        if type(v) in [ParserNode.Type, InterpreterNode.Type]:
            return RichRepr.from_str(
                {
                    ParserNode.Type: "ParserNode",
                    InterpreterNode.Type: "InterpreterNode",
                }[type(v)]
                + f".{v.name}"
            )
        if type(v) in [ParserNode, InterpreterNode]:
            return RichRepr.from_any(v.type) + RichRepr.from_any(v.value).indent()
        if type(v) == Scope:
            return (
                RichRepr.from_str("Scope:")
                + (
                    RichRepr.from_str("Variables:")
                    + RichRepr.from_any(
                        [name for name, value in v.vars.items()]
                    ).indent()
                ).indent()
                + (
                    RichRepr.from_str("Parent Scope:")
                    + RichRepr.from_any(v.parent_scope).indent()
                ).indent()
            )
        if type(v) == TopScope:
            return RichRepr.from_str("TopScope")
        if type(v) == FunctionType:
            return RichRepr.from_str(str(v))
        if hasattr(v, "__call__"):
            res = RichRepr.from_str("Python Function:")
            res += (
                RichRepr.from_str("Name:") + RichRepr.from_str(v.__name__).indent()
            ).indent()
            if hasattr(v, "__code__"):
                res += (
                    RichRepr.from_str("Captured Variables:")
                    + RichRepr.from_any(
                        {
                            name: value
                            for name, value in zip(
                                v.__code__.co_freevars,
                                [cell.cell_contents for cell in v.__closure__],
                            )
                        }
                    ).indent()
                ).indent()
                res += (
                    RichRepr.from_str("Code:")
                    + RichRepr.concatenate(
                        [
                            RichRepr.from_str(inst.opname)
                            for inst in dis.get_instructions(v)
                        ]
                    ).indent()
                ).indent()
            else:
                res += (
                    RichRepr.from_str("Code:") + RichRepr.from_str("<builtin>").indent()
                ).indent()
            return res
        raise Exception(f"Could not format value of type {type(v)}")

    def string(self):
        return "\n".join(
            ["  " * indentation + line for (line, indentation) in self.lst]
        )


def rich_repr(v):
    return RichRepr.from_any(v).string()


def debug_print(s):
    __print__(s, file=sys.stderr)


def output_print(s):
    __print__(s, file=sys.stdout)


__print__ = print
print = None

language_name = "Bracketless"


@enum.unique
class FunctionType(enum.Enum):
    External = 0
    Internal = 1


class TopScope:
    def __init__(self):
        pass

    def get_variable(self, name):
        raise Exception(f"Could not find variable named {repr(name)}")

    def define_variable(self, name, value):
        raise Exception(f"Could not define variable named {repr(name)}")

    def set_variable(self, name, value):
        raise Exception(f"Could not find variable named {repr(name)}")


class Scope:
    def __init__(self, parent_scope):
        self.vars = dict()
        self.parent_scope = parent_scope

    def get_variable(self, name):
        if name in self.vars.keys():
            return self.vars[name]
        else:
            return self.parent_scope.get_variable(name)

    def define_variable(self, name, value):
        if name in self.vars.keys():
            raise Exception(f"Variable {repr(name)} is already defined")
        self.vars[name] = value

    def set_variable(self, name, value):
        if name in self.vars.keys():
            self.vars[name] = value
        else:
            self.parent_scope.set_variable(name, value)


class Return(Exception):
    def __init__(self, return_value):
        self.return_value = return_value


class ParserNode:
    @enum.unique
    class Type(enum.Enum):
        Identifier = 0
        Integer = 1
        PostfixOperator = 2
        InfixOperator = 3
        OpeningCurly = 4
        ClosingCurly = 5
        Start = 6
        End = 7
        Block = 8
        Comma = 10
        PrefixOperator = 11
        List = 13
        Assignment = 14
        String = 15
        ConditionalExpression = 16  # WIP
        StatementKeyword = 17
        Function = 18
        ForLoop = 19  # WIP
        WhileLoop = 20  # WIP
        Class = 21  # WIP
        Boolean = 22
        FunctionCallOrListIndexing = 23
        PrefixOperation = 24
        PostfixOperation = 25
        InfixOperation = 26
        Colon = 27
        Type = 28
        DeclarationAssignment = 30
        BuiltinIdentifier = 31
        Float = 32
        ForLoopExpression = 33
        Try = 34
        Hexadecimal = 35
        Binary = 36
        Octal = 37
        IfStatement = 38
        IfElseStatement = 39
        InternalInterpreterNode = 40
        ForStatement = 41
        WhileStatement = 42
        LibImportStatement = 43
        PyLibImportStatement = 44

        def is_expression(self):
            return self in [
                ParserNode.Type.Identifier,
                ParserNode.Type.Integer,
                ParserNode.Type.Block,
                ParserNode.Type.List,
                ParserNode.Type.Assignment,
                ParserNode.Type.String,
                ParserNode.Type.ConditionalExpression,
                ParserNode.Type.Function,
                ParserNode.Type.Class,
                ParserNode.Type.Boolean,
                ParserNode.Type.FunctionCallOrListIndexing,
                ParserNode.Type.PrefixOperation,
                ParserNode.Type.PostfixOperation,
                ParserNode.Type.InfixOperation,
                ParserNode.Type.DeclarationAssignment,
                ParserNode.Type.BuiltinIdentifier,
                ParserNode.Type.Float,
            ]

        def is_iterable(self):
            return self in [ParserNode.Type.String, ParserNode.Type.List]

        def is_number(self):
            return self in [ParserNode.Type.Integer, ParserNode.Type.Float]

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        raise Exception

    def interpret(self, current_scope):

        # debug_print(rich_repr(current_scope))

        if self.type == ParserNode.Type.Block:
            if len(self.value) != 1:
                # The scope of a block is always simply a child of the scope the block is being interpreted in
                block_scope = Scope(current_scope)
                for thing in self.value:
                    thing.interpret(block_scope)
                debug_print("Environment at end of block:")
                debug_print(rich_repr(block_scope))
                return None
            else:
                return self.value[0].interpret(current_scope)

        if self.type == ParserNode.Type.Assignment:
            name = self.value[0]
            value = self.value[1].interpret(current_scope)
            current_scope.set_variable(name, value)
            return value

        if self.type == ParserNode.Type.DeclarationAssignment:
            name = self.value[0]
            value = self.value[1].interpret(current_scope)
            current_scope.define_variable(name, value)
            return value

        if self.type == ParserNode.Type.FunctionCallOrListIndexing:
            func_or_list_expr = self.value[0]
            param_values = [value.interpret(current_scope) for value in self.value[1]]
            func_or_list = func_or_list_expr.interpret(current_scope)
            if func_or_list.type == InterpreterNode.Type.List:
                lst = func_or_list
                lst = lst.value
                if len(param_values) != 1:
                    raise Exception
                index = param_values[0]
                if index.type != InterpreterNode.Type.Integer:
                    raise Exception
                index = index.value
                return lst[index]
            if func_or_list.type == InterpreterNode.Type.Function:
                func = func_or_list
                func_param_values = param_values
                func = func.value
                if func["type"] == FunctionType.External:
                    func_body = func["body"]
                    func_param_names = [name for (name, type) in func["param_names"]]
                    # The scope of a function is a child of the scope the function was defined in
                    func_scope = Scope(func["parent_scope"])
                    if len(func_param_names) != len(func_param_values):
                        raise Exception
                    for i in range(len(func_param_names)):
                        func_scope.define_variable(
                            func_param_names[i], func_param_values[i]
                        )
                    try:
                        func_body.interpret(func_scope)
                    except Return as r:
                        return_value = r.return_value
                    else:
                        return_value = None
                    return return_value
                elif func["type"] == FunctionType.Internal:
                    func_body = func["body"]
                    return func_body(current_scope, func_param_values)
                else:
                    raise Exception
            debug_print(rich_repr(func_or_list_expr))
            debug_print(rich_repr(func_or_list))
            debug_print(type(func_or_list))
            raise Exception(
                f"Cannot interpret FunctionCallOrListIndexing because {func_or_list} is neither a function nor a list"
            )

        if self.type == ParserNode.Type.Class:
            class_name = self.value[0]
            class_functions = [
                value.interpret(current_scope) for value in self.value[1]
            ]
            class_ = current_scope.get_variable(class_name).value

        if self.type == ParserNode.Type.PrefixOperation:
            op = self.value[0]
            v = self.value[1].interpret(current_scope)
            if op == "->":
                raise Return(v)
            if op == "-":
                if v.type == InterpreterNode.Type.Integer:
                    return InterpreterNode(InterpreterNode.Type.Integer, -v.value)
                v_as_float = v.convert_to_float()
                if v_as_float != None:
                    return InterpreterNode(InterpreterNode.Type.Float, -v.value)
            if op == "not":
                if v.type == InterpreterNode.Type.Boolean:
                    return InterpreterNode(InterpreterNode.Type.Boolean, not v.value)
            debug_print(rich_repr(self.value))
            raise Exception(f"Could not interpret PrefixOperation")

        if self.type == ParserNode.Type.PostfixOperation:
            v = self.value[0].interpret(current_scope)
            op = self.value[1]
            if op == "!":
                if v.type == InterpreterNode.Type.Integer:
                    return InterpreterNode(
                        InterpreterNode.Type.Integer, factorial(v.value)
                    )
            if op == "?":
                v_as_float = v.convert_to_float()
                if v_as_float != None:
                    return InterpreterNode(
                        InterpreterNode.Type.Float, inverse_factorial(v_as_float.value)
                    )
            raise Exception(f"Could not interpret PostfixOperation with {self.value}")

        if self.type == ParserNode.Type.InfixOperation and self.value[1] == ".":
            obj = self.value[0].interpret(current_scope)
            attr = self.value[2]
            if attr.type != ParserNode.Type.Identifier:
                raise Exception
            if obj.type == InterpreterNode.Type.PyLib:
                return obj.lookup(attr.value)

        if self.type == ParserNode.Type.InfixOperation:
            lhs = self.value[0].interpret(current_scope)
            op = self.value[1]
            rhs = self.value[2].interpret(current_scope)
            # Numerical operators that allow both integers and floats
            if op in ["+", "-", "*", "^", "//"]:
                func = {
                    "+": (lambda x, y: x + y),
                    "-": (lambda x, y: x - y),
                    "*": (lambda x, y: x * y),
                    "^": (lambda x, y: x**y),
                    "//": (lambda x, y: x ** (1 / y)),
                }[op]
                if (
                    lhs.type == InterpreterNode.Type.Integer
                    and rhs.type == InterpreterNode.Type.Integer
                ):
                    return InterpreterNode(
                        InterpreterNode.Type.Integer, func(lhs.value, rhs.value)
                    )
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return InterpreterNode(
                        InterpreterNode.Type.Float,
                        func(lhs_as_float.value, rhs_as_float.value),
                    )
            # Numerical operators that allow only floats
            if op == "/":
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return InterpreterNode(
                        InterpreterNode.Type.Float,
                        lhs_as_float.value / rhs_as_float.value,
                    )
            # Numerical operators that allow only integers
            if op == "%":
                if (
                    lhs.type == InterpreterNode.Type.Integer
                    and rhs.type == InterpreterNode.Type.Integer
                ):
                    return InterpreterNode(
                        InterpreterNode.Type.Integer, lhs.value % rhs.value
                    )
            # Numerical operators that return a boolean and allow both integers and floats
            if op in ["==", ">"]:
                func = {"==": (lambda x, y: x == y), ">": (lambda x, y: x > y)}[op]
                if (
                    lhs.type == InterpreterNode.Type.Integer
                    and rhs.type == InterpreterNode.Type.Integer
                ):
                    return InterpreterNode(
                        InterpreterNode.Type.Boolean, func(lhs.value, rhs.value)
                    )
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return InterpreterNode(
                        InterpreterNode.Type.Boolean,
                        func(lhs_as_float.value, rhs_as_float.value),
                    )
            if op == ".":
                if (
                    lhs.type == InterpreterNode.Type.Function
                    and rhs.type == InterpreterNode.Type.Function
                ):

                    def combined_func(current_scope, params):
                        if len(params) != 1:
                            raise Exception
                        param = params[0]
                        return ParserNode(
                            ParserNode.Type.FunctionCallOrListIndexing,
                            (
                                ParserNode(
                                    ParserNode.Type.InternalInterpreterNode, lhs
                                ),
                                [
                                    ParserNode(
                                        ParserNode.Type.FunctionCallOrListIndexing,
                                        (
                                            ParserNode(
                                                ParserNode.Type.InternalInterpreterNode,
                                                rhs,
                                            ),
                                            [
                                                ParserNode(
                                                    ParserNode.Type.InternalInterpreterNode,
                                                    param,
                                                )
                                            ],
                                        ),
                                    )
                                ],
                            ),
                        ).interpret(None)

                    return InterpreterNode(
                        InterpreterNode.Type.Function,
                        {"type": FunctionType.Internal, "body": combined_func},
                    )
            debug_print(rich_repr(lhs))
            debug_print(rich_repr(op))
            debug_print(rich_repr(rhs))
            raise Exception(f"Could not interpret InfixOperation")

        if self.type == ParserNode.Type.IfStatement:
            predicate = self.value[0].interpret(current_scope)
            if predicate.type != InterpreterNode.Type.Boolean:
                raise Exception
            if predicate.value:
                consequent = self.value[1].interpret(current_scope)
            return None

        if self.type == ParserNode.Type.IfElseStatement:
            predicate = self.value[0].interpret(current_scope)
            if predicate.type != InterpreterNode.Type.Boolean:
                raise Exception
            if predicate.value:
                consequent = self.value[1].interpret(current_scope)
            else:
                alternative = self.value[2].interpret(current_scope)
            return None

        if self.type == ParserNode.Type.ForStatement:
            identifier = self.value[0]
            iterable = self.value[1]
            block = self.value[2]
            iterable = iterable.interpret(current_scope)
            if iterable.type != InterpreterNode.Type.List:
                raise Exception(
                    f"For-loop contains {iterable.type}, which is not iterable!"
                )
            for elem in iterable.value:
                loop_scope = Scope(current_scope)
                loop_scope.define_variable(identifier, elem)
                block.interpret(loop_scope)
            return None

        if self.type == ParserNode.Type.WhileStatement:
            condition = self.value[0]
            block = self.value[1]
            while True:
                should_continue = condition.interpret(current_scope)
                if should_continue.type != InterpreterNode.Type.Boolean:
                    raise Exception
                if not should_continue.value:
                    break
                block.interpret(current_scope)
            return None

        if self.type == ParserNode.Type.PyLibImportStatement:
            lib_name = self.value
            lib = __import__(lib_name)
            interpreted_self = InterpreterNode(InterpreterNode.Type.PyLib, lib)
            current_scope.define_variable(lib_name, interpreted_self)
            return interpreted_self

        if self.type == ParserNode.Type.Identifier:
            name = self.value
            return current_scope.get_variable(name)

        if self.type == ParserNode.Type.BuiltinIdentifier:
            name = self.value
            return Builtins.builtins[name]

        if self.type == ParserNode.Type.Function:
            interpreted_self = InterpreterNode(
                InterpreterNode.Type.Function,
                {
                    "type": FunctionType.External,
                    "param_names": self.value["param_names"],
                    "body": self.value["body"],
                    "parent_scope": current_scope,
                },
            )
            if "name" in self.value.keys():
                current_scope.define_variable(self.value["name"], interpreted_self)
            return interpreted_self

        if self.type == ParserNode.Type.List:
            return InterpreterNode(
                InterpreterNode.Type.List,
                [elem.interpret(current_scope) for elem in self.value],
            )

        if self.type == ParserNode.Type.String:
            return InterpreterNode(InterpreterNode.Type.String, self.value)

        if self.type == ParserNode.Type.Integer:
            return InterpreterNode(InterpreterNode.Type.Integer, self.value)

        if self.type == ParserNode.Type.Float:
            return InterpreterNode(InterpreterNode.Type.Float, self.value)

        if self.type == ParserNode.Type.Boolean:
            return InterpreterNode(InterpreterNode.Type.Boolean, self.value)

        if self.type == ParserNode.Type.InternalInterpreterNode:
            return self.value

        raise Exception(f"Could not interpret ParserNode of type {self.type}")


class InterpreterNode:
    @enum.unique
    class Type(enum.Enum):
        Integer = 1
        List = 13
        String = 15
        Function = 18
        Class = 21  # WIP
        Boolean = 22
        Float = 32
        Hexadecimal = 35
        Binary = 36
        Octal = 37
        PyLib = 38

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        raise Exception

    def representation(self):
        if self.type == InterpreterNode.Type.Integer:
            return str(self.value)
        if self.type == InterpreterNode.Type.Float:
            return str(self.value)
        if self.type == InterpreterNode.Type.String:
            return '"' + repr(self.value)[1:-1] + '"'
        if self.type == InterpreterNode.Type.Hexadecimal:
            return self.value
        if self.type == InterpreterNode.Type.Binary:
            return self.value
        if self.type == InterpreterNode.Type.List:
            return "{" + ", ".join([node.representation() for node in self.value]) + "}"
        if self.type == InterpreterNode.Type.Boolean:
            return "True" if self.value else "False"
        if self.type == InterpreterNode.Type.Function:
            return rich_repr(self)
        raise Exception

    def convert_to_float(self):
        if self.type == InterpreterNode.Type.Float:
            return self
        if self.type == InterpreterNode.Type.Integer:
            return InterpreterNode(InterpreterNode.Type.Float, self.value)
        return None

    def lookup(self, name):
        if self.type == InterpreterNode.Type.PyLib:
            return bracketless_value_from_python_value(getattr(self.value, name))
        raise Exception


class Builtins:
    builtins = dict()

    def drucke(current_scope, params):
        for param in params:
            output_print(param.representation())

    builtins["drucke"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": drucke}
    )

    def max(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        if not all([node.type == InterpreterNode.Type.Integer for node in lst.value]):
            raise Exception
        return InterpreterNode(
            InterpreterNode.Type.Integer, max([node.value for node in lst.value])
        )

    builtins["max"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": max}
    )

    def min(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        if not all([node.type == InterpreterNode.Type.Integer for node in lst.value]):
            raise Exception
        return InterpreterNode(
            InterpreterNode.Type.Integer, min([node.value for node in lst.value])
        )

    builtins["min"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": min}
    )

    def count(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        return InterpreterNode(
            InterpreterNode.Type.Integer, len([node.value for node in lst.value])
        )

    builtins["count"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": count}
    )

    def sum(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        if not all([node.type.is_number() for node in lst.value]) or all(
            [node.type == InterpreterNode.Type.Float for node in lst.value]
        ):
            raise Exception

        if not all([node.type == InterpreterNode.Type.Integer for node in lst.value]):
            l_ = [node.value for node in lst.value]
            res = 0
            for l in l_:
                res += l
            return InterpreterNode(InterpreterNode.Type.Integer, res)

        if not all([node.type == InterpreterNode.Type.Float for node in lst.value]):
            l_ = [node.value for node in lst.value]
            res = 0.0
            for l in l_:
                res += l
            return InterpreterNode(InterpreterNode.Type.Float, res)

    builtins["min"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": min}
    )

    def avg(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        if not all(
            [
                node.type in [InterpreterNode.Type.Integer, InterpreterNode.Type]
                for node in lst.value
            ]
        ):
            raise Exception

        l_ = [node.value for node in lst.value]
        return InterpreterNode(InterpreterNode.Type.Integer, sum(l_) / len(l_))

    builtins["avg"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": avg}
    )

    def hex(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.Integer:
            raise Exception

        return InterpreterNode(InterpreterNode.Type.Hexadecimal, hex(lst.value))

    builtins["hex"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": hex}
    )

    def bin(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.Integer:
            raise Exception

        return InterpreterNode(InterpreterNode.Type.Binary, bin(lst.value))

    builtins["bin"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": bin}
    )

    def oct(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.Integer:
            raise Exception

        return InterpreterNode(InterpreterNode.Type.Octal, oct(lst.value))

    def builtin_range(current_scope, params):
        if len(params) != 1:
            raise Exception
        end = params[0]
        if end.type != InterpreterNode.Type.Integer:
            raise Exception
        end = end.value
        return InterpreterNode(
            InterpreterNode.Type.List,
            [InterpreterNode(InterpreterNode.Type.Integer, i) for i in range(end)],
        )

    builtins["range"] = InterpreterNode(
        InterpreterNode.Type.Function,
        {"type": FunctionType.Internal, "body": builtin_range},
    )

    def for_each(current_scope, params):
        if len(params) != 2:
            raise Exception
        lst, func = params
        if lst.type != InterpreterNode.Type.List:
            raise Exception
        if func.type != InterpreterNode.Type.Function:
            raise Exception
        return InterpreterNode(
            InterpreterNode.Type.List,
            [
                ParserNode(
                    ParserNode.Type.FunctionCallOrListIndexing,
                    (
                        ParserNode(ParserNode.Type.InternalInterpreterNode, func),
                        [ParserNode(ParserNode.Type.InternalInterpreterNode, elem)],
                    ),
                ).interpret(None)
                for elem in lst.value
            ],
        )

    builtins["for_each"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": for_each}
    )

    def builtin_all(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if lst.type != InterpreterNode.Type.List:
            raise Exception
        lst = lst.value
        for elem in lst:
            if elem.type != InterpreterNode.Type.Boolean:
                raise Exception
        return InterpreterNode(
            InterpreterNode.Type.Boolean, all([elem.value for elem in lst])
        )

    builtins["all"] = InterpreterNode(
        InterpreterNode.Type.Function,
        {"type": FunctionType.Internal, "body": builtin_all},
    )

    def builtin_round(current_scope, params):
        if len(params) == 1:
            num = params[0]
            return InterpreterNode(InterpreterNode.Type.Integer, round(num.value))
        elif len(params) == 2:
            num = params[0]
            position = params[1]
            return InterpreterNode(
                InterpreterNode.Type.Float, round(num.value, position.value)
            )
        else:
            raise Exception

    builtins["round"] = InterpreterNode(
        InterpreterNode.Type.Function,
        {"type": FunctionType.Internal, "body": builtin_round},
    )


class Error(Exception):  # TODO: Implement in own language
    def __init__(self, error_name, details):
        self.error_name = error_name
        self.details = details


class CommittedDeadlySinError(Error):
    def __init__(self, details):
        super().__init__("You committed a deadly sin: ", details)


class WhereToStartError(Error):
    def __init__(self, details):
        self.details = details
        super().__init__(
            f"{language_name} does not know where to start: ", self.details
        )  # Error: chars  @ x y IN file


class File:
    def __init__(self, content):
        self.content = content
        self.position = 0
        self.line_counter = 0
        self.column_counter = 0
        self.prefix_operators = ["->", "°", "not"]
        self.statements = ["if", "elif", "else", "while", "for"]

        self.separators = ";:."
        self.pos1 = 0  # for saving positions
        self.virtpos = 0  # for simulating parse-progressions

    def get(self):
        return self.content[self.position]

    def slice(self, length):
        return self.content[self.position : (self.position + length)]

    def is_whitespace(self):
        return self.get() in string.whitespace

    def skip_whitespace(self):
        while self.is_whitespace():
            if self.get() == "\n":
                self.line_counter += 1
            self.position += 1

    def is_singleline_comment(self):
        return self.get() == "§"

    def skip_singleline_comment(self):
        self.position += 1
        while self.get() != "\n":
            self.position += 1
        self.position += 1

    def is_multiline_comment(self):
        return self.slice(3) == "§§§"

    def skip_multiline_comment(self):
        self.position += 3
        while self.slice(3) != "§§§":
            self.position += 1
        self.position += 3

    def is_comment(self):
        return self.is_multiline_comment() or self.is_singleline_comment()

    def skip_comment(self):
        if self.is_multiline_comment():
            self.skip_multiline_comment()
        elif self.is_singleline_comment():
            self.skip_singleline_comment()

    def skip_useless(self):
        while self.is_whitespace() or self.is_comment():
            if self.is_whitespace():
                self.skip_whitespace()
            elif self.is_comment():
                self.skip_comment()

    def totally_transform_thing_list(self, things):

        if any([thing.type == ParserNode.Type.Comma for thing in things]):
            i = 0
            list_elements = []
            while True:
                element_things = []
                while True:
                    element_things.append(things[i])
                    i += 1
                    if i == len(things):
                        break
                    if things[i].type == ParserNode.Type.Comma:
                        break
                element_things = self.repeatedly_transform_thing_list(element_things)
                if len(element_things) != 1:
                    raise Exception
                element = element_things[0]
                list_elements.append(element)
                if i == len(things):
                    break
                if things[i].type != ParserNode.Type.Comma:
                    raise Exception
                i += 1
                if i == len(things):
                    break
            return ParserNode(ParserNode.Type.List, list_elements)

        return None

    recognize_patterns_list = [
        (
            "assignment",
            [
                (lambda elem_0: elem_0.type == ParserNode.Type.Identifier),
                (
                    lambda elem_1: elem_1.type == ParserNode.Type.InfixOperator
                    and elem_1.value == "="
                ),
                (lambda elem_2: elem_2.type.is_expression()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.Assignment, (arr[0].value, arr[2])
                )
            ),
        ),
        (
            "for_loop_expression",
            [
                (lambda elem_0: True),
                (lambda elem_1: elem_1.type == ParserNode.Type.Identifier),
                (
                    lambda elem_2: elem_2.type == ParserNode.Type.InfixOperator
                    and elem_2.value == ":"
                ),
                (lambda elem_3: elem_3.type.is_iterable()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.ForLoopExpression, (arr[1].value, arr[3].value)
                )
            ),
        ),
        (
            "conditional_expression",
            [
                (lambda elem_0: elem_0.type == ParserNode.Type.StatementKeyword),
                (
                    lambda elem_1: elem_1.type
                    in [
                        ParserNode.Type.Identifier,
                        ParserNode.Type.String,
                        ParserNode.Type.Integer,
                        ParserNode.Type.List,
                        ParserNode.Type.Function,
                    ]
                ),
                (
                    lambda elem_2: elem_2.type == ParserNode.Type.InfixOperator
                    and not elem_2.value in ["==", "<", ">", ">=", "<=", "%"]
                ),
                (
                    lambda elem_3: elem_3.type
                    in [
                        ParserNode.Type.Identifier,
                        ParserNode.Type.String,
                        ParserNode.Type.Integer,
                        ParserNode.Type.List,
                        ParserNode.Type.Function,
                    ]
                ),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.ConditionalExpression,
                    (arr[0].value, arr[1].value, arr[2].value),
                )
            ),
        ),
        (
            "prefix_operation",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.PrefixOperator
                    and elem_0.value != "°"
                ),
                (lambda elem_1: elem_1.type.is_expression()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.PrefixOperation, (arr[0].value, arr[1])
                )
            ),
        ),
        (
            "prefix_operation",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.InfixOperator
                    and elem_0.value == "-"
                ),
                (lambda elem_1: elem_1.type.is_expression()),
            ],
            (lambda arr: ParserNode(ParserNode.Type.PrefixOperation, ("-", arr[1]))),
        ),
        (
            "postfix_operation",
            [
                (lambda elem_0: elem_0.type.is_expression()),
                (lambda elem_1: elem_1.type == ParserNode.Type.PostfixOperator),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.PostfixOperation, (arr[0], arr[1].value)
                )
            ),
        ),
        (
            "infix_operation",
            [
                (lambda elem_0: elem_0.type.is_expression()),
                (
                    lambda elem_1: elem_1.type == ParserNode.Type.InfixOperator
                    and elem_1.value != "="
                ),
                (lambda elem_2: elem_2.type.is_expression()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.InfixOperation, (arr[0], arr[1].value, arr[2])
                )
            ),
        ),
        (
            "declaration_assignment",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.PrefixOperator
                    and elem_0.value == "°"
                ),
                (lambda elem_1: elem_1.type == ParserNode.Type.Assignment),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.DeclarationAssignment, arr[1].value
                )
            ),
        ),
        (
            "function_call_or_list_indexing",
            [
                (lambda elem_0: elem_0.type.is_expression()),
                (
                    lambda elem_1: elem_1.type == ParserNode.Type.Block
                    and len(elem_1.value) == 0
                ),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.FunctionCallOrListIndexing, (arr[0], [])
                )
            ),
        ),
        (
            "function_call_or_list_indexing",
            [
                (lambda elem_0: elem_0.type.is_expression()),
                (
                    lambda elem_1: elem_1.type == ParserNode.Type.Block
                    and len(elem_1.value) == 1
                ),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.FunctionCallOrListIndexing,
                    (arr[0], [arr[1].value[0]]),
                )
            ),
        ),
        (
            "function_call_or_list_indexing",
            [
                (lambda elem_0: elem_0.type.is_expression()),
                (lambda elem_1: elem_1.type == ParserNode.Type.List),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.FunctionCallOrListIndexing, (arr[0], arr[1].value)
                )
            ),
        ),
        (
            "if_statement",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.StatementKeyword
                    and elem_0.value == "if"
                ),
                (lambda elem_1: elem_1.type.is_expression()),
                (lambda elem_2: elem_2.type.is_expression()),
            ],
            (lambda arr: ParserNode(ParserNode.Type.IfStatement, (arr[1], arr[2]))),
        ),
        (
            "if_else_statement",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.StatementKeyword
                    and elem_0.value == "if"
                ),
                (lambda elem_1: elem_1.type.is_expression()),
                (lambda elem_2: elem_2.type.is_expression()),
                (
                    lambda elem_3: elem_3.type == ParserNode.Type.StatementKeyword
                    and elem_3.value == "else"
                ),
                (lambda elem_4: elem_4.type.is_expression()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.IfElseStatement, (arr[1], arr[2], arr[4])
                )
            ),
        ),
        (
            "for_statement",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.StatementKeyword
                    and elem_0.value == "for"
                ),
                (lambda elem_1: elem_1.type == ParserNode.Type.Identifier),
                (lambda elem_2: elem_2.type == ParserNode.Type.Colon),
                (lambda elem_3: elem_3.type.is_expression()),
                (lambda elem_4: elem_4.type.is_expression()),
            ],
            (
                lambda arr: ParserNode(
                    ParserNode.Type.ForStatement, (arr[1].value, arr[3], arr[4])
                )
            ),
        ),
        (
            "while_statement",
            [
                (
                    lambda elem_0: elem_0.type == ParserNode.Type.StatementKeyword
                    and elem_0.value == "while"
                ),
                (lambda elem_1: elem_1.type.is_expression()),
                (lambda elem_2: elem_2.type.is_expression()),
            ],
            (lambda arr: ParserNode(ParserNode.Type.WhileStatement, (arr[1], arr[2]))),
        ),
    ]

    recognize_patterns_dict = dict()
    for (pattern_name, pattern_list, pattern_xform) in recognize_patterns_list:
        if pattern_name not in recognize_patterns_dict.keys():
            recognize_patterns_dict[pattern_name] = []
        recognize_patterns_dict[pattern_name].append((pattern_list, pattern_xform))

    def recognize_pattern(self, pattern_name, things, o):
        for pattern in self.recognize_patterns_dict[pattern_name]:
            pattern_list, pattern_xform = pattern
            if len(things) >= o + len(pattern_list) and all(
                [pattern_list[i](things[o + i]) for i in range(len(pattern_list))]
            ):
                return (
                    things[:o]
                    + [pattern_xform(things[o : (o + len(pattern_list))])]
                    + things[(o + len(pattern_list)) :],
                    True,
                )
        return things, False

    def repeatedly_transform_thing_list(self, things):
        # The order of this list is important because it dictates the precedence of different types of expressions
        recognize_list = [
            "if_else_statement",
            "if_statement",
            "for_statement",
            "while_statement",
            "function_call_or_list_indexing",
            "postfix_operation",
            "infix_operation",
            "conditional_expression",
            "prefix_operation",
            "assignment",
            "declaration_assignment",
            "for_loop_expression",
        ]
        i = 0
        while i < len(recognize_list):
            did_recognize_any = False
            for offset in range(len(things)):
                things, did_recognize = self.recognize_pattern(
                    recognize_list[i], things, offset
                )
                if did_recognize:
                    did_recognize_any = True
                    break
            if did_recognize_any:
                i = 0
            else:
                i += 1
        return things

    def parse_general_block(self, opening_type, closing_type):

        thing = self.parse_thing(no_blocks=True)
        if thing.type != opening_type:
            raise Exception(f"{thing.type} is not {opening_type}")
        things = []
        while True:
            thing = self.parse_thing()
            if thing.type == closing_type:
                break
            else:
                things.append(thing)

        final_node = self.totally_transform_thing_list(things)
        if final_node != None:
            return final_node

        things = self.repeatedly_transform_thing_list(things)
        final_node = ParserNode(ParserNode.Type.Block, things)
        return final_node

    def is_boolean(self):
        return self.slice(4) == "true" or self.slice(5) == "false"

    def parse_boolean(self):
        if self.slice(5) == "False":
            self.position += 5
            return ParserNode(ParserNode.Type.Boolean, self.slice(5))
        elif self.slice(4) == "True":
            self.position += 4
            return ParserNode(ParserNode.Type.Boolean, self.slice(4))

    def is_block(self):
        return self.get() == "{"

    def parse_block(self):
        return self.parse_general_block(
            ParserNode.Type.OpeningCurly, ParserNode.Type.ClosingCurly
        )

    def parse(self):
        # self.check_start()
        # self.skip_useless()
        # block = self.parse_block()
        # self.check_end()
        # return ParserNode()
        return self.parse_general_block(ParserNode.Type.Start, ParserNode.Type.End)

    def check_start(self):
        if not self.slice(5) == "START" in self.content:
            raise WhereToStartError("ABC")  # details = position of error

    def check_end(self):
        if self.slice(3) == "END":
            sys.exit()

    def is_separator(self):
        return self.get() in self.separators

    def is_builtin_identifier(self):
        return self.get() == "#"

    def parse_builtin_identifier(self):
        self.position += 1
        name = self.parse_identifier().value
        return ParserNode(ParserNode.Type.BuiltinIdentifier, name)

    def is_prefix_operator(self):
        return any([self.slice(len(op)) == op for op in self.prefix_operators])

    def parse_prefix_operator(self):
        for op in self.prefix_operators:
            if self.slice(len(op)) == op:
                self.position += len(op)
                return ParserNode(ParserNode.Type.PrefixOperator, op)
        raise Exception

    def is_statement(self):
        for statement in self.statements:
            if self.slice(len(statement)) == statement:
                return True

    def parse_statement(self):
        for statement in self.statements:
            if self.slice(len(statement)) == statement:
                self.position += len(statement)
                return ParserNode(ParserNode.Type.StatementKeyword, statement)

    def is_type_assignment(self):
        ...

    def return_string(self):
        return ParserNode(ParserNode.Type.String, self.string)

    def parse_thing(self, no_blocks=False):
        self.skip_useless()
        for (is_x, parse_x) in [
            (self.is_start, self.parse_start),
            (self.is_function, self.parse_function),
            (self.is_end, self.parse_end),
            (self.is_string, self.parse_string),
            (self.is_try, self.parse_try),
            (self.is_statement, self.parse_statement),
            (self.is_prefix_operator, self.parse_prefix_operator),
            (self.is_postfix_operator, self.parse_postfix_operator),
            (self.is_infix_operator, self.parse_infix_operator),
            (self.is_hex, self.parse_hex),
            (self.is_bin, self.parse_bin),
            (self.is_number, self.parse_number),
            (self.is_import_statement, self.parse_import_statement),
            (self.is_python_import_statement, self.parse_python_import_statement),
            (self.is_boolean, self.parse_boolean),
            (self.is_identifier, self.parse_identifier),
            (
                (self.is_opening_curly, self.parse_opening_curly)
                if no_blocks
                else (self.is_block, self.parse_block)
            ),
            (self.is_closing_curly, self.parse_closing_curly),
            (self.is_comma, self.parse_comma),
            (self.is_builtin_identifier, self.parse_builtin_identifier),
            (self.is_boolean, self.parse_boolean),
            (self.is_colon, self.parse_colon),
        ]:
            if is_x():
                return parse_x()
        raise Exception(
            f"Could not detect thing type at current character {repr(self.get())}"
        )

    def is_identifier(self):
        return self.get() in string.ascii_letters

    def parse_identifier(self):
        identifier = ""
        while self.get() in string.ascii_letters + "_" + string.digits:
            identifier += self.get()
            self.position += 1
        return ParserNode(ParserNode.Type.Identifier, identifier)

    # def is_integer(self):
    #     return self.get() in string.digits

    # def parse_integer(self):
    #     integer = 0
    #     while self.get() in string.digits:
    #         integer *= 10
    #         integer += int(self.get())
    #         self.position += 1
    #     return ParserNode(ParserNode.Type.Integer, integer)

    def is_hex(self):
        return (
            self.slice(2) == "0x"
            and self.content[self.position + 2] in string.digits + "AaBbCcDdEeFf"
        )

    def parse_hex(self):
        hex_number = "0x"
        self.position += 2

        while self.get() in string.digits + "AaBbCcDdEeFf":
            hex_number += self.get()
            self.position += 1

        return ParserNode(ParserNode.Type.Hexadecimal, hex_number)

    def is_oct(self):
        return self.slice(2) == "0o" and self.content[self.position + 2] in "01234567"

    def parse_oct(self):
        oct_number = "0o"
        self.position += 2

        while self.get() in "01234567":
            oct_number += self.get()
            self.position += 1

        return ParserNode(ParserNode.Type.Octal, oct_number)

    def is_bin(self):
        return self.slice(2) == "0b" and self.content[self.position + 2] in ["0", "1"]

    def parse_bin(self):
        bin_number = "0b"
        self.position += 2

        while self.get() in ["0", "1"]:
            bin_number += self.get()
            self.position += 1

        return ParserNode(ParserNode.Type.Binary, bin_number)

    def is_number(self):
        if self.get() == "." and self.content[self.position + 1] in string.digits:
            return True
        elif self.get() in string.digits + "π" or self.slice(3) in ["inf", "NaN"]:
            return True

    def parse_number(self):
        number = 0
        points = 0
        decimals = 0
        exponential = 0
        exponent = 0

        if self.slice(3) in ["NaN", "inf"]:
            self.position += 3
            return ParserNode(
                ParserNode.Type.Float, self.content[self.position - 3 : self.position]
            )

        if self.get() == "π":
            self.position += 1
            return ParserNode(ParserNode.Type.Float, math.pi)

        while self.get() in string.digits + "." and not exponential == 1:
            if points == 0:
                if self.get() in string.digits:
                    number *= 10
                    number += int(self.get())
                    self.position += 1
                if self.get() == ".":
                    points += 1
                    self.position += 1
            if points == 1:
                if self.get() in string.digits:
                    decimals += 1
                    number *= 10
                    number += int(self.get())
                    self.position += 1

            if self.slice(2) == "e+":
                exponential += 1
                self.position += 2

        while self.get() in string.digits and exponential == 1:
            exponent *= 10
            exponent += int(self.get())
            self.position += 1
            print(exponent)

        if points == 1 and decimals > 0:
            number /= 10**decimals

        if exponential == 1:
            number *= 10**exponent

        if points == 1 or exponential == 1:
            return ParserNode(ParserNode.Type.Float, number)
        else:
            return ParserNode(ParserNode.Type.Integer, number)

    def is_postfix_operator(self):
        return self.get() in ["!", "?"]

    def parse_postfix_operator(self):
        c = self.get()
        self.position += 1
        return ParserNode(ParserNode.Type.PostfixOperator, c)

    def is_infix_operator(self):
        if self.slice(3) in ["//=", "and", "not"]:
            return True
        elif self.slice(2) in [
            "//",
            "%=",
            "+=",
            "==",
            "-=",
            "*=",
            "^=",
            "==",
            "/=",
            ">=",
            "<=",
            "or",
        ]:
            return True
        elif self.get() in ["^", ">", "<", "*", "/", "=", "+", "-", ".", "%"]:
            return True

    def parse_infix_operator(self):
        if self.slice(3) in ["//=", "and", "not"]:
            s = self.slice(3)
            self.position += 3
            return ParserNode(ParserNode.Type.InfixOperator, s)
        elif self.slice(2) in [
            "//",
            "%=",
            "+=",
            "==",
            "-=",
            "*=",
            "^=",
            "==",
            "/=",
            ">=",
            "<=",
            "or",
        ]:
            s = self.slice(2)
            self.position += 2
            return ParserNode(ParserNode.Type.InfixOperator, s)
        elif self.get() in ["^", ">", "<", "*", "/", "=", "+", "-", ".", "%"]:
            c = self.get()
            self.position += 1
            return ParserNode(ParserNode.Type.InfixOperator, c)
        raise Exception

    def is_start(self):
        return self.slice(5) == "START"

    def parse_start(self):
        self.position += 5
        return ParserNode(ParserNode.Type.Start, None)

    def is_end(self):
        return self.slice(3) == "END"

    def parse_end(self):
        self.position += 3
        return ParserNode(ParserNode.Type.End, None)

    def is_opening_curly(self):
        return self.get() == "{"

    def parse_opening_curly(self):
        self.position += 1
        return ParserNode(ParserNode.Type.OpeningCurly, None)

    def is_closing_curly(self):
        return self.get() == "}"

    def parse_closing_curly(self):
        self.position += 1
        return ParserNode(ParserNode.Type.ClosingCurly, None)

    def is_type(self):
        for typename in [
            "complex",
            "int",
            "str",
            "float",
            "bin",
            "hex",
            "oct",
            "list",
            "bool",
            "dict",
        ]:
            if self.slice(len(typename)) == typename:
                return True
        return False

    def parse_type(self):
        for typename in [
            "complex",
            "int",
            "str",
            "float",
            "bin",
            "hex",
            "oct",
            "list",
            "bool",
            "dict",
        ]:
            if self.slice(len(typename)) == typename:
                self.position += len(typename)
                return ParserNode(ParserNode.Type.Type, typename)
        raise Exception

    def is_comma(self):
        return self.get() == ","

    def parse_comma(self):
        self.position += 1
        return ParserNode(ParserNode.Type.Comma, None)

    def is_import_statement(self):
        return self.slice(3) == "lib"

    def parse_import_statement(self):
        self.position += 3
        self.skip_useless()
        lib = self.parse_identifier().value
        self.skip_useless()
        return ParserNode(ParserNode.Type.LibImportStatement, lib)

    def is_python_import_statement(self):
        return self.slice(5) == "pylib"

    def parse_python_import_statement(self):
        self.position += 5
        self.skip_useless()
        lib = self.parse_identifier().value
        self.skip_useless()
        return ParserNode(ParserNode.Type.PyLibImportStatement, lib)

    def is_string(self):
        return self.get() in ['"', "'"]

    def parse_string(self):
        if self.get() not in ['"', "'"]:
            raise Exception
        quote_type = self.get()
        self.position += 1
        s = ""
        while self.get() != quote_type:
            s += self.get()
            self.position += 1
        self.position += 1
        return ParserNode(ParserNode.Type.String, s)

    def is_colon(self):
        return self.get() == ":"

    def parse_colon(self):
        self.position += 1
        return ParserNode(ParserNode.Type.Colon, ":")

    def is_try(self):
        return self.slice(3) == "try"

    def parse_try_keyword(self):
        if self.slice(3) != "try":
            raise Exception
        self.position += 3

    def parse_except_keyword(self):
        if self.slice(6) != "except":
            raise Exception
        self.position += 6

    def parse_error_keyword(self):
        self.errors = ["Exception", "CommittedDeadlySinError"]
        lens = []

        for error in self.errors:
            lens.append(len(error))

        for i in range(max(lens), 1, -1):
            if self.slice(i) in self.errors:
                err = self.slice(i)
                self.position += i
                return err

    def parse_try(self):
        self.parse_try_keyword()
        self.skip_useless()
        try_ = self.parse_block()
        self.skip_useless()
        self.parse_except_keyword()
        self.skip_useless()
        error = self.parse_error_keyword()
        self.skip_useless()
        except_ = self.parse_block()
        self.skip_useless()

        return ParserNode(
            ParserNode.Type.Try,
            {"try_block": try_, "error": error, "except_block": except_},
        )

    def is_class(self):
        return self.slice(2) == "cl"

    def parse_class_keyword(self):
        if self.slice(2) != "cl":
            raise Exception
        self.position += 2

    def parse_class_name(self):
        if not self.is_identifier():
            raise Exception
        return self.parse_identifier().value

    def parse_class(self):
        self.parse_class_keyword()
        self.skip_useless()
        name = self.parse_class_name()
        self.skip_useless()
        functions = []
        self.parse_opening_curly()
        while not self.is_closing_curly():
            self.parse_function()
        self.parse_closing_curly()
        self.skip_useless()

        return ParserNode(
            ParserNode.Type.Class, {"name": name, "function_names": functions}
        )

    def is_function(self):
        return self.slice(2) == "fn"

    def parse_function_keyword(self):
        if self.slice(2) != "fn":
            raise Exception
        self.position += 2

    def is_function_name(self):
        return self.is_identifier()

    def parse_function_name(self):
        if not self.is_identifier():
            raise Exception
        return self.parse_identifier().value

    def parse_function_parameter_list(self):
        if not self.is_opening_curly():
            raise Exception
        self.parse_opening_curly()
        self.skip_useless()

        type = "any"
        lst = []
        while True:
            if self.is_closing_curly():
                break
            if not self.is_identifier():
                raise Exception
            name = self.parse_identifier().value
            self.skip_useless()

            if self.is_colon():
                self.parse_colon()
                self.skip_useless()
                if not self.is_type():
                    raise Exception
                type = self.parse_type()
                self.skip_useless()
                lst.append((name, type))
            else:
                lst.append((name, "any"))

            if self.is_closing_curly():
                break

            if not self.is_comma():
                raise Exception
            self.parse_comma()
            self.skip_useless()

        self.parse_closing_curly()
        self.skip_useless()

        return lst

    def parse_function_body(self):
        return self.parse_block()

    def parse_function(self):
        res = dict()
        self.parse_function_keyword()
        self.skip_useless()
        if self.is_function_name():
            res["name"] = self.parse_function_name()
            self.skip_useless()
        res["param_names"] = self.parse_function_parameter_list()
        self.skip_useless()
        res["body"] = self.parse_function_body()
        self.skip_useless()
        return ParserNode(ParserNode.Type.Function, res)

    def is_boolean(self):
        for s in ["True", "False"]:
            if self.slice(len(s)) == s:
                return True
        return False

    def parse_boolean(self):
        for s in ["True", "False"]:
            if self.slice(len(s)) == s:
                self.position += len(s)
                return ParserNode(ParserNode.Type.Boolean, s == "True")


def main(filename):
    debug_print(f"Running file: {filename}")
    with open(filename, "rt", encoding="utf-8") as f:
        contents = f.read()
    file = File(contents)
    root_node = file.parse()
    debug_print("Root ParserNode:")
    debug_print(rich_repr(root_node))
    debug_print("")
    debug_print("Interpreting...")
    root_node.interpret(TopScope())
    debug_print("")


main(sys.argv[1])

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
                    {
                        "func_or_list_ref": ParserNodeRef(
                            ParserNode(
                                ParserNode.Type.InternalInterpreterNode,
                                bracketless_func,
                            )
                        ),
                        "param_values": [
                            ParserNode(
                                ParserNode.Type.InternalInterpreterNode,
                                bracketless_value_from_python_value(python_param),
                            )
                            for python_param in python_params
                        ],
                    },
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
        if type(v) in [str, int, bool, float, type(None)]:
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
        if type(v) == ParserNodeRef:
            return (
                RichRepr.from_str("ParserNodeRef:") + RichRepr.from_any(v.node).indent()
            )
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
        if type(v) == OperatorType:
            return RichRepr.from_str(str(v))
        if type(v) == FormatStringPart:
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
        OpeningCurly = 4
        ClosingCurly = 5
        Start = 6
        End = 7
        Block = 8
        Comma = 10
        List = 13
        Assignment = 14
        String = 15
        StatementKeyword = 17
        Function = 18
        ForLoop = 19
        WhileLoop = 20
        Class = 21
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
        Operator = 45
        SwitchStatement = 46
        AssignmentOperator = 47
        DeclarationAssignmentPrefix = 48
        FormatString = 49

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        raise Exception

    def is_expression(self):
        if self.type in [
            ParserNode.Type.Identifier,
            ParserNode.Type.Integer,
            ParserNode.Type.List,
            ParserNode.Type.Assignment,
            ParserNode.Type.String,
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
            ParserNode.Type.FormatString,
        ]:
            return True
        if self.type == ParserNode.Type.Block and len(self.value) == 1:
            return True
        return False

    def is_prefix_operator(self):
        if self.type != ParserNode.Type.Operator:
            return False
        return self.value in Syntax.operators[OperatorType.Prefix]

    def is_infix_operator(self):
        if self.type != ParserNode.Type.Operator:
            return False
        return self.value in Syntax.operators[OperatorType.Infix]

    def is_postfix_operator(self):
        if self.type != ParserNode.Type.Operator:
            return False
        return self.value in Syntax.operators[OperatorType.Postfix]

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
            name = self.value["name"]
            value = self.value["value_ref"].node.interpret(current_scope)
            current_scope.set_variable(name, value)
            return value

        if self.type == ParserNode.Type.DeclarationAssignment:
            name = self.value["name"]
            value = self.value["value_ref"].node.interpret(current_scope)
            current_scope.define_variable(name, value)
            return value

        if self.type == ParserNode.Type.FunctionCallOrListIndexing:
            func_or_list_expr = self.value["func_or_list_ref"].node
            param_values = [
                value.interpret(current_scope) for value in self.value["param_values"]
            ]
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
            op = self.value["op"]
            v = self.value["value_ref"].node.interpret(current_scope)
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
            v = self.value["value_ref"].node.interpret(current_scope)
            op = self.value["op"]
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

        if self.type == ParserNode.Type.InfixOperation and self.value["op"] == ".":
            obj = self.value["lhs_ref"].node.interpret(current_scope)
            attr = self.value["rhs_ref"].node
            if attr.type != ParserNode.Type.Identifier:
                raise Exception
            if obj.type == InterpreterNode.Type.PyLib:
                return obj.lookup(attr.value)

        if self.type == ParserNode.Type.InfixOperation:
            lhs = self.value["lhs_ref"].node.interpret(current_scope)
            op = self.value["op"]
            rhs = self.value["rhs_ref"].node.interpret(current_scope)
            if op == "+":
                if (
                    lhs.type == InterpreterNode.Type.String
                    and rhs.type == InterpreterNode.Type.String
                ):
                    return InterpreterNode(
                        InterpreterNode.Type.String, lhs.value + rhs.value
                    )
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
                            {
                                "func_or_list_ref": ParserNodeRef(
                                    ParserNode(
                                        ParserNode.Type.InternalInterpreterNode, lhs
                                    )
                                ),
                                "param_values": [
                                    ParserNode(
                                        ParserNode.Type.FunctionCallOrListIndexing,
                                        {
                                            "func_or_list_ref": ParserNodeRef(
                                                ParserNode(
                                                    ParserNode.Type.InternalInterpreterNode,
                                                    rhs,
                                                )
                                            ),
                                            "param_values": [
                                                ParserNode(
                                                    ParserNode.Type.InternalInterpreterNode,
                                                    param,
                                                )
                                            ],
                                        },
                                    )
                                ],
                            },
                        ).interpret(None)

                    return InterpreterNode(
                        InterpreterNode.Type.Function,
                        {"type": FunctionType.Internal, "body": combined_func},
                    )
            if op == "and":
                if (
                    lhs.type == InterpreterNode.Type.Boolean
                    and rhs.type == InterpreterNode.Type.Boolean
                ):
                    return InterpreterNode(
                        InterpreterNode.Type.Boolean, lhs.value and rhs.value
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
                    debug_print(rich_repr(should_continue))
                    raise Exception
                if not should_continue.value:
                    break
                block.interpret(current_scope)
            return None

        if self.type == ParserNode.Type.SwitchStatement:
            value = self.value[0].interpret(current_scope)
            for case in self.value[1]:
                other_value = case["case"].interpret(current_scope)
                comparison = ParserNode(
                    ParserNode.Type.InfixOperation,
                    {
                        "lhs_ref": ParserNodeRef(
                            ParserNode(ParserNode.Type.InternalInterpreterNode, value)
                        ),
                        "op": "==",
                        "rhs_ref": ParserNodeRef(
                            ParserNode(
                                ParserNode.Type.InternalInterpreterNode, other_value
                            )
                        ),
                    },
                ).interpret(None)
                if comparison.type != InterpreterNode.Type.Boolean:
                    raise Exception
                if comparison.value:
                    block = case["block"]
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

        if self.type == ParserNode.Type.FormatString:
            part_list = self.value
            res = ""
            for part_type, part_value in part_list:
                if part_type == FormatStringPart.String:
                    res += part_value
                elif part_type == FormatStringPart.Expression:
                    res += part_value.interpret(current_scope).representation()
                else:
                    raise Exception
            return InterpreterNode(InterpreterNode.Type.String, res)

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

    def rightmost_child(self):
        if self.type == ParserNode.Type.PrefixOperation:
            return self.value["value_ref"]
        if self.type == ParserNode.Type.InfixOperation:
            return self.value["rhs_ref"]
        if self.type == ParserNode.Type.Assignment:
            return self.value["value_ref"]
        if self.type == ParserNode.Type.DeclarationAssignment:
            return self.value["value_ref"]
        else:
            return None

    def operator_precedence(self):
        if self.type == ParserNode.Type.Operator:
            return Syntax.operator_precedence[self.value]
        if self.type == ParserNode.Type.FunctionCallOrListIndexing:
            return Syntax.function_call_or_list_indexing_precedence
        debug_print(rich_repr(self))
        raise Exception

    def operation_precedence(self):
        if self.type == ParserNode.Type.FunctionCallOrListIndexing:
            return Syntax.function_call_or_list_indexing_precedence
        if self.type == ParserNode.Type.DeclarationAssignment:
            return Syntax.declaration_assignment_precedence
        if self.type == ParserNode.Type.Assignment:
            return Syntax.assignment_precedence
        if self.type == ParserNode.Type.PrefixOperation:
            return Syntax.operator_precedence[self.value["op"]]
        if self.type == ParserNode.Type.InfixOperation:
            return Syntax.operator_precedence[self.value["op"]]
        if self.type == ParserNode.Type.PostfixOperation:
            return Syntax.operator_precendece[self.value["op"]]
        debug_print(rich_repr(self))
        raise Exception


class ParserNodeRef:
    def __init__(self, node):
        self.node = node

    def incorporate_right(self, op, construct_new_node):
        rightmost_child = self.node.rightmost_child()
        if rightmost_child != None:
            if op.operator_precedence() > self.node.operation_precedence():
                self.node.rightmost_child().incorporate_right(op, construct_new_node)
                return
        self.node = construct_new_node(self.node)


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
            return "true" if self.value else "false"
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

    def builtin_string(current_scope, params):
        if len(params) != 1:
            raise Exception
        return InterpreterNode(InterpreterNode.Type.String, params[0].representation())

    builtins["string"] = InterpreterNode(
        InterpreterNode.Type.Function,
        {"type": FunctionType.Internal, "body": builtin_string},
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
        if lst.type == InterpreterNode.Type.List:
            return InterpreterNode(InterpreterNode.Type.Integer, len(lst.value))
        if lst.type == InterpreterNode.Type.String:
            return InterpreterNode(InterpreterNode.Type.Integer, len(lst.value))
        raise Exception

    builtins["count"] = InterpreterNode(
        InterpreterNode.Type.Function, {"type": FunctionType.Internal, "body": count}
    )

    def builtin_sum(current_scope, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == InterpreterNode.Type.List:
            raise Exception
        if not (
            all([node.type == InterpreterNode.Type.Integer for node in lst.value])
            or all([node.type == InterpreterNode.Type.Float for node in lst.value])
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

    builtins["sum"] = InterpreterNode(
        InterpreterNode.Type.Function,
        {"type": FunctionType.Internal, "body": builtin_sum},
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
                    {
                        "func_or_list_ref": ParserNodeRef(
                            ParserNode(ParserNode.Type.InternalInterpreterNode, func)
                        ),
                        "param_values": [
                            ParserNode(ParserNode.Type.InternalInterpreterNode, elem)
                        ],
                    },
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


class InternalError(Exception):
    pass


class UserError(Exception):
    def __init__(self, file, msg):
        self.file = file
        self.msg = msg

    def __str__(self):
        return f"Error in file {self.file.filename} on line {self.file.line_counter + 1} in column {self.file.column_counter + 1}: {self.msg}"


@enum.unique
class OperatorType(enum.Enum):
    Prefix = 0
    Infix = 1
    Postfix = 2


class Syntax:
    operators = {
        OperatorType.Prefix: ["->", "not", "-"],
        OperatorType.Infix: ["^", ">", "<", "*", "/", "+", "-", ".", "%"]
        + ["//", "%=", "+=", "==", "-=", "*=", "^=", "==", "/=", ">=", "<=", "or"]
        + ["//=", "and"],
        OperatorType.Postfix: ["!", "?"],
    }
    operator_precedence = {
        "not": 30,
        "!": 30,
        "^": 20,
        "//": 20,
        ".": 20,
        "*": 19,
        "and": 19,
        "/": 19,
        "+": 18,
        "-": 18,
        ">": 10,
        "==": 10,
        "->": 1,
    }
    function_call_or_list_indexing_precedence = 50
    declaration_assignment_precedence = 3
    assignment_precedence = 2


@enum.unique
class ParseMinimalExpressionResult(enum.Enum):
    StartNew = 0
    ExtendPrevious = 1
    Finish = 2


@enum.unique
class FormatStringPart(enum.Enum):
    String = 0
    Expression = 1


class File:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "rt", encoding="utf-8") as f:
            self.content = f.read()
        self.position = 0
        self.line_counter = 0
        self.column_counter = 0

    def get(self):
        return self.content[self.position]

    def slice(self, length):
        return self.content[self.position : (self.position + length)]

    def advance(self, n):
        if self.is_str("\n"):
            self.line_counter += 1
            self.column_counter = 0
        else:
            self.column_counter += 1
        self.position += n

    def is_str(self, s):
        return self.slice(len(s)) == s

    def is_any_str(self, lst):
        return any([self.is_str(s) for s in lst])

    def is_whitespace(self):
        return self.is_any_str(string.whitespace)

    def skip_whitespace(self):
        while self.is_whitespace():
            self.advance(1)

    def is_singleline_comment(self):
        return self.is_str("§")

    def skip_singleline_comment(self):
        self.advance(1)
        while not self.is_str("\n"):
            self.advance(1)
        self.advance(1)

    def is_multiline_comment(self):
        return self.is_str("§§§")

    def skip_multiline_comment(self):
        self.advance(3)
        while not self.is_str("§§§"):
            self.advance(1)
        self.advance(3)

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

    def parse_minimal_expression(self, is_end, prev_expr):
        if is_end():
            return ParseMinimalExpressionResult.Finish, None
        thing = self.parse_thing()
        self.skip_useless()
        # Infix operators need to be parsed before prefix operators because `x - y` should be parsed as `{x - y}` and not `{x} {- y}`
        if (
            prev_expr != None and prev_expr.node.is_expression()
        ) and thing.is_infix_operator():
            result, rhs_value = self.parse_minimal_expression(is_end, None)
            if not result == ParseMinimalExpressionResult.StartNew:
                raise Exception
            if not rhs_value.is_expression():
                raise Exception
            return ParseMinimalExpressionResult.ExtendPrevious, (
                thing,
                (
                    lambda lhs_value: ParserNode(
                        ParserNode.Type.InfixOperation,
                        {
                            "lhs_ref": ParserNodeRef(lhs_value),
                            "op": thing.value,
                            "rhs_ref": ParserNodeRef(rhs_value),
                        },
                    )
                ),
            )
        elif thing.is_prefix_operator():
            result, value = self.parse_minimal_expression(is_end, None)
            if not result == ParseMinimalExpressionResult.StartNew:
                raise Exception
            if not value.is_expression():
                raise Exception
            return ParseMinimalExpressionResult.StartNew, ParserNode(
                ParserNode.Type.PrefixOperation,
                {"op": thing.value, "value_ref": ParserNodeRef(value)},
            )
        elif thing.is_postfix_operator():
            return ParseMinimalExpressionResult.ExtendPrevious, (
                thing,
                (
                    lambda value: ParserNode(
                        ParserNode.Type.PostfixOperation,
                        {"value_ref": ParserNodeRef(value), "op": thing.value},
                    )
                ),
            )
        elif thing.type == ParserNode.Type.AssignmentOperator:
            result, value = self.parse_minimal_expression(is_end, None)
            if not result == ParseMinimalExpressionResult.StartNew:
                raise Exception
            if not value.is_expression():
                raise Exception
            return ParseMinimalExpressionResult.ExtendPrevious, (
                thing,
                (
                    lambda identifier: ParserNode(
                        ParserNode.Type.Assignment,
                        {"name": identifier.value, "value_ref": ParserNodeRef(value)},
                    )
                ),
            )
        elif thing.type == ParserNode.Type.DeclarationAssignmentPrefix:
            identifier = self.parse_thing()
            self.skip_useless()
            assignment = self.parse_thing()
            self.skip_useless()
            result, value = self.parse_minimal_expression(is_end, None)
            if not result == ParseMinimalExpressionResult.StartNew:
                raise Exception
            if not value.is_expression():
                raise Exception
            return ParseMinimalExpressionResult.StartNew, ParserNode(
                ParserNode.Type.DeclarationAssignment,
                {"name": identifier.value, "value_ref": ParserNodeRef(value)},
            )
        elif (prev_expr != None and prev_expr.node.is_expression()) and (
            (thing.type == ParserNode.Type.Block and len(thing.value) == 0)
            or (thing.type == ParserNode.Type.Block and len(thing.value) == 1)
            or (thing.type == ParserNode.Type.List)
        ):
            return ParseMinimalExpressionResult.ExtendPrevious, (
                ParserNode(ParserNode.Type.FunctionCallOrListIndexing, None),
                (
                    lambda func_or_list: ParserNode(
                        ParserNode.Type.FunctionCallOrListIndexing,
                        {
                            "func_or_list_ref": ParserNodeRef(func_or_list),
                            "param_values": thing.value,
                        },
                    )
                ),
            )
        else:
            return ParseMinimalExpressionResult.StartNew, thing

    def parse_expressions_until(self, is_end):
        exprs = []
        while True:
            result, value = self.parse_minimal_expression(
                is_end, exprs[-1] if len(exprs) >= 1 else None
            )
            if result == ParseMinimalExpressionResult.Finish:
                break
            elif result == ParseMinimalExpressionResult.StartNew:
                exprs.append(ParserNodeRef(value))
            elif result == ParseMinimalExpressionResult.ExtendPrevious:
                exprs[-1].incorporate_right(*value)
            else:
                raise Exception
        return [expr.node for expr in exprs]

    def is_block_or_list(self):
        return self.is_str("{")

    def parse_block_or_list(self):
        if not self.is_opening_curly():
            raise Exception
        self.parse_opening_curly()
        self.skip_useless()
        exprs = self.parse_expressions_until(
            lambda: self.is_comma() or self.is_closing_curly()
        )
        if self.is_closing_curly():
            self.parse_closing_curly()
            self.skip_useless()
            return ParserNode(ParserNode.Type.Block, exprs)
        else:
            self.parse_comma()
            self.skip_useless()
            list_elements = []
            if len(exprs) != 1:
                raise Exception
            list_elements.append(exprs[0])
            while not self.is_closing_curly():
                exprs = self.parse_expressions_until(
                    lambda: self.is_comma() or self.is_closing_curly()
                )
                if not self.is_closing_curly():
                    self.parse_comma()
                    self.skip_useless()
                if len(exprs) != 1:
                    debug_print(rich_repr(exprs))
                    raise Exception
                list_elements.append(exprs[0])
            self.parse_closing_curly()
            self.skip_useless()
            return ParserNode(ParserNode.Type.List, list_elements)

    def parse_file(self):
        if not self.is_start_keyword():
            raise Exception
        self.parse_start_keyword()
        self.skip_useless()
        exprs = self.parse_expressions_until(lambda: self.is_end_keyword())
        self.parse_end_keyword()
        self.skip_useless()
        return ParserNode(ParserNode.Type.Block, exprs)

    def is_builtin_identifier(self):
        return self.is_str("#")

    def parse_builtin_identifier(self):
        self.advance(1)
        name = self.parse_identifier().value
        return ParserNode(ParserNode.Type.BuiltinIdentifier, name)

    def is_operator(self):
        return self.is_any_str(flatten_list(Syntax.operators.values()))

    def parse_operator(self):
        for op in sorted(
            flatten_list(Syntax.operators.values()), key=len, reverse=True
        ):
            if self.is_str(op):
                self.advance(len(op))
                types = list(
                    filter(
                        lambda typ: op in Syntax.operators[typ], Syntax.operators.keys()
                    )
                )
                return ParserNode(ParserNode.Type.Operator, op)

    def is_type_assignment(self):
        ...

    def parse_thing(self):
        for (is_x, parse_x) in [
            (self.is_start_keyword, self.parse_start_keyword),
            (self.is_class, self.parse_class),
            (self.is_switch, self.parse_switch),
            (self.is_function, self.parse_function),
            (self.is_end_keyword, self.parse_end_keyword),
            (self.is_string, self.parse_string),
            (self.is_try, self.parse_try),
            (self.is_if_statement, self.parse_if_statement),
            (self.is_for_statement, self.parse_for_statement),
            (self.is_while_statement, self.parse_while_statement),
            (self.is_operator, self.parse_operator),
            (self.is_assignment_operator, self.parse_assignment_operator),
            (
                self.is_declaration_assignment_prefix,
                self.parse_declaration_assignment_prefix,
            ),
            (self.is_hex, self.parse_hex),
            (self.is_bin, self.parse_bin),
            (self.is_number, self.parse_number),
            (self.is_import_statement, self.parse_import_statement),
            (self.is_python_import_statement, self.parse_python_import_statement),
            (self.is_boolean, self.parse_boolean),
            (self.is_identifier, self.parse_identifier),
            (self.is_block_or_list, self.parse_block_or_list),
            (self.is_closing_curly, self.parse_closing_curly),
            (self.is_comma, self.parse_comma),
            (self.is_builtin_identifier, self.parse_builtin_identifier),
            (self.is_colon, self.parse_colon),
        ]:
            if is_x():
                return parse_x()
        raise Exception(
            f"Could not detect thing type at current character {repr(self.get())}"
        )

    def is_identifier(self):
        return self.is_any_str(string.ascii_letters)

    def parse_identifier(self):
        identifier = ""
        while self.is_any_str(string.ascii_letters + "_" + string.digits):
            identifier += self.get()
            self.advance(1)
        return ParserNode(ParserNode.Type.Identifier, identifier)

    def is_hex(self):
        return (
            self.is_str("0x")
            and self.content[self.position + 2] in string.digits + "AaBbCcDdEeFf"
        )

    def parse_hex(self):
        hex_number = "0x"
        self.advance(2)

        while self.is_any_str(string.digits + "AaBbCcDdEeFf"):
            hex_number += self.get()
            self.advance(1)

        return ParserNode(ParserNode.Type.Hexadecimal, hex_number)

    def is_oct(self):
        return self.is_str("0o") and self.content[self.position + 2] in "01234567"

    def parse_oct(self):
        oct_number = "0o"
        self.advance(2)

        while self.is_any_str("01234567"):
            oct_number += self.get()
            self.advance(1)

        return ParserNode(ParserNode.Type.Octal, oct_number)

    def is_bin(self):
        return self.is_str("0b") and self.content[self.position + 2] in ["0", "1"]

    def parse_bin(self):
        bin_number = "0b"
        self.advance(2)

        while self.is_any_str(["0", "1"]):
            bin_number += self.get()
            self.advance(1)

        return ParserNode(ParserNode.Type.Binary, bin_number)

    def is_number(self):
        if self.is_str(".") and self.content[self.position + 1] in string.digits:
            return True
        elif self.is_any_str(list(string.digits) + ["π", "inf", "NaN"]):
            return True

    def parse_number(self):
        number = 0
        points = 0
        decimals = 0
        exponential = 0
        exponent = 0

        if self.is_any_str(["NaN", "inf"]):
            self.advance(3)
            return ParserNode(
                ParserNode.Type.Float, self.content[self.position - 3 : self.position]
            )

        if self.is_str("π"):
            self.advance(1)
            return ParserNode(ParserNode.Type.Float, math.pi)

        while self.is_any_str(string.digits + ".") and not exponential == 1:
            if points == 0:
                if self.is_any_str(string.digits):
                    number *= 10
                    number += int(self.get())
                    self.advance(1)
                if self.is_str("."):
                    points += 1
                    self.advance(1)
            if points == 1:
                if self.is_any_str(string.digits):
                    decimals += 1
                    number *= 10
                    number += int(self.get())
                    self.advance(1)

            if self.is_str("e+"):
                exponential += 1
                self.advance(2)

        while self.is_any_str(string.digits) and exponential == 1:
            exponent *= 10
            exponent += int(self.get())
            self.advance(1)
            print(exponent)

        if points == 1 and decimals > 0:
            number /= 10**decimals

        if exponential == 1:
            number *= 10**exponent

        if points == 1 or exponential == 1:
            return ParserNode(ParserNode.Type.Float, number)
        else:
            return ParserNode(ParserNode.Type.Integer, number)

    def is_start_keyword(self):
        return self.is_str("START")

    def parse_start_keyword(self):
        self.advance(5)
        return ParserNode(ParserNode.Type.Start, None)

    def is_end_keyword(self):
        return self.is_str("END")

    def parse_end_keyword(self):
        self.advance(3)
        return ParserNode(ParserNode.Type.End, None)

    def is_opening_curly(self):
        return self.is_str("{")

    def parse_opening_curly(self):
        self.advance(1)
        return ParserNode(ParserNode.Type.OpeningCurly, None)

    def is_closing_curly(self):
        return self.is_str("}")

    def parse_closing_curly(self):
        self.advance(1)
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
            if self.is_str(typename):
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
            if self.is_str(typename):
                self.advance(len(typename))
                return ParserNode(ParserNode.Type.Type, typename)
        raise Exception

    def is_comma(self):
        return self.is_str(",")

    def parse_comma(self):
        self.advance(1)
        return ParserNode(ParserNode.Type.Comma, None)

    def is_import_statement(self):
        return self.is_str("lib")

    def parse_import_statement(self):
        self.advance(3)
        self.skip_useless()
        lib = self.parse_identifier().value
        self.skip_useless()
        return ParserNode(ParserNode.Type.LibImportStatement, lib)

    def is_python_import_statement(self):
        return self.is_str("pylib")

    def parse_python_import_statement(self):
        self.advance(5)
        self.skip_useless()
        lib = self.parse_identifier().value
        self.skip_useless()
        return ParserNode(ParserNode.Type.PyLibImportStatement, lib)

    def is_string(self):
        return self.is_str('"')

    def parse_string(self):
        if not self.is_str('"'):
            raise Exception
        self.advance(1)
        s = ""
        part_list = None
        while not self.is_str('"'):
            if self.is_str("'"):
                self.advance(1)
                self.skip_useless()
                if part_list == None:
                    part_list = []
                part_list.append((FormatStringPart.String, s))
                s = ""
                exprs = self.parse_expressions_until(lambda: self.is_str("'"))
                self.advance(1)
                if len(exprs) != 1:
                    raise UserError(
                        self,
                        "Multiple expressions in one pair of single quotes in format-string",
                    )
                part_list.append((FormatStringPart.Expression, exprs[0]))
            else:
                s += self.get()
                self.advance(1)
        self.advance(1)
        if part_list == None:
            return ParserNode(ParserNode.Type.String, s)
        else:
            part_list.append((FormatStringPart.String, s))
            s = ""
            return ParserNode(ParserNode.Type.FormatString, part_list)

    def is_colon(self):
        return self.is_str(":")

    def parse_colon(self):
        self.advance(1)
        return ParserNode(ParserNode.Type.Colon, ":")

    def is_try(self):
        return self.is_str("try")

    def parse_try_keyword(self):
        if not self.is_str("try"):
            raise Exception
        self.advance(3)

    def parse_except_keyword(self):
        if not self.is_str("except"):
            raise Exception
        self.advance(6)

    def parse_error_keyword(self):
        self.errors = ["Exception", "CommittedDeadlySinError"]
        lens = []

        for error in self.errors:
            lens.append(len(error))

        for i in range(max(lens), 1, -1):
            if self.slice(i) in self.errors:
                err = self.slice(i)
                self.advance(i)
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
        return self.is_str("cl")

    def parse_class_keyword(self):
        if not self.is_str("cl"):
            raise Exception
        self.advance(2)

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
        own = []
        self.parse_opening_curly()
        self.skip_useless()
        while not self.is_closing_curly():
            functions.append(self.parse_function())
        self.parse_closing_curly()
        self.skip_useless()

        return ParserNode(
            ParserNode.Type.Class,
            {"name": name, "function_names": functions, "own": own},
        )

    def is_function(self):
        return self.is_str("fn")

    def parse_function_keyword(self):
        if not self.is_str("fn"):
            raise Exception
        self.advance(2)

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
        return self.parse_block_or_list()

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
        for s in ["true", "false"]:
            if self.is_str(s):
                return True
        return False

    def parse_boolean(self):
        for s in ["true", "false"]:
            if self.is_str(s):
                self.advance(len(s))
                return ParserNode(ParserNode.Type.Boolean, s == "true")

    def is_switch(self):
        return self.is_str("switch")

    def parse_switch_keyword(self):
        if not self.is_str("switch"):
            raise Exception
        self.advance(6)

    def parse_switch(self):
        self.parse_switch_keyword()
        self.skip_useless()
        sw = self.parse_thing()
        self.skip_useless()
        cases = self.parse_case_block()
        self.skip_useless()

        return ParserNode(ParserNode.Type.SwitchStatement, (sw, cases))

    def parse_case_keyword(self):
        if not self.is_str("case"):
            raise Exception
        self.advance(4)

    def parse_case_block(self):
        cases = []
        self.parse_opening_curly()
        while not self.is_closing_curly():
            self.skip_useless()
            self.parse_case_keyword()
            self.skip_useless()
            case = self.parse_thing()
            self.skip_useless()
            block = self.parse_block_or_list()
            cases.append({"case": case, "block": block})
            self.skip_useless()

        self.parse_closing_curly()

        return cases

    def is_if_statement(self):
        return self.is_str("if")

    def parse_if_statement(self):
        if not self.is_str("if"):
            raise Exception
        self.advance(len("if"))
        self.skip_useless()
        predicate = self.parse_thing()
        self.skip_useless()
        consequent = self.parse_thing()
        self.skip_useless()
        if not self.is_str("else"):
            return ParserNode(ParserNode.Type.IfStatement, (predicate, consequent))
        else:
            self.advance(len("else"))
            self.skip_useless()
            alternative = self.parse_thing()
            self.skip_useless()
            return ParserNode(
                ParserNode.Type.IfElseStatement, (predicate, consequent, alternative)
            )

    def is_for_statement(self):
        return self.is_str("for")

    def parse_for_statement(self):
        if not self.is_str("for"):
            raise Exception
        self.advance(len("for"))
        self.skip_useless()
        identifier = self.parse_thing()
        self.skip_useless()
        if identifier.type != ParserNode.Type.Identifier:
            raise Exception
        if not self.is_colon():
            raise Exception
        self.parse_colon()
        self.skip_useless()
        iterable = self.parse_thing()
        self.skip_useless()
        block = self.parse_thing()
        self.skip_useless()
        return ParserNode(
            ParserNode.Type.ForStatement, (identifier.value, iterable, block)
        )

    def is_while_statement(self):
        return self.is_str("while")

    def parse_while_statement(self):
        if not self.is_str("while"):
            raise Exception
        self.advance(len("while"))
        self.skip_useless()
        condition = self.parse_thing()
        self.skip_useless()
        block = self.parse_thing()
        self.skip_useless()
        return ParserNode(ParserNode.Type.WhileStatement, (condition, block))

    def is_assignment_operator(self):
        return self.is_str("=")

    def parse_assignment_operator(self):
        if not self.is_str("="):
            raise Exception
        self.advance(1)
        return ParserNode(ParserNode.Type.AssignmentOperator, None)

    def is_declaration_assignment_prefix(self):
        return self.is_str("°")

    def parse_declaration_assignment_prefix(self):
        if not self.is_str("°"):
            raise Exception
        self.advance(1)
        return ParserNode(ParserNode.Type.DeclarationAssignmentPrefix, None)


def main(filename):
    debug_print(f"Running file: {filename}")
    file = File(filename)
    root_node = file.parse_file()
    debug_print("Root ParserNode:")
    debug_print(rich_repr(root_node))
    debug_print("")
    debug_print("Interpreting...")
    root_node.interpret(TopScope())
    debug_print("")


main(sys.argv[1])

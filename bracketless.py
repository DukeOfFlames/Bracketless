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
    coefs = [1.0, -0.5717359821489323, 0.9364531044181281, -0.6892160181080689, 0.4597437410503836,
             -0.15662271468032285, 0.016194354022299642, 0.005183515446512647]
    res *= sum([coefs[exp] * f ** exp for exp in range(len(coefs))])
    return res
    """
    p_x = 0.461632144968362341262659542325721328468196204006446351295988409
    p_y = 0.1143968055891112997211840994174112667920484663300965511287998341
    a = - p_y * (2 * p_x - 1) / ((p_x**2 - p_x)**2)
    b = (2 * p_x - 3 * p_x**2) / (2 * p_x - 1)
    res = f * (f - 1) * (a * (f + b))
    return 1 - res
    """
    """
    return (2 * math.pi * f)**(1/2) * (f / math.e)**f
    """


def inverse_factorial_approximation(f):
    pow = 0
    while factorial_approximation(2 ** pow) < f:
        pow += 1
    pow -= 1
    res = 2 ** pow
    while res + 2 ** pow != res:  # while `2**pow` still has an effect when added to `res`
        while factorial_approximation(res + 2 ** pow) < f:
            res += 2 ** pow
        pow -= 1
    return res


def inverse_factorial(f):
    return inverse_factorial_approximation(f)


language_name = 'Bracketless'


class NodeType:
    Identifier = 0
    Integer = 1
    PostfixOperator = 2
    InfixOperator = 3
    OpeningCurly = 4
    ClosingCurly = 5
    Start = 6
    End = 7
    Block = 8
    # InternalFunctionPrefix = 9
    Comma = 10
    PrefixOperator = 11
    # Quote = 12
    List = 13
    Assignment = 14
    String = 15
    ConditionalExpression = 16  # WIP
    Statement = 17
    Function = 18
    ForLoop = 19  # WIP
    WhileLoop = 20  # WIP
    Class = 21  # WIP
    Boolean = 22  # WIP
    FunctionCall = 23
    PrefixOperation = 24
    PostfixOperation = 25
    InfixOperation = 26
    Colon = 27
    Type = 28
    # InternalFunction = 29
    DeclarationAssignment = 30
    BuiltinIdentifier = 31
    Float = 32
    ForLoopExpression = 33
    Try = 34
    Hexadecimal = 35
    Binary = 36

    def string(node_type):
        return {
            NodeType.Identifier: "Identifier",
            NodeType.Integer: "Integer",
            NodeType.PostfixOperator: "PostfixOperator",
            NodeType.InfixOperator: "InfixOperator",
            NodeType.OpeningCurly: "OpeningCurly",
            NodeType.ClosingCurly: "ClosingCurly",
            NodeType.Start: "Start",
            NodeType.End: "End",
            NodeType.Block: "Block",
            NodeType.Comma: "Comma",
            NodeType.PrefixOperator: "PrefixOperator",
            NodeType.List: "List",
            NodeType.Assignment: "Assignment",
            NodeType.String: "String",
            NodeType.ConditionalExpression: "ConditionalExpression",
            NodeType.Statement: "Statement",
            NodeType.Function: "Function",
            NodeType.ForLoop: "ForLoop",
            NodeType.WhileLoop: "WhileLoop",
            NodeType.Class: "Class",
            NodeType.Boolean: "Boolean",
            NodeType.FunctionCall: "FunctionCall",
            NodeType.PrefixOperation: "PrefixOperation",
            NodeType.PostfixOperation: "PostfixOperation",
            NodeType.InfixOperation: "InfixOperation",
            NodeType.Colon: "Colon",
            NodeType.Type: "Type",
            NodeType.DeclarationAssignment: "DeclarationAssignment",
            NodeType.BuiltinIdentifier: "BuiltinIdentifier",
            NodeType.Float: "Float",
            NodeType.ForLoopExpression: "ForLoopExpression",
            NodeType.Try: "Try",
            NodeType.Hexadecimal: "Hexadecimal",
            NodeType.Binary: "Binary"
        }[node_type]

    def is_expression(node_type):
        return node_type in [
            NodeType.Identifier, NodeType.Integer, NodeType.Block,
            NodeType.List, NodeType.Assignment, NodeType.String,
            NodeType.ConditionalExpression, NodeType.Function, NodeType.Class,
            NodeType.Boolean, NodeType.FunctionCall, NodeType.PrefixOperation,
            NodeType.PostfixOperation, NodeType.InfixOperation,
            NodeType.DeclarationAssignment,
            NodeType.BuiltinIdentifier,
            NodeType.Float
        ]

    def is_iterable(node_type):
        return node_type in [
            NodeType.String, NodeType.List
        ]

    def is_number(node_type):
        return node_type in [
            NodeType.Integer, NodeType.Float
        ]


class FunctionType:
    External = 0
    Internal = 1


class ExecutionEnvironment:
    def __init__(self):
        # env is a list of dictionaries with all the defined variables:
        # [{'a': 2, 'l': [4, 6, 8]}, {'n': 5, 'i': 3}, {'i': 2}]
        self.env = [dict()]

    def print(self):
        print("[")
        for scope in self.env:
            print("  {")
            for (key, value) in scope.items():
                print(f"    {key}:")
                print('\n'.join(
                    ["      " + line for line in repr(value).split('\n')]))
            print("  }")
        print("]")

    def get_variable(self, name):
        # Search through all scopes and return the value of the innermost variable with a matching name
        for i in range(len(self.env))[::-1]:
            if name in self.env[i].keys():
                return self.env[i][name]
        # If no variable matches the name, raise an Error
        raise Exception(f"Could not find variable named {repr(name)}")

    def define_variable(self, name, value):
        if name in self.env[-1].keys():
            raise Exception(f"Variable {repr(name)} is already defined")
        self.env[-1][name] = value

    def set_variable(self, name, value):
        # Search through all scopes and set the value of the innermost variable with a matching name
        for i in range(len(self.env))[::-1]:
            if name in self.env[i].keys():
                self.env[i][name] = value
                return
        # If no variable matches the name, raise an Error
        raise Exception(f"Could not find variable named {repr(name)}")

    def enter_scope(self):
        self.env.append(dict())

    def leave_scope(self):
        self.env.pop()

    def __enter__(self):
        self.enter_scope()

    def __exit__(self, *args):
        self.leave_scope()


class Return(Exception):
    def __init__(self, return_value):
        self.return_value = return_value


class Node:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def repr_as_list(self, short_toggle):

        if type(self.value) in [list, tuple]:
            inline_value_repr = "["
            outline_value_repr = [
                                     (line, indentation + 4)
                                     for (line, indentation) in flatten_list([(
                    elem.repr_as_list(short_toggle) if type(elem) ==
                                                       Node else [(repr(elem), 0)]) for elem in self.value])
                                 ] + [("]", 2)]
        else:
            inline_value_repr = repr(self.value)
            outline_value_repr = []

        if short_toggle:
            return [(f"{NodeType.string(self.type)}:", 0),
                    (f"{inline_value_repr}", 2)] \
                   + outline_value_repr
        else:
            return [("Node:", 0),
                    (f"Type = {NodeType.string(self.type)}", 2),
                    (f"Value = {inline_value_repr}", 2)] \
                   + outline_value_repr

    def __format__(self, spec):
        short_toggle = False
        if spec == 's':
            short_toggle = True
        if spec == 'l':
            short_toggle = False
        return '\n'.join([
            ' ' * indentation + line
            for (line, indentation) in self.repr_as_list(short_toggle)
        ])

    def __repr__(self):
        return f"{self:s}"

    def convert_to_float(self):
        if self.type == NodeType.Float:
            return self
        if self.type == NodeType.Integer:
            return Node(NodeType.Float, self.value)
        return None

    def interpret(self, execution_environment):

        # execution_environment.print()

        if self.type == NodeType.Block:
            if len(self.value) != 1:
                with execution_environment:
                    for thing in self.value:
                        thing.interpret(execution_environment)
                    print("Environment at end of block:")
                    execution_environment.print()
                return None
            else:
                return self.value[0].interpret(execution_environment)

        if self.type == NodeType.Assignment:
            name = self.value[0]
            value = self.value[1].interpret(execution_environment)
            execution_environment.set_variable(name, value)
            return value

        if self.type == NodeType.DeclarationAssignment:
            name = self.value[0]
            value = self.value[1].interpret(execution_environment)
            execution_environment.define_variable(name, value)
            return value

        if self.type == NodeType.FunctionCall:
            func_expr = self.value[0]
            if type(self.value[1]) == Node:
                print("!!!")
                print(self)
                raise Exception
            func_arg_values = [
                value.interpret(execution_environment)
                for value in self.value[1]
            ]
            func = func_expr.interpret(execution_environment)
            if not func.type == NodeType.Function:
                raise Exception(f"Cannot interpret FunctionCall because {func} is not a function")
            func = func.value
            if func["type"] == FunctionType.External:
                func_body = func["body"]
                func_arg_names = [name for (name, type) in func["arg_names"]]
                with execution_environment:
                    if len(func_arg_names) != len(func_arg_values):
                        raise Exception
                    for i in range(len(func_arg_names)):
                        execution_environment.define_variable(func_arg_names[i],
                                                              func_arg_values[i])
                    try:
                        func_body.interpret(execution_environment)
                    except Return as r:
                        return_value = r.return_value
                    else:
                        raise Exception("Function body did not return any value")
                return return_value
            elif func["type"] == FunctionType.Internal:
                func_body = func["body"]
                return func_body(execution_environment, func_arg_values)
            else:
                raise Exception

        if self.type == NodeType.Class:
            class_name = self.value[0]
            class_functions = [
                value.interpret(execution_environment)
                for value in self.value[1]
            ]
            class_ = execution_environment.get_variable(class_name).value

        if self.type == NodeType.PrefixOperation:
            op = self.value[0]
            v = self.value[1].interpret(execution_environment)
            if op == "->":
                raise Return(v)
            raise Exception(
                f"Could not interpret PrefixOperation with {self.value}")

        if self.type == NodeType.PostfixOperation:
            v = self.value[0].interpret(execution_environment)
            op = self.value[1]
            if op == '!':
                if v.type == NodeType.Integer:
                    return Node(NodeType.Integer, factorial(v.value))
            if op == '?':
                v_as_float = v.convert_to_float()
                if v_as_float != None:
                    return Node(NodeType.Float, inverse_factorial(v_as_float.value))
            raise Exception(f"Could not interpret PostfixOperation with {self.value}")

        if self.type == NodeType.InfixOperation:
            lhs = self.value[0].interpret(execution_environment)
            op = self.value[1]
            rhs = self.value[2].interpret(execution_environment)
            if op in ['+', '-', '*']:
                func = {'+': (lambda x, y: x + y), '-': (lambda x, y: x - y), '*': (lambda x, y: x * y)}[op]
                if lhs.type == NodeType.Integer and rhs.type == NodeType.Integer:
                    return Node(NodeType.Integer, func(lhs.value, rhs.value))
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return Node(NodeType.Float, func(lhs_as_float.value, rhs_as_float.value))
            if op == '/':
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return Node(NodeType.Float, lhs_as_float.value / rhs_as_float.value)
            if op == '^':
                if lhs.type == NodeType.Integer and rhs.type == NodeType.Integer:
                    return Node(NodeType.Integer, lhs.value ** rhs.value)
                lhs_as_float = lhs.convert_to_float()
                rhs_as_float = rhs.convert_to_float()
                if lhs_as_float != None and rhs_as_float != None:
                    return Node(NodeType.Float, lhs_as_float.value ** rhs_as_float.value)
            if op == '.':
                if lhs.type == NodeType.Function and rhs.type == NodeType.Function:
                    def combined_func(execution_environment, params):
                        if len(params) != 1:
                            raise Exception
                        param = params[0]
                        with execution_environment:
                            return Node(NodeType.FunctionCall,
                                        (lhs, [Node(NodeType.FunctionCall, (rhs, [param]))])).interpret(
                                execution_environment)

                    return Node(NodeType.Function, {"type": FunctionType.Internal, "body": combined_func})
            raise Exception(
                f"Could not interpret InfixOperation with ({lhs}, {op}, {rhs})")

        if self.type == NodeType.Identifier:
            name = self.value
            return execution_environment.get_variable(name)

        if self.type == NodeType.BuiltinIdentifier:
            name = self.value
            return Builtins.builtins[name]

        if self.type == NodeType.Function:
            execution_environment.define_variable(self.value["name"], self)
            return self

        if self.type in [NodeType.String, NodeType.Integer, NodeType.List]:
            return self

        raise Exception(
            f"Could not interpret Node of type {NodeType.string(self.type)}")


class Builtins:
    builtins = dict()

    def drucke(execution_environment, params):
        print(params)

    builtins["drucke"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": drucke})

    def max(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == NodeType.List:
            raise Exception
        if not all([node.type == NodeType.Integer for node in lst.value]):
            raise Exception
        return Node(NodeType.Integer, max([node.value for node in lst.value]))

    builtins["max"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": max})

    def min(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == NodeType.List:
            raise Exception
        if not all([node.type == NodeType.Integer for node in lst.value]):
            raise Exception
        return Node(NodeType.Integer, min([node.value for node in lst.value]))

    builtins["min"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": min})

    def count(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == NodeType.List:
            raise Exception
        return Node(NodeType.Integer, len([node.value for node in lst.value]))

    builtins["count"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": count})

    def sum(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == NodeType.List:
            raise Exception
        if not all([NodeType.is_number(node.type) for node in lst.value]) or all(
                [node.type == NodeType.Float for node in lst.value]):
            raise Exception

        if not all([node.type == NodeType.Integer for node in lst.value]):
            l_ = [node.value for node in lst.value]
            res = 0
            for l in l_:
                res += l
            return Node(NodeType.Integer, res)

        if not all([node.type == NodeType.Float for node in lst.value]):
            l_ = [node.value for node in lst.value]
            res = 0.0
            for l in l_:
                res += l
            return Node(NodeType.Float, res)

    builtins["min"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": min})

    def avg(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not lst.type == NodeType.List:
            raise Exception
        if not all([node.type in [NodeType.Integer, NodeType] for node in lst.value]):
            raise Exception

        l_ = [node.value for node in lst.value]
        return Node(NodeType.Integer, sum(l_) / len(l_))

    builtins["avg"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": avg})

    def hex(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not NodeType.is_number(lst.type):
            raise Exception

        return Node(NodeType.Hexadecimal, hex(lst.value))

    builtins["hex"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": hex})

    def bin(execution_environment, params):
        if len(params) != 1:
            raise Exception
        lst = params[0]
        if not NodeType.is_number(lst.type):
            raise Exception

        return Node(NodeType.Binary, bin(lst.value))

    builtins["bin"] = Node(NodeType.Function, {"type": FunctionType.Internal, "body": bin})

class Error(Exception):  # TODO: Implement in own language
    def __init__(self, error_name, details):
        self.error_name = error_name
        self.details = details


class CommittedDeadlySinError(Error):
    def __init__(self, details):
        super().__init__('You committed a deadly sin: ', details)


class WhereToStartError(Error):
    def __init__(self, details):
        self.details = details
        super().__init__(f'{language_name} does not know where to start: ',
                         self.details)  # Error: chars  @ x y IN file


class File:
    def __init__(self, content):
        self.content = content
        self.position = 0
        self.line_counter = 0
        self.column_counter = 0
        self.prefix_operators = ['->', '°']

        self.separators = ';:.'
        self.pos1 = 0  # for saving positions
        self.virtpos = 0  # for simulating parse-progressions

    def get(self):
        return self.content[self.position]

    def slice(self, length):
        return self.content[self.position:(self.position + length)]

    def is_whitespace(self):
        return self.get() in string.whitespace

    def skip_whitespace(self):
        while self.is_whitespace():
            if self.get() == '\n':
                self.line_counter += 1
            self.position += 1

    def is_singleline_comment(self):
        return self.get() == '§'

    def skip_singleline_comment(self):
        self.position += 1
        while self.get() != '\n':
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

        if any([thing.type == NodeType.Comma for thing in things]):
            i = 0
            list_elements = []
            while True:
                list_elements.append(things[i])
                i += 1
                if i == len(things):
                    break
                if things[i].type != NodeType.Comma:
                    raise Exception
                i += 1
                if i == len(things):
                    break
            return Node(NodeType.List, list_elements)

        return None

    def recognize_assignment(self, things, o):
        if len(things) >= o + 2 and \
                (
                        things[o + 0].type == NodeType.Identifier \
                        and things[o + 1].type == NodeType.InfixOperator \
                        and things[o + 1].value == '='
                ):
            if things[o + 2].value == '"':
                return things[:o] + [
                    Node(NodeType.Assignment,
                         (things[o + 0].value, things[o + 3]))
                ] + things[(o + 4):], True
            else:
                return things[:o] + [
                    Node(NodeType.Assignment,
                         (things[o + 0].value, things[o + 2]))
                ] + things[(o + 3):], True

        return things, False

    def recognize_for_loop_expression(self, things, o):
        if len(things) >= o + 4 and things[o + 1].type == NodeType.Identifier \
                and things[o + 2].type == NodeType.InfixOperator and things[o + 2].value == ':' \
                and NodeType.is_iterable(things[o + 3].type):
            return things[:o] + [
                Node(NodeType.ForLoopExpression, (things[o + 1].value, things[o + 3].value))
            ], True

        else:
            return things, False

    def recognize_conditional_expression(self, things, o):
        if len(things) >= o + 4 and \
                (
                        (
                                things[o + 0].type == NodeType.Statement \
                                and things[o + 1].type in [NodeType.Identifier, NodeType.String, NodeType.Integer,
                                                           NodeType.List, NodeType.Function] \
                                and things[o + 2].type == NodeType.InfixOperator \
                                and not things[o + 2].value in ['==', '<', '>', '>=', '<=', '%'] \
                                and things[o + 3].type in [NodeType.Identifier, NodeType.String, NodeType.Integer,
                                                           NodeType.List, NodeType.Function]

                        ) \
                        or things[o + 1].type == NodeType.Boolean
                ):
            return things[:o] + [
                Node(NodeType.ConditionalExpression,
                     (things[o + 0].value, things[o + 1].value,
                      things[o + 2].value))
            ] + things[(o + 4):], True

        return things, False

    def recognize_function_call(self, things, o):
        if len(things) >= o + 2 and \
                (
                        NodeType.is_expression(things[o + 0].type) \
                        and things[o + 1].type == NodeType.Block \
                        and len(things[o + 1].value) == 1
                ):
            func_expr = things[o + 0]
            func_args = things[o + 1].value[0]
            if func_args.type != NodeType.List:
                func_args = [func_args]
            else:
                func_args = func_args.value
            return things[:o] + [
                Node(NodeType.FunctionCall, (func_expr, func_args))
            ] + things[(o + 2):], True

        return things, False

    def recognize_prefix_operation(self, things, o):
        if len(things) >= o + 2 and \
                (
                        things[o + 0].type == NodeType.PrefixOperator \
                        and NodeType.is_expression(things[o + 1].type)
                ):
            return things[:o] + [
                Node(NodeType.PrefixOperation,
                     (things[o + 0].value, things[o + 1]))
            ] + things[(o + 2):], True

        return things, False

    def recognize_postfix_operation(self, things, o):
        if len(things) >= o + 2 and \
                (
                        NodeType.is_expression(things[o + 0].type) \
                        and things[o + 1].type == NodeType.PostfixOperator
                ):
            return things[:o] + [
                Node(NodeType.PostfixOperation,
                     (things[o + 0], things[o + 1].value))
            ] + things[(o + 2):], True

        return things, False

    def recognize_infix_operation(self, things, o):
        if len(things) >= o + 3 and \
                (
                        NodeType.is_expression(things[o + 0].type) \
                        and things[o + 1].type == NodeType.InfixOperator \
                        and things[o + 1].value != '=' \
                        and NodeType.is_expression(things[o + 2].type)
                ):
            return things[:o] + [
                Node(NodeType.InfixOperation,
                     (things[o + 0], things[o + 1].value, things[o + 2]))
            ] + things[(o + 3):], True

        return things, False

    def recognize_declaration_assignment(self, things, o):
        if len(things) >= o + 2 and things[o + 0].type == NodeType.PrefixOperator and things[o + 0].value == '°' \
                and things[o + 1].type == NodeType.Assignment:
            return things[:o] + [Node(NodeType.DeclarationAssignment, things[o + 1].value)] + things[(o + 2):], True
        return things, False

    def repeatedly_transform_thing_list(self, things):
        # The order of this list is important because it dictates the precedence of different types of expressions
        recognize_list = [
            self.recognize_function_call,
            self.
                recognize_postfix_operation,
            # If you write `x!` or `x?`, you probably always expect that to be parsed before any expression that it's part of
            self.recognize_infix_operation,
            self.recognize_conditional_expression,
            self.recognize_assignment,
            self.recognize_declaration_assignment,
            self.recognize_prefix_operation,
            self.recognize_for_loop_expression
            # `-> <expr>` is a PrefixOperation and you always want to keep the whole `<expr>` together
        ]
        i = 0
        while i < len(recognize_list):
            did_recognize_any = False
            for offset in range(len(things)):
                things, did_recognize = recognize_list[i](things, offset)
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
            raise Exception
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
        final_node = Node(NodeType.Block, things)
        return final_node

    def is_boolean(self):
        return self.slice(4) == 'true' or self.slice(5) == 'false'

    def parse_boolean(self):
        if self.slice(5) == 'False':
            self.position += 5
            return Node(NodeType.Boolean, self.slice(5))
        elif self.slice(4) == 'True':
            self.position += 4
            return Node(NodeType.Boolean, self.slice(4))

    def is_block(self):
        return self.get() == '{'

    def parse_block(self):
        return self.parse_general_block(NodeType.OpeningCurly,
                                        NodeType.ClosingCurly)

    def parse(self):
        # self.check_start()
        # self.skip_useless()
        # block = self.parse_block()
        # self.check_end()
        # return Node()
        return self.parse_general_block(NodeType.Start, NodeType.End)

    def check_start(self):
        if not self.slice(5) == 'START' in self.content:
            raise WhereToStartError('ABC')  # details = position of error

    def check_end(self):
        if self.slice(3) == 'END':
            sys.exit()

    def is_separator(self):
        return self.get() in self.separators

    def is_builtin_identifier(self):
        return self.get() == '#'

    def parse_builtin_identifier(self):
        self.position += 1
        name = self.parse_identifier().value
        return Node(NodeType.BuiltinIdentifier, name)

    def is_prefix_operator(self):
        return any([self.slice(len(op)) == op for op in self.prefix_operators])

    def parse_prefix_operator(self):
        for op in self.prefix_operators:
            if self.slice(len(op)) == op:
                self.position += len(op)
                return Node(NodeType.PrefixOperator, op)
        raise Exception

    def is_statement(self):
        self.statements = ['if', 'elif', 'else', 'while', 'for']

        for i in range(5):

            if self.slice(i) in self.statements:
                return True

    def parse_statement(self):

        for i in range(5):
            if self.slice(i) in self.statements:
                self.position += i
                return Node(NodeType.Statement,
                            self.content[self.position - i:self.position])

    def is_type_assignment(self):
        ...

    def return_string(self):
        return Node(NodeType.String, self.string)

    def parse_thing(self, no_blocks=False):
        self.skip_useless()
        for (is_x, parse_x) in [
            (self.is_start, self.parse_start),
            (self.is_function, self.parse_function),
            (self.is_end, self.parse_end), (self.is_string, self.parse_string),
            (self.is_for_loop, self.parse_for),
            (self.is_try, self.parse_try),
            (self.is_statement, self.parse_statement),
            (self.is_hex, self.parse_hex),
            (self.is_bin, self.parse_bin),
            (self.is_number, self.parse_number),
            (self.is_identifier, self.parse_identifier),
            (self.is_prefix_operator, self.parse_prefix_operator),
            (self.is_postfix_operator, self.parse_postfix_operator),
            (self.is_infix_operator, self.parse_infix_operator),
            ((self.is_opening_curly,
              self.parse_opening_curly) if no_blocks else
            (self.is_block, self.parse_block)),
            (self.is_closing_curly, self.parse_closing_curly),
            (self.is_comma, self.parse_comma),
            (self.is_builtin_identifier, self.parse_builtin_identifier),
            (self.is_boolean, self.parse_boolean),
            (self.is_colon, self.parse_colon)
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
        while self.get() in string.ascii_letters + '_' + string.digits:
            identifier += self.get()
            self.position += 1
        return Node(NodeType.Identifier, identifier)

    # def is_integer(self):
    #     return self.get() in string.digits

    # def parse_integer(self):
    #     integer = 0
    #     while self.get() in string.digits:
    #         integer *= 10
    #         integer += int(self.get())
    #         self.position += 1
    #     return Node(NodeType.Integer, integer)

    def is_hex(self):
        return self.slice(2) == '0x' and self.content[self.position + 2] in string.digits

    def parse_hex(self):
        hex_number = '0x'
        self.position += 2

        while self.get() in string.digits:
            hex_number += self.get()
            self.position += 1

        return Node(NodeType.Hexadecimal, hex_number)

    def is_bin(self):
        return self.slice(2) == '0b' and self.content[self.position + 2] in string.digits

    def parse_bin(self):
        bin_number = '0b'
        self.position += 2

        while self.get() in string.digits:
            bin_number += self.get()
            self.position += 1

        return Node(NodeType.Binary, bin_number)

    def is_number(self):
        if self.get() == '.' and self.content[self.position + 1] in string.digits:
            return True
        elif self.get() in string.digits or self.slice(3) in ['inf', 'NaN']:
            return True

    def parse_number(self):
        number = 0
        points = 0
        decimals = 0
        exponential = 0
        exponent = 0

        if self.slice(3) in ['NaN', 'inf']:
            self.position += 3
            return Node(NodeType.Float, self.content[self.position - 3: self.position])

        while self.get() in string.digits + '.' and not exponential == 1:
            if points == 0:
                if self.get() in string.digits:
                    number *= 10
                    number += int(self.get())
                    self.position += 1
                if self.get() == '.':
                    points += 1
                    self.position += 1
            if points == 1:
                if self.get() in string.digits:
                    decimals += 1
                    number *= 10
                    number += int(self.get())
                    self.position += 1

            if self.slice(2) == 'e+':
                exponential += 1
                self.position += 2

        while self.get() in string.digits and exponential == 1:
            exponent *= 10
            exponent += int(self.get())
            self.position += 1
            print(exponent)

        if points == 1 and decimals > 0:
            number /= 10 ** decimals

        if exponential == 1:
            number *= 10 ** exponent

        if points == 1 or exponential == 1:
            return Node(NodeType.Float, number)
        else:
            return Node(NodeType.Integer, number)

    def is_postfix_operator(self):
        return self.get() in ['!', '?']

    def parse_postfix_operator(self):
        c = self.get()
        self.position += 1
        return Node(NodeType.PostfixOperator, c)

    def is_infix_operator(self):
        if self.slice(3) in ['//=', 'and', 'not']:
            return True
        elif self.slice(2) in [
            '//', '%=', '+=', '==', '-=', '*=', '^=', '==', '/=', '>=',
            '<=', 'or'
        ]:
            return True
        elif self.get() in ['^', '>', '<', '*', '/', '=', '+', '-', '.']:
            return True

    def parse_infix_operator(self):
        if self.slice(3) in ['//=', 'and', 'not']:
            s = self.slice(3)
            self.position += 3
            return Node(NodeType.InfixOperator, s)
        elif self.slice(2) in [
            '//', '%=', '+=', '==', '-=', '*=', '^=', '==', '/=', '>=',
            '<=', 'or'
        ]:
            s = self.slice(2)
            self.position += 2
            return Node(NodeType.InfixOperator, s)
        elif self.get() in ['^', '>', '<', '*', '/', '=', '+', '-', '.']:
            c = self.get()
            self.position += 1
            return Node(NodeType.InfixOperator, c)
        raise Exception

    def is_start(self):
        return self.slice(5) == "START"

    def parse_start(self):
        self.position += 5
        return Node(NodeType.Start, None)

    def is_end(self):
        return self.slice(3) == "END"

    def parse_end(self):
        self.position += 3
        return Node(NodeType.End, None)

    def is_opening_curly(self):
        return self.get() == '{'

    def parse_opening_curly(self):
        self.position += 1
        return Node(NodeType.OpeningCurly, None)

    def is_closing_curly(self):
        return self.get() == '}'

    def parse_closing_curly(self):
        self.position += 1
        return Node(NodeType.ClosingCurly, None)

    def is_type(self):
        for typename in [
            'complex', 'int', 'str', 'float', 'bin', 'hex', 'oct', 'list',
            'bool', 'dict'
        ]:
            if self.slice(len(typename)) == typename:
                return True
        return False

    def parse_type(self):
        for typename in [
            'complex', 'int', 'str', 'float', 'bin', 'hex', 'oct', 'list',
            'bool', 'dict'
        ]:
            if self.slice(len(typename)) == typename:
                self.position += len(typename)
                return Node(NodeType.Type, typename)
        raise Exception

    def is_comma(self):
        return self.get() == ','

    def parse_comma(self):
        self.position += 1
        return Node(NodeType.Comma, None)

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
        return Node(NodeType.String, s)

    def is_colon(self):
        return self.get() == ':'

    def parse_colon(self):
        self.position += 1
        return Node(NodeType.Colon, ':')

    def is_try(self):
        return self.slice(3) == 'try'

    def parse_try_keyword(self):
        if self.slice(3) != 'try':
            raise Exception
        self.position += 3

    def parse_except_keyword(self):
        if self.slice(6) != 'except':
            raise Exception
        self.position += 6

    def parse_error_keyword(self):
        self.errors = ['Exception', 'CommittedDeadlySinError']
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

        return Node(NodeType.Try, {"try_block": try_, "error": error, "except_block": except_})

    def is_for_loop(self):
        return self.slice(3) == 'for'

    def parse_for_loop_var(self):
        if not self.is_identifier():
            raise Exception
        return self.parse_identifier().value

    def parse_for_loop_keyword(self):
        if self.slice(3) != 'for':
            raise Exception
        self.position += 3

    def parse_for(self):
        self.parse_for_loop_keyword()
        self.skip_useless()
        var = self.parse_for_loop_var()
        self.skip_useless()
        self.parse_colon()
        self.skip_useless()
        if self.is_opening_curly():
            iter_ = self.parse_block()
            self.skip_useless()
        elif self.is_string():
            iter_ = self.parse_string()
            self.skip_useless()
        elif self.is_identifier():
            iter_ = self.parse_identifier()
            self.skip_useless()
        self.parse_block()

        return Node(NodeType.ForLoop, {'iterated_variable': var, 'iterable': iter_})

    def is_if(self):
        return self.slice(2) == 'if'

    def parse_if_keyword(self):
        if self.slice(2) != 'if':
            raise Exception
        self.position += 2

    def parse_if(self):
        self.parse_if()
        self.skip_useless()
        ...

    def is_class(self):
        return self.slice(2) == 'cl'

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

        return Node(NodeType.Class, {'name': name, 'function_names': functions})

    def is_function(self):
        return self.slice(2) == "fn"

    def parse_function_keyword(self):
        if self.slice(2) != "fn":
            raise Exception
        self.position += 2

    def parse_function_name(self):
        if not self.is_identifier():
            raise Exception
        return self.parse_identifier().value

    def parse_function_parameter_list(self):
        if not self.is_opening_curly():
            raise Exception
        self.parse_opening_curly()
        self.skip_useless()

        type = 'any'
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
                lst.append((name, 'any'))

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
        self.parse_function_keyword()
        self.skip_useless()
        name = self.parse_function_name()
        self.skip_useless()
        parameters = self.parse_function_parameter_list()
        self.skip_useless()
        body = self.parse_function_body()
        self.skip_useless()
        return Node(NodeType.Function,
                    {"type": FunctionType.External, "name": name, "arg_names": parameters, "body": body})


def main(filename):
    with open(filename, "rt", encoding="utf-8") as f:
        contents = f.read()
    file = File(contents)
    root_node = file.parse()
    print("Root Node:")
    print(root_node)
    print("")
    print("Interpreting...")
    root_node.interpret(ExecutionEnvironment())
    print("")


tests = []
# tests += ["assignment", "scope_test", "comments", "functions"]
# tests += ["if_test"]
# tests += ["type_assignment"]
# tests += ["declarations"]
# tests += ["dot"]
# tests += ["factorial"]
# tests += ["brack"]
# tests += ["drucke"]
tests += ["for_test"]
tests += ["try"]
tests += ["abc"]
tests += ["builtin_functions"]

for test in tests:
    print(f"Running test: {test}")
    main(f"Tests/{test}.br")

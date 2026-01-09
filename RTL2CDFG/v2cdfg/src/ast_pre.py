from ast_from_json import *
import os
import sys
import subprocess
from anytree.exporter import JsonExporter, UniqueDotExporter
from anytree import AnyNode


# Unary operators.
operator1list = ["LNot", "Not", "URxor", "URand", "URor", "URnand", "URnor"]
# Binary operators.
operator2list = [
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "ShiftLeft",
    "ShiftRight",
    "AshiftLeft",
    "AshiftRight",
    "And",
    "Or",
    "Eq",
    "Neq",
    "Eeq",
    "Neeq",
    "BitAnd",
    "BitOr",
    "BitXor",
    "BitNXor",
]
original_stdout = sys.stdout
operator_dict = {}

for operator in operator1list:
    operator_dict[operator] = 0
for operator in operator2list:
    operator_dict[operator] = 0

operator_dict["Lconc"] = 0
operator_dict["Cond"] = 0
operator_dict["Cond_If"] = 0
operator_dict["Idx"] = 0
operator_dict["Conc"] = 0
operator_dict["Casez"] = 0
operator_dict["PartAssign"] = 0

var_exp = {}
Idx_dict = {}
Idx_list = []
func_scope = False


def identical_Idx(node1, node2):
    # print(node1.name + node2.name)
    if node1.name.startswith("Idx") and node2.name.startswith("Idx"):
        if len(node1.children) != len(node2.children):
            return False
        for index, subnode1 in enumerate(node1.children):
            if index < len(node2.children):
                # print(subnode1)
                # print(node2.children[index])
                if not identical_Idx(subnode1, node2.children[index]):
                    return False
            else:
                return False
    else:
        if node1.name != node2.name:
            return False
        else:
            for index, subnode1 in enumerate(node1.children):
                if node2.children[index]:
                    if not identical_Idx(subnode1, node2.children[index]):
                        return False
                else:
                    return False
    return True

def check_node_in_nodelist(node, node_dict):
    for key, value in node_dict.items():
        if identical_Idx(node, value):
            return True, key
    return False, None


def pre(ast):
    if ast.name == "Case":
        ast.name = "Casez"
    # yanlw 26/8/2024
    if ast.name == "TpConcList" and ast.parent.name.startswith("Lconc"):
        num_ele = len(ast.children)
        for tnode in ast.children:
            if tnode.children[0].name == "Idx":
                operator_dict["PartAssign"] += 1
                tnode.children[0].name = "PartAssign_" + str(operator_dict["PartAssign"])
                # print(f"partassign1: {tnode.children[0].name}{tnode.children[0]}")
            num_ele -= 1

    # Two cases: predecessor is Lname or predecessor is Lconc (num_partassign is non-zero).
    if ast.name == "Idx" and ast.parent.name == "Lname":
        operator_dict["PartAssign"] += 1
        ast.name = "PartAssign_" + str(operator_dict["PartAssign"])
        # print(f"partassign2: {ast.name}")

    if ast.name == "If":
        ast.name = "Cond_If"
    if (
        ast.name in operator2list
        or ast.name in operator1list
        or ast.name == "Cond"
        or ast.name == "Idx"
        or ast.name == "Conc"
        or ast.name == "Casez"
        or ast.name == "Lconc"
        or ast.name == "Cond_If"
    ):
        if ast.name == "Idx":
            if Idx_dict and not ast.children[0].name == 'b':
                result, index = check_node_in_nodelist(ast, Idx_dict)
                if result:
                    # print("exit a identical Idx")
                    ast.name = Idx_dict[index].name
                else:
                    operator_dict[ast.name] = operator_dict[ast.name] + 1
                    ast.name = ast.name + "_" + str(operator_dict[ast.name])
            else:
                operator_dict[ast.name] = operator_dict[ast.name] + 1
                ast.name = ast.name + "_" + str(operator_dict[ast.name])
        else:
            operator_dict[ast.name] = operator_dict[ast.name] + 1
            ast.name = ast.name + "_" + str(operator_dict[ast.name])
        if ast.name.startswith("Idx"):
            Idx_dict[ast.name] = ast
    # if ast.name == "Lconc":
    #     ast.name = "Lname"
    #     tnode = AnyNode(name = "Conc", level = ast.level+1, parent = None)
    #     if ast.children:
    #         for subnode in ast.children:
    #             subnode.parent = tnode
    #     tnode.parent = ast
    # extend var_exp
    if ast.name == "Num":
        node_name = "Constant_"
        if ast.children[0]:
            node_name = node_name + ast.children[0].name + "'"
        if len(ast.children) > 1 and ast.children[1]:
            if (
                ast.children[1].name == "Hex"
                or ast.children[1].name == "Bin"
                or ast.children[1].name == "Dec"
                or ast.children[1].name == "Oct"
            ):
                node_name = node_name + ast.children[1].name[0]
            if ast.children[1].children[0]:
                node_name = node_name + ast.children[1].children[0].name
        node_name = "Const," + ast.children[0].name + "," + node_name
        var_exp[node_name] = node_name.split(',')[-1].replace("Constant_", "")
        print(var_exp[node_name])
    if ast.children:
        for subnode in ast.children:
            pre(subnode)


def pre_ast2json(ast, filename):
    input_directory = "./data"
    os.makedirs(input_directory, exist_ok=True)
    input_path = os.path.join(input_directory, filename)
    with open(input_path, 'w') as f:
        sys.stdout = f
        exporter = JsonExporter(indent=2, sort_keys=True)
        print(exporter.export(ast))
        sys.stdout = original_stdout
    print("Preprocessed AST JSON conversion completed.")


def pre_ast2png(ast, filename):
    name, ext = os.path.splitext(filename)
    input_directory = "./data"
    output_directory = "./ast_png"
    os.makedirs(input_directory, exist_ok=True)
    input_path = os.path.join(input_directory, name + ".json")
    dot_output_path = os.path.join(output_directory, name + ".dot")
    output_path = os.path.join(output_directory, name + ".svg")
    # graphviz needs to be installed for the next line!
    UniqueDotExporter(ast).to_dotfile(dot_output_path)
    subprocess.check_call(['dot', dot_output_path, '-T', 'svg', '-o', output_path])
    sys.stdout = original_stdout
    print("Preprocessed AST PNG conversion completed.")

if __name__ == "__main__":
    data = read_json_file("./data/75_ast_clean.json")
    ast = build_tree_from_json(data)
    pre(ast)
    pre_ast2json(ast, "75_ast_clean.json")
    pre_ast2png(ast, "75_ast_clean.json")
    exit()

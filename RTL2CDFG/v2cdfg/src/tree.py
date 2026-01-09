# -*- coding: utf-8 -*-

# Build a tree from S-expression files and output AST as JSON.
from anytree import NodeMixin, AnyNode, RenderTree
from anytree.exporter import JsonExporter, UniqueDotExporter
from anytree.importer import JsonImporter
import sys
import os
import argparse
import json
from io import StringIO
from anytree.exporter import DotExporter
import subprocess

original_stdout = sys.stdout


def arguments():
    parser = argparse.ArgumentParser(description="Example program for CLI arguments.")
    parser.add_argument('-r', '--reduced', action='store_true', help="Use reduced syntax tree.")
    parser.add_argument('-j', '--json', action='store_true', help="Generate JSON file.")
    parser.add_argument('-p', '--png', action='store_true', help="Generate PNG file (JSON required).")
    parser.add_argument('filename', type=str, help="File name to process.", nargs='?')
    parser.add_argument('-i', '--input_directory', type=str, default='.', help="Input directory.")
    parser.add_argument('-o', '--output_directory', type=str, default='.', help="Output directory.")
    parser.add_argument('-v', '--verify', action='store_true', help="Read JSON file and verify no loss.")
    parser.add_argument('-e', '--exist', action='store_true', help="Check if *.json already exists in output.")
    return parser.parse_args()


# Read an S-expression file (e.g., comparator.txt) and build the AST.
# If reduced is true, build a reduced tree; otherwise build the full tree.
def read_file(file_name, red):
    with open(file_name, 'r') as file:
            content = file.read()
            #print(content)
    tag = -1
    while content[tag] != ')':
        tag = tag - 1
    # Slice from index 1 to the second-to-last character (strip outer parentheses).
    sexpr = content[1:tag]
    #print(sexpr)

    # Tokenize (strip inner parentheses of the S-expression).
    #tokens = tokenize(sexpr)
    #print(tokens)

    # Parse S-expression.
    parsed_expr = parse_sexpr(sexpr)
    #print("Parsed S-Expression:", parsed_expr)
    # Build AST.
    if red:
        ast = build_reduced_ast(parsed_expr)
        ast_walker_reduce(ast)
    else:
        ast = build_complete_ast(parsed_expr)
    return ast

# Process tokenized S-expression into a nested list.
def parse_sexpr(expr):
    tokens = tokenize(expr)
    LL =[]
    while tokens:
        LL.append(read_from_tokens(tokens))
    return LL

# Tokenize (strip inner parentheses of the S-expression).
def tokenize(expr):
    # Add spaces to ensure parentheses are separated correctly.
    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    return expr.split()

# Token processing algorithm: append contents inside parentheses to L.
def read_from_tokens(tokens):
    if not tokens:
        raise SyntaxError('Unexpected EOF while reading')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens and tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        if tokens:
            tokens.pop(0)  # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('Unexpected )')
    else:
        return atom(token)

# Parse as numeric or symbol.
def atom(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return str(token)

# Read operator list to help validate JSON.
operatorlist = []
with open("operatorlist.txt", "r+") as file:
    content = file.read()
    operatorlist = content.split(",")
    operatorlist = list(set(operatorlist))
tpvartplist = []

# Syntax tree construction algorithm.
def build_complete_ast(parsed_expr,parent = None):
    if isinstance(parsed_expr, list):
        #print(parsed_expr)
        operator = str(parsed_expr.pop(0))
        # Maintain operator list.
        if not operator in operatorlist and not operator.isdigit() and not len(operator) == 1:
            operatorlist.append(operator)
        # Combine module/task name into one node, e.g., "TpTask fullAdder".
        if operator == "TpTask":
            taskname = parsed_expr.pop(0)
            operator = operator + " " + taskname
        node = AnyNode(name = str(operator),parent = parent)
        for sub_expr in parsed_expr:
            child_node = build_complete_ast(sub_expr,parent = node)
        return node
    else:
        nodename = str(parsed_expr)
        nodename = nodename.replace("\n","").replace('"', '')
        nodename = nodename.replace("\\","")#.replace("\\", "")
        return AnyNode(name = nodename,parent = parent)

def build_reduced_ast(parsed_expr,parent = None):
    if isinstance(parsed_expr, list):
        # print(parsed_expr)
        operator = str(parsed_expr.pop(0))
        # Maintain operator list.
        if not operator in operatorlist and not operator.isdigit() and not len(operator) == 1:
            operatorlist.append(operator)
        # Combine module/task name into one node, e.g., "TpTask fullAdder".
        if operator == "TpTask":
            taskname = parsed_expr.pop(0)
            operator = operator + " " + taskname
        # Drop Atom nodes.
        if operator == "Atom" or operator == "Sym" or operator == "Parameter" or operator == "Assign" or operator == "Port" or operator == "Clist":
            build_reduced_ast(parsed_expr[0], parent=parent)
        else:
            if parsed_expr:
                if operator == "sharp":
                    operator = "TpVarTplist"
                    operatorlist.append("TpVarTplist")
                if operator == "Svar":
                    operator = "Var"
                    operatorlist.append("Var")
                node = AnyNode(name=str(operator), parent=parent)
                # Add condition to if.
                if operator == "If":
                    lst = ['TrueBlock']
                    lst.append(parsed_expr[1])
                    parsed_expr[1] = lst
                    lst = ['FalseBlock']
                    lst.append(parsed_expr[2])
                    parsed_expr[2] = lst
                if operator == "Ge" or operator == "Le" or operator == "Lt" or operator == "Gt" or operator == "Sub" or operator == "Mod" or operator == "Div" or operator == "Add" or operator == "Mul" or operator == "And" or operator == "Or" or operator == "BitAnd" or operator == "BitOr" or operator == "BitXor" or operator == "Eq" or operator == "Neq" or operator == "Eeq" or operator == "Neeq" or operator == "ShiftRight" or operator == "ShiftLeft" or operator == "AshiftLeft" or operator == "AshiftRight" or operator == "Power" or operator == "BitNXor":
                    lst = ['Left']
                    lst.append(parsed_expr[0])
                    parsed_expr[0] = lst
                    lst = ['Right']
                    lst.append(parsed_expr[1])
                    parsed_expr[1] = lst
                # Add condition to '?' operator.
                if operator == "Cond":
                    lst = ['TrueBlock']
                    lst.append(parsed_expr[1])
                    parsed_expr[1] = lst
                    lst = ['FalseBlock']
                    lst.append(parsed_expr[2])
                    parsed_expr[2] = lst
                if operator == "TpVarTp":
                    tpvartplist.append(parsed_expr[0])
                for sub_expr in parsed_expr:
                    child_node = build_reduced_ast(sub_expr, parent=node)
                return node
            else:
                ()
    else:
        nodename = str(parsed_expr)
        nodename = nodename.replace("\n","").replace('"', '')
        nodename = nodename.replace("\\","")#.replace("\\", "")
        #print(nodename)
        # Drop symbols and delays.
        if nodename == "NoDelay" or nodename == "false" or nodename == "NoDE" or nodename == "NullExp" or nodename == "Rng0" or nodename == "NullPara" or nodename == "NoPathDelay" or nodename == "NoCond" or nodename == "Nullnet" or nodename == "NoElse" or nodename == "EmptyStmt" or nodename == "null":
            ()
        else:
            return AnyNode(name=nodename, parent=parent)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def tree_to_list(node):
    if not node.children:
        for operator in operatorlist:
            if node.name == operator:
                return [node.name]
        return node.name
    else:
        result = [node.name]
        for child in node.children:
            result.append(tree_to_list(child))
        return result
def list_to_s_expression(lst):
    if isinstance(lst, list):
        return '(' + ' '.join(list_to_s_expression(el) for el in lst) + ')'
    else:
        return str(lst)

def build_tree_from_json(data, level=1, parent=None):
    operator = data['name']
    node = AnyNode(name=operator, level=level, parent=parent)
    for child_data in data.get('children', []):
        child_node = build_tree_from_json(child_data, level=level + 1, parent=node)
    return node

# Insert a node, e.g., a-children[ b,c ] -> insert_node(a,'d') -> a-children[ d-children[ b,c ] ]
def insert_node(node, name):
    i_node = AnyNode(name=name, parent=None)
    if node.children:
        for subnode in node.children:
            subnode.parent = i_node
        i_node.parent = node


def delete_subtree(node, index):
    lst = list(node.parent.children)
    del lst[index]
    tple = tuple(lst)
    node.parent.children = tple


def delete_node(node, index):
    if node.children:
        lst = list(node.parent.children)
        index1 = 0
        for subnode in node.children:
            subnode.parent = node.parent
            lst.insert(index + index1, subnode)
            index1 = index1 + 1
        tple = tuple(lst)
        node.parent.children = tple
    else:
        delete_subtree(node, index)

# Prune tpvar nodes.
def reduce_strategy_tpvar(node):
    index = 0
    if node.parent is not None and node.parent.children:
        for subnode in node.parent.children:
            if subnode.name == node.name:
                delete_node(subnode, index)
                break
            else:
                index = index + 1

# Prune duplicate input/output in sigset.
def reduce_strategy_sigset(node):
    index = 0
    if node.parent.name == "TpFunBody":
        return
    if node.children:
        for subnode in node.children:
            if subnode.name == "Input" or subnode.name == "Output":
                delete_subtree(subnode, index)
            else:
                index = index + 1

# Drop leaf operator nodes with no parameters.
def reduce_strategy_leaf(node):
    index = 0
    if node.parent is not None and node.parent.children:
        for subnode in node.parent.children:
            if subnode.name == node.name:
                delete_subtree(subnode, index)
            else:
                index = index + 1


def reduce_strategy_NullExp(node):
    if node.name == "NullExp":
        ()


def reduce_strategy_RngC(node):
    if node.parent.name == "Idx":
        return
    if node.children[0].children[0].name == '0' and node.children[1].children[0].name == '0':
        index = 0
        for subnode in node.parent.children:
            if subnode == node:
                break
            else:
                index = index + 1
        delete_subtree(node, index)


def ast_walker_reduce(node):
    # Traverse and simplify nodes; add more rules here as needed.
    if node.name == "Genvar":
        node.children = node.children[0].children
    if node.name == "TpVar":
        reduce_strategy_tpvar(node)
    if node.name == "RngC":
        reduce_strategy_RngC(node)
    if node.name == "sigset":
        reduce_strategy_sigset(node)
    if not node.children and node.name in operatorlist and not node.name in tpvartplist and not node.name.isdigit() and not len(node.name) == 1:
        if node.parent and not node.parent.name == "Strg" and not node.parent.name == "Pstr" and not node.parent.name == "PathVar" and not node.parent.name == "Define" and not node.parent.name == "tpBlkLbl" and not node.parent.name == "Var":
            reduce_strategy_leaf(node)
    if node.children:
        for child in node.children:
            ast_walker_reduce(child)

# Main: read the S-expression file (e.g., comparator.txt) and output JSON/PNG.
# comparator.json: intermediate AST form.
# comparator.png: visualized AST structure.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python readfile.py <filename>")
    else:
        args = arguments()
        file_name = args.filename
        name, ext = os.path.splitext(file_name)
        input_path = os.path.join(args.input_directory, file_name)
        # Ensure directory exists.
        os.makedirs(args.input_directory, exist_ok=True)
        if args.verify:
            data = read_json_file(input_path)
            ast = build_tree_from_json(data)
            #print(operatorlist)
            # exporter = JsonExporter(indent=2, sort_keys=True)
            # data = exporter.export(ast)
            # importer = JsonImporter()
            # root = importer.import_(data)
            # UniqueDotExporter(root).to_picture("test.png")
            # exit()
            list1 = tree_to_list(ast)
            str1 = list_to_s_expression(list1).replace(" ", "").replace("\n", "")
            # Only the parts that do not affect the tree still lack parentheses.
            with open(name + ".s", 'r') as file:
                str2 = file.read().replace(" ", "").replace("\n", "")
            print("S-expression reconstructed from JSON:\n" + repr(str1))
            print("Original S-expression:\n" + repr(str2))
            if str1 == str2:
                print("Verification passed")
            else:
                print("Verification failed")
        else:
            output_file_1 = name + ".json"
            dot_file = name + ".dot"
            output_file_2 = name + ".svg"
            output_file_3 = name + ".txt"
            # Build full file paths.
            output_path1 = os.path.join(args.output_directory , output_file_1)
            output_path2 = os.path.join(args.output_directory , output_file_2)
            output_path3 = os.path.join(args.output_directory , output_file_3)
            output_path4 = os.path.join(args.output_directory , dot_file)
            # Ensure directory exists.
            if args.exist:
                if os.path.exists(output_path1):
                    print(f"File '{output_path1}' exists in directory '{args.output_directory}'.")
                    exit()
                else:
                    ()
            os.makedirs(args.output_directory, exist_ok=True)
            ast = read_file(input_path,args.reduced)
            with open("operatorlist.txt", "w+") as file:
                operatorlist = list(set(operatorlist))
                for operator in operatorlist:
                    file.write(str(operator) + ',')
            # print("Filename without extension: " + name)
            # with open(output_path3,'w') as txtfile:
            #     sys.stdout = txtfile
            #     print(RenderTree(ast))
            if args.json:
                with open(output_path1, 'w') as f:
                    sys.stdout = f
                    exporter = JsonExporter(indent=2, sort_keys=True)
                    print(exporter.export(ast))
                sys.stdout = original_stdout
                print("JSON file generated.")
            if args.png:
                importer = JsonImporter()
                with open(output_path1, 'r') as file:
                    root = importer.read(file)
                # graphviz needs to be installed for the next line!
                # UniqueDotExporter(root).to_picture(output_path2)
                # UniqueDotExporter(root).to_dotfile(output_path4)
                # subprocess.check_call(['dot', output_path4, '-T', 'png', '-o', output_path2])
                UniqueDotExporter(root).to_dotfile(output_path4)
                # subprocess.check_call(['dot', output_path4, '-T', 'svg', '-o', output_path2])
                sys.stdout = original_stdout
                print("PNG file generated.")

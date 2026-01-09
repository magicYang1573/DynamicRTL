import networkx as nx
from anytree import NodeMixin, AnyNode, RenderTree
# import matplotlib.pyplot as plt
import sys
import re
import os
from ast_from_json import *
from ast_pre import *
from collections import defaultdict
# import pygraphviz as pgv
import matplotlib.pyplot as plt
from structures import *
import math
import copy

original_init = AnyNode.__init__

# AnyNode.__init__ = new_init

def has_suffix(s):
    # Suffix matching: eg case_2
    return bool(re.search(r"_\d+$", s))

# Delete the level attribute in the AnyNode node, because the node in the AST has a level attribute
def copy_without_level(node):
    attributes = node.__dict__.copy()
    # delete level 
    attributes.pop('level', None)

    if '_NodeMixin__children' in attributes:
        attributes['children'] = attributes.pop('_NodeMixin__children')
    else:
        attributes['children'] = []
    if '_NodeMixin__parent' in attributes:
        attributes['parent'] = attributes.pop('_NodeMixin__parent')
    else:
        attributes['parent'] = []
    
    # Use the remaining attribute
    new_node = AnyNode(**attributes)
    
    return new_node

class variable:
    def __init__(self,range,type=None) :
        self.Range = range
        self.type = type

class AnynodeVisitor(object):
    def visit(self, node):
        """Visit a node."""
        method_name = node.name
        if any(node.name.startswith(operator) for operator in operator2list) and not node.name == "Left" and not node.name == "Right" and has_suffix(node.name):
            method_name = "Operator2"
        if node.name.startswith("Cond") and has_suffix(node.name):
            method_name = "Cond"
        if node.name.startswith("Idx") and has_suffix(node.name):
            method_name = "Idx"
        if node.name.startswith("Conc") and has_suffix(node.name):
            method_name = "Conc"
        if node.name.startswith("Lconc") and has_suffix(node.name):
            method_name = "Lconc"
        if node.name.startswith("Casez") and has_suffix(node.name):
            method_name = "Casez"
        if node.name.startswith("TpSigDeclBody") and has_suffix(node.name):
            method_name = "TpSigDeclBody"
        if node.name.startswith("PartAssign") and has_suffix(node.name):
            method_name = "PartAssign"
        if any(node.name.startswith(operator) for operator in operator1list) and has_suffix(node.name):
            method_name = "Operator1"
        method = 'visit_' + method_name
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        if node.children:
            for child in node.children:
                self.visit(child)
        else:
            ()
    def generic_rev_visit(self, node):
        if node.children:
            for child in reversed(node.children):
                self.visit(child)
        else:
            ()

class DFGExtractor(AnynodeVisitor):
    def __init__(self, graph=None):
        # Dictionary definitions.
        self.defs = {}  # Variable definition locations.
        self.uses = {}  # Variable use locations.
        self.dfg = graph if graph is not None else nx.MultiDiGraph()  # Initialize a directed graph to represent CDFG

        # Operand and target stacks for arithmetic operator matching.
        self.target = []
        self.Source1 = [] 
        self.Source2 = []

        # Store ternary cond operands and targets.
        self.cond_source1 = []
        self.cond_source2 = []
        self.cond_source3 = []
        self.cond_target = []

        # Operator stack for matching.
        self.operators = []

        # Condition stack.
        self.conditions = []

        # Concat stack for matching concat content.
        self.conc = []

        # Case stack.
        self.casez = []

        # Branch indicator; used to detect current branch during traversal.
        self.ast_branch = "init"

        self.con_label = 0  # Label for concat edges.
        self.oper_label = 0
        self.const_label = 0
        self.case_num = 0 #
        self.parameter = 0  # Index of function parameter during calls.
        self.conc_num = 0  # Count of conc instances.

        self.wire_set = set()
        self.var_set = set() 
        self.in_set = set()  # Inputs for graph rendering.
        self.out_set = set()  # Outputs for graph rendering.
        self.sigset_set = set()  # Internal signal set (function locals, etc.).
        self.seq_set = set() 
        self.comb_set = set()  # Connector set: operators and condition symbols.
        self.label_set = set()  # Case label set.

        self.edge_set = set()

        self.node_dict = {}  # Node lookup for edge wiring.
        self.edge_dict = {}  # Edge lookup by id.
        self.wire_dict = {} #
        self.temp_dict = {} #

        self.fundec = 0  # Inside function definition; avoid name collisions.
        self.func_input_num = 0  # Function parameter count.
        self.func_input_list = []  # Function parameter list.
        self.func_list = []  # Track function definitions.
        self.func_dict = {}  # Function node lookup.
        self.func_call = ""  # Current function name during calls.
        self.func_left = None  # Record function assignment target.
        self.current_scope = {}  # Current scope for nested functions/control structures.

        self.if_scope = None  # Track if-scope status.

        # hzq
        self.left = 0
        self.part = 0

        #hzq
        self.tpRange = [0, 0]  # Store variable range; [0,0] means length 1.
        self.type = None
        self.vars = {}  # Variable definitions and metadata.
        self.partselects = []  # Partselects in lconc.

        #block
        self.block_num = 0

        #expression
        self.var_exp = var_exp
        self.Idx_operator2_left = None
        self.Idx_operator2_right = None
        self.Idx_exp = None

    def var_exp_directory(self,key):
        if key in self.var_exp:
            return self.var_exp[key]
        else:
            return key
        
    def in_nodes(self, node):
        for node_id in self.dfg.nodes:
            # print(f"nodename : {node.name}")
            attributes = node.__dict__.copy()
            if node.name == node_id.name:
                # print(f"repeat : {node.name}")
                return True
        self.node_dict[node.name] = node
        return False

    def visit_Function(self, node):
        self.func_input_num = 0
        self.func_input_list = []
        self.fundec = 1
        if node.children[0].name == "TpFunHead":
            tnode = node.children[0]
            if len(tnode.children) > 1:
                function_name = tnode.children[1].name
                self.current_scope["Function" + function_name] = [self.func_input_num,self.func_input_list]
                self.func_list.insert(0, "Function" + function_name)
        self.generic_visit(node)
        if self.func_dict["Function" + function_name].startswith("Casez"):
            print(f"remove {self.func_list[0] + function_name}")
            self.dfg.remove_node(self.node_dict[self.func_list[0] + function_name])
        self.fundec = 0
        # self.current_scope.pop("Function" + function_name)

    # hzq
    def visit_TpSigDeclBody(self,node):
        if node.parent.name == "Wire" or node.parent.name == "Reg":
            self.type = node.parent.name
        else:
            self.type = None
        if node.children[0].name == "TpVarTplist" :
            self.tpRange[0] = 0
            self.tpRange[1] = 0
        elif node.children[0].name == "RngC" :
            self.tpRange[0] = int(node.children[0].children[0].children[0].name)
            self.tpRange[1] = int(node.children[0].children[1].children[0].name)
        self.generic_visit(node)
        
    
   # yanlw 26/8/2024
    def visit_PartAssign(self, node):
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        isLname = 1 # 1 = name, 0 = conc
        
        # print(f"partassign: {node.name}; ast_branch：{self.ast_branch}, {node.children} ")
        if node.children:
            if node.children[0].name in self.node_dict.keys():
                self.add_edge(self.node_dict[node.name],self.node_dict[node.children[0].name], label = 1)
            i = 1
            while len(node.children) > i and node.children[i].name == "TpExpList":
                tnode = node.children[i]
                # hzq
                if self.ast_branch.startswith("Lconc") :
                    if tnode.children[0]:
                        #yanlw 8/15
                        isLname = 0 

                        tnode1 = AnyNode(name = "PartSelect_" + str(self.part))
                        #yanlw 8/15
                        self.partselects.append("PartSelect_" + str(self.part))
                       
                        self.add_node(tnode1)
                        self.comb_set.add(tnode1.name)
                        self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.name], label = 1)

                        tnode2 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode2)
                        self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 3)

                        # print (self.left + int(tnode.children[0].children[0].name))
                        tnode3 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode3)
                        self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 2)

                        # self.add_edge(self.node_dict[self.conc[0]],self.node_dict[tnode1.name], label = self.part + 1)
                        
                        self.left = self.left + 1
                        self.part = self.part + 1
                        # print(f"self.left: {self.left}")
                if tnode.children[0]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[0].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = i+1)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = i+2)

                i = i + 1
            if len(node.children) > i and node.children[i].name == "RngC":
                tnode = node.children[i]
                tnode = copy_without_level(tnode)
                # print(f"partassign: {node.name}; ast_branch：{self.ast_branch} ")
                if self.ast_branch.startswith("Lconc") :
                    if tnode.children[0]:
                        #yanlw 8/15
                        isLname = 0
                        # partselect -> partassign
                        tnode1 = AnyNode(name = "PartSelect_" + str(self.part))

                        #yanlw 8/15
                        self.partselects.append("PartSelect_" + str(self.part))
                        
                        self.add_node(tnode1)
                        self.comb_set.add(tnode1.name)
                        self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.name], label =  1)

                        tnode2 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode2)
                        self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 3)

                        self.left = self.left + int(tnode.children[0].children[0].name) -  int(tnode.children[1].children[0].name)
                        # print (self.left + int(tnode.children[0].children[0].name))
                        tnode3 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode3)
                        self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 2)
                        
                        # self.add_edge(self.node_dict[self.conc[0]],self.node_dict[tnode1.name], label = self.part + 1)

                        self.left = self.left + 1
                        self.part = self.part + 1
                if tnode.children[0]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[0].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
            if len(node.children) > i and node.children[i].name == "RngP":
                tnode = node.children[i]
                tnode = copy_without_level(tnode)
                if self.ast_branch.startswith("Lconc") :
                    if tnode.children[0]:
                        #yanlw 8/15
                        isLname = 0
                        # partselect -> partassign
                        tnode1 = AnyNode(name = "PartSelect_" + str(self.part))

                        #yanlw 8/15
                        self.partselects.append("PartSelect_" + str(self.part))
                        
                        self.add_node(tnode1)
                        self.comb_set.add(tnode1.name)
                        self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.name], label =  1)

                        tnode2 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode2)
                        self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 2)

                        self.left = self.left + int(tnode.children[0].children[0].name) -  int(tnode.children[1].children[0].name)
                        # print (self.left + int(tnode.children[0].children[0].name))
                        tnode3 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode3)
                        self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 3)
                        
                        # self.add_edge(self.node_dict[self.conc[0]],self.node_dict[tnode1.name], label = self.part + 1)

                        self.left = self.left + 1
                        self.part = self.part + 1
                if tnode.children[0]:
                    # Assume nodes already created.
                    if tnode.children[0].name == "Var":
                        # print(f"node: {tnode.children[0].children[0]}")
                        self.add_edge(self.node_dict[tnode.children[0].children[0].name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_plus" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
            if len(node.children) > i and node.children[i].name == "RngN":
                if tnode.children[0]:
                    # Assume nodes already created.
                    if tnode.children[0].name == "Var":
                        # print(f"node: {tnode.children[0].children[0]}")
                        self.add_edge(self.node_dict[tnode.children[0].children[0].name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_neg" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
            
        # flag is target
        if isLname :
            self.target.insert(0, node.name)

    def visit_ilists(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_olists(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_sigset(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_body(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_TpVarTp(self,node):
        temp_ast_branch = self.ast_branch
        if self.ast_branch == "init":
            ()
        elif self.ast_branch.startswith("ilists"):
            if node.children[0] and not self.in_nodes(node.children[0]):
                # hzq
                # print(f"{node.children[0].name} : {self.tpRange}")
                self.vars[node.children[0].name] = variable(self.tpRange.copy(),self.type)
                self.in_set.add(node.children[0].name)
                self.var_set.add(node.children[0].name)
                tnode = copy_without_level(node.children[0])
                self.add_node(tnode)
            if len(node.children) > 1:
                self.target.insert(0,node.children[0].name)
        elif self.ast_branch.startswith("olists"):
            if node.children[0] and not self.in_nodes(node.children[0]):
                self.vars[node.children[0].name] = variable(self.tpRange.copy(),self.type)
                self.out_set.add(node.children[0].name)
                self.var_set.add(node.children[0].name)
                tnode = copy_without_level(node.children[0])
                self.add_node(tnode)
            if len(node.children) > 1:
                self.target.insert(0,node.children[0].name)
        elif self.ast_branch.startswith("sigset"):
            if node.children[0] and not self.in_nodes(node.children[0]):
                # print(f"sigset:{node.children[0]}")
                self.vars[node.children[0].name] = variable(self.tpRange.copy(),self.type)
                self.sigset_set.add(node.children[0].name)
                self.var_set.add(node.children[0].name)
                tnode = copy_without_level(node.children[0])
                self.add_node(tnode)
            if len(node.children) > 1:
                self.target.insert(0,node.children[0].name)
            if node.children[0] and self.fundec == 1 and self.current_scope:
                # hzq
                node.children[0].name = self.func_list[0] + node.children[0].name
                self.vars[node.children[0].name] = variable(self.tpRange.copy(),self.type)
                self.sigset_set.add(node.children[0].name)
                self.var_set.add(node.children[0].name)
                tnode = copy_without_level(node.children[0])
                self.add_node(tnode)
                # print(node.children[0].name)
                self.current_scope[self.func_list[0]][0] += 1
                self.current_scope[self.func_list[0]][1].append(node.children[0].name)
        else:
            if node.children[0] and not self.in_nodes(node.children[0]):
                self.vars[node.children[0].name] = variable(self.tpRange.copy(),self.type)
                self.sigset_set.add(node.children[0].name)
                self.var_set.add(node.children[0].name)
                tnode = copy_without_level(node.children[0])
                self.add_node(tnode)
            if len(node.children) > 1:
                self.target.insert(0,node.children[0].name)
        self.ast_branch = node.name
        self.generic_visit(node)
        self.ast_branch = temp_ast_branch

    def visit_TpEqAssign(self, node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_Cond(self, node):
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        # print(self.ast_branch)
        # print(self.ast_branch)
        # hzq
        if self.ast_branch.startswith("Lconc"):
            # if self.target:
            target_name = self.target[0]
            # print("target:" + target_name)
            if target_name.startswith("Lconc"):
                # print(self.partselects)
                # print(self.partselects)
                for partsel in self.partselects :
                    # print(node.name + " " + partsel + "\n")
                    self.add_edge(self.node_dict[node.name], self.node_dict[partsel], label = 1)
        if self.ast_branch.startswith("Lname"):
            # if self.target:
            target_name = self.target[0]
            self.cond_target.insert(0,node.name)
            self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
            if self.if_scope == "TrueBlock" :
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 2)
            if self.if_scope == "FalseBlock":
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 3)
        # if self.if_scope == "TrueBlock" :
        #             self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 2)
        if self.if_scope:
            if node.parent.name == "TrueBlock":
                tlabel = 2
            if node.parent.name == "FalseBlock":
                tlabel = 3
            self.add_edge(self.node_dict[node.name],self.node_dict[self.conditions[0]], label = tlabel)
        if node.name.startswith("Cond_If"):
            assert False,"Exist if in design!"
            self.if_scope = "If"
        self.conditions.insert(0,node.name)
        self.ast_branch = node.name           
        self.generic_visit(node)
        if node.name.startswith("Cond_If"):
            self.if_scope = None

    def visit_TrueBlock(self, node):
        # if self.ast_branch.startswith("Cond"):
        self.ast_branch = "TrueBlock"
        # else:
        #     ()
        if self.if_scope == "If":
            self.if_scope = "TrueBlock"
        self.generic_visit(node)

    def visit_FalseBlock(self, node):
        # if self.ast_branch.startswith("Cond") or self.ast_branch.startswith("TrueBlock"):
        self.ast_branch = "FalseBlock"
        # else:
        #     ()
        if self.if_scope == "TrueBlock":
            self.if_scope = "FalseBlock"
        self.generic_visit(node)

    def visit_Lname(self, node):
        self.ast_branch = node.name
        self.generic_visit(node)

    # def visit_Lconc(self, node):
    #     # hzq
    #     self.left = 0
    #     self.add_node(node)
    #     self.node_dict[node.name] = copy_without_level(node)
    #     self.comb_set.add(node.name)
    #     node.is_target = True
    #     # if node.children:
    #     #     if node.children[0].name == "TpConcList":
    #     #         tnode = node.children[0]
    #     #         for element in tnode.children:
    #     #             if element.name == "Ele":
    #     self.conc.insert(0, node.name)
    #     self.target.insert(0, node.name)
    #     self.ast_branch = node.name
    #     self.generic_visit(node)
    #     self.con_label = 0

    # hzq 
    def visit_Lconc(self, node):
        # hzq 
        self.left = 0
        self.partselects = []
        self.target.insert(0, node.name)
        self.ast_branch = node.name
        self.conc_num += 1
        if node.children:
            if node.children[0].name == "TpConcList":
                self.generic_rev_visit(node.children[0])
            else : self.generic_visit(node)

    def visit_Left(self, node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_Right(self, node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_Conc(self,node):
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        # if node.children:
        #     if node.children[0].name == "TpConcList":
        #         tnode = node.children[0]
        #         for element in tnode.children:
        #             if element.name == "Ele":
        self.conc.insert(0, node.name)
        # print(self.ast_branch)
        self.conc_num += 1
        if self.ast_branch.startswith("Lname") or self.ast_branch.startswith("Lconc"):
            if not node.parent.name == "Lname" and not node.parent.name == "Lconc":
            # if self.target:
                if self.ast_branch.startswith("Lconc"):
                    for partsel in self.partselects :
                    # print(node.name + " " + partsel + "\n")
                        self.add_edge(self.node_dict[node.name], self.node_dict[partsel], label = 1)
                else:
                    target_name = self.target[0]
                    self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
            else:
                self.target.insert(0,node.name)
                if self.if_scope == "TrueBlock" :
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 2)
                if self.if_scope == "FalseBlock" :
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 3)
        if self.ast_branch.startswith("TpParaList") or (node.parent and node.parent.name == "Pexp"):
            function_name = self.func_call
            inputnumber = self.current_scope[function_name][0]
            inputlist = self.current_scope[function_name][1]
            # print(self.parameter)
            # print(inputlist)
            self.add_edge(self.node_dict[node.name], self.node_dict[inputlist[self.parameter]], label = 1)
            self.parameter += 1
        if self.ast_branch.startswith("Cond"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =1)
        if self.ast_branch.startswith("TrueBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =2)
        if self.ast_branch.startswith("FalseBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =3)
        if self.ast_branch == 'Left':   
            self.Source1.insert(0, node.name)
        if self.ast_branch == 'Right': 
            self.Source2.insert(0, node.name)
            target_name = self.target[0]
            operator_name = self.operators[0]
            # print(self.Source1[0])
            # print(self.Source2[0])
            self.add_edge(self.node_dict[self.Source1[0]], self.node_dict[operator_name], label =1)
            self.add_edge(self.node_dict[self.Source2[0]], self.node_dict[operator_name], label =2)
            if self.if_scope == "If":
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
            else:
                self.add_edge(self.node_dict[operator_name], self.node_dict[target_name], label =1)
        if self.ast_branch.startswith("Operator1"):
            self.add_edge(self.node_dict[node.name], self.node_dict[self.operators[0]], label = 1)
            if self.if_scope == "If":
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
            else:
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.target[0]], label = 1)
        # yanlw
        if self.ast_branch.startswith("PartAssign"):
            target_name = self.target[0]
            if target_name.startswith("Lconc"):
                for partsel in self.partselects :
                    # print(node.name + " " + partsel + "\n")
                    self.add_edge(self.node_dict[node.name], self.node_dict[partsel], label = 1)
            if target_name.startswith("PartAssign"): 
                self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
        self.ast_branch = node.name
        self.generic_visit(node)
        self.con_label = 0

    def visit_Idx(self,node):
        node = copy_without_level(node)
        self.add_node(node)
        self.Idx_exp = str(node.children[0].name)
        self.comb_set.add(node.name)
        if node.children:
            if self.fundec == 1:
                node.children[0].name = self.func_list[0] + node.children[0].name
            if node.children[0].name in self.node_dict.keys():
                self.add_edge(self.node_dict[node.children[0].name],self.node_dict[node.name], label = 1)
            i = 1
            while len(node.children) > i and node.children[i].name == "TpExpList":
                tnode = node.children[i]
                # hzq
                if self.ast_branch.startswith("Lconc"):
                    if tnode.children[0]:
                        tnode1 = AnyNode(name = "PartSelect_" + str(self.part))
                        self.partselects.append("PartSelect_" + str(self.part))
                        self.add_node(tnode1)
                        self.comb_set.add(tnode1.name)
                        self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.name], label = 1)

                        tnode2 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode2)
                        self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 3)

                        # print (self.left + int(tnode.children[0].children[0].name))
                        tnode3 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode3)
                        self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 2)

                        # self.add_edge(self.node_dict[self.conc[0]],self.node_dict[tnode1.name], label = self.part + 1)
                        
                        self.left = self.left + 1
                        self.part = self.part + 1

                if tnode.children[0]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[0].children[0].name)
                    self.add_node(tnode2)
                    self.Idx_exp = self.Idx_exp + f"[{str(tnode.children[0].children[0].name)}]"
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = i+1)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = i+2)
                i = i + 1
            if len(node.children) > i and node.children[i].name == "RngC":
                tnode = node.children[i]
                # hzq
                if self.ast_branch.startswith("Lconc"):
                    if tnode.children[0]:
                        tnode1 = AnyNode(name = "PartSelect_" + str(self.part))
                        self.partselects.append("PartSelect_" + str(self.part))
                        self.add_node(tnode1)
                        self.comb_set.add(tnode1.name)
                        self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.name], label = 1)

                        tnode2 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode2)
                        self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 3)

                        self.left = self.left + int(tnode.children[0].children[0].name) -  int(tnode.children[1].children[0].name)
                        # print (self.left + int(tnode.children[0].children[0].name))
                        tnode3 = AnyNode(name = "Constant_" + str(self.left))
                        self.add_node(tnode3)
                        self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 2)
                        
                        # self.add_edge(self.node_dict[self.conc[0]],self.node_dict[tnode1.name], label = self.part + 1)

                        self.left = self.left + 1
                        self.part = self.part + 1
                if tnode.children[0]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[0].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
            # yanlw
            if len(node.children) > i and node.children[i].name == "RngP":
                tnode = node.children[i]
                # print(f"node: {tnode}")
                # print(f"node: {tnode.children[0]}")
                # print(f"node: {tnode.children[1]}")
                if tnode.children[0]:
                    # Assume nodes already created.
                    if tnode.children[0].name == "Var":
                        # print(f"node: {tnode.children[0].children[0]}")
                        self.add_edge(self.node_dict[tnode.children[0].children[0].name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_plus" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
            if len(node.children) > i and node.children[i].name == "RngN":
                tnode = node.children[i]
                if tnode.children[0]:
                    # Assume nodes already created.
                    if tnode.children[0].name == "Var":
                        # print(f"node: {tnode.children[0].children[0]}")
                        self.add_edge(self.node_dict[tnode.children[0].children[0].name],self.node_dict[node.name], label = 2)
                if tnode.children[1]:
                    tnode2 = AnyNode(name = "Constant_neg" + tnode.children[1].children[0].name)
                    self.add_node(tnode2)
                    self.add_edge(self.node_dict[tnode2.name],self.node_dict[node.name], label = 3)
        if self.ast_branch.startswith("Lname"):
            # if self.target:
            if self.target:
                target_name = self.target[0]
            if self.func_list and target_name.startswith(self.func_list[0]):
                self.add_edge(self.node_dict[node.name], self.node_dict[self.casez[0]], label = self.case_num)
            elif node.parent.name == "Lname":
                self.target.insert(0, node.name)
                if self.if_scope == "TrueBlock" :
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 2)
                if self.if_scope == "FalseBlock" :
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 3)
            elif node.parent.name == "TpEqAssign":
                self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("TpVarTp"):
            if self.target:
                target_name = self.target[0]
                self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("TpParaList") or (node.parent and node.parent.name == "Pexp"):
            function_name = self.func_call
            inputnumber = self.current_scope[function_name][0]
            inputlist = self.current_scope[function_name][1]
            # print(self.parameter)
            # print(inputlist)
            self.add_edge(self.node_dict[node.name], self.node_dict[inputlist[self.parameter]], label = 1)
            self.parameter += 1
        if self.ast_branch.startswith("Cond"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =1)
        if self.ast_branch.startswith("TrueBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =2)
        if self.ast_branch.startswith("FalseBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[node_name], label =3)
        if self.ast_branch.startswith("Conc"):
            self.con_label += 1
            # print(node.name)
            # print(self.con_label)
            self.add_edge(self.node_dict[node.name],self.node_dict[self.conc[0]], label = self.con_label)
        if self.ast_branch.startswith("Left"):
            self.Source1.insert(0,node.name)
        if self.ast_branch.startswith("Right"):
            self.Source2.insert(0,node.name)
            if self.target and self.Source1 and self.Source2 and self.operators:
                # Pop stack.
                target_name = self.target[0]
                operator_name = self.operators[0]
                self.add_edge(self.node_dict[self.Source1[0]], self.node_dict[operator_name], label =1)
                self.add_edge(self.node_dict[self.Source2[0]], self.node_dict[operator_name], label =2)
                if self.if_scope == "If":
                    self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
                else:
                    self.add_edge(self.node_dict[operator_name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("Operator1"):
            self.add_edge(self.node_dict[node.name], self.node_dict[self.operators[0]], label = 1)
            if self.if_scope == "If":
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
            else:
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.target[0]], label = 1)
        # self.ast_branch = node.name
        return
        self.generic_visit(node)

    def visit_Operator1(self, node):
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        self.operators.insert(0,node.name)
        self.ast_branch = "Operator1"
        self.generic_visit(node)

    def visit_Operator2(self, node):
        # print(node.name)
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        self.operators.insert(0,node.name)
        # yanlw 23/8/2024
        if self.ast_branch.startswith("Lconc"):
            # print(f"operator2:{node.name}, operate_target: {self.target[0]}, {self.ast_branch}")
            # if self.target:
            target_name = self.target[0]
            # print("target:" + target_name)
            if target_name.startswith("Lconc"):
                for partsel in self.partselects :
                    self.add_edge(self.node_dict[node.name], self.node_dict[partsel], label = 1)
        if self.ast_branch.startswith("Lname"):
            # if self.target:
            target_name = self.target[0]
            self.add_edge(self.node_dict[node.name], self.node_dict[target_name], label = 1)
        self.ast_branch = "Operator2"
        self.generic_visit(node)

    def visit_Num(self, node):
        str = "Constant_"
        if node.children[0]:
            str = str + node.children[0].name + "'"
        if len(node.children) > 1 and node.children[1]:
            if node.children[1].name == "Hex" or node.children[1].name == "Bin" or node.children[1].name == "Dec" or node.children[1].name == "Oct":
                str = str + node.children[1].name[0]
            if node.children[1].children[0]:
                str = str + node.children[1].children[0].name
        str = "Const,"+ node.children[0].name + "," + str
        tnode = AnyNode(name = str)
        # print(tnode.name)
        self.add_node(tnode)
        # print(self.ast_branch)
        if self.ast_branch.startswith("Lname") or self.ast_branch.startswith("Lconc"):
            target_name = self.target[0]
            if "Function" + target_name in self.func_list:
                self.add_edge(self.node_dict[tnode.name], self.node_dict[self.casez[0]], label = self.case_num)
            elif node.parent.name == "TpEqAssign":
                self.add_edge(self.node_dict[tnode.name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("TpVarTp"):
            if self.target:
                target_name = self.target[0]
                self.add_edge(self.node_dict[tnode.name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("TpParaList"):
            function_name = self.func_call
            inputnumber = self.current_scope[function_name][0]
            inputlist = self.current_scope[function_name][1]
            self.add_edge(self.node_dict[tnode.name], self.node_dict[inputlist[self.parameter]], label = 1)
            # print(self.node_dict[inputlist[self.parameter]])
            self.parameter += 1
        if self.ast_branch.startswith("Cond"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[tnode.name], self.node_dict[node_name], label =1)
        if self.ast_branch.startswith("TrueBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[tnode.name], self.node_dict[node_name], label =2)
        if self.ast_branch.startswith("FalseBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[tnode.name], self.node_dict[node_name], label =3)
        if self.ast_branch.startswith("TpCase"):
            node_name = self.casez[0]
            self.label_set.add(tnode.name)
            self.add_edge(self.node_dict[tnode.name], self.node_dict[node_name], label = self.case_num)
        if self.ast_branch.startswith("Left"):
            self.Source1.insert(0, tnode.name)
        if self.ast_branch.startswith("Right"):
            self.Source2.insert(0, tnode.name)
            if self.target and self.Source1 and self.Source2 and self.operators:
                # Pop stack.
                target_name = self.target[0]
                operator_name = self.operators[0]
                # print(self.Source1[0])
                # print(self.Source2[0])
                self.add_edge(self.node_dict[self.Source1[0]], self.node_dict[operator_name], label =1)
                self.add_edge(self.node_dict[self.Source2[0]], self.node_dict[operator_name], label =2)
                # print(f"{target_name}+ {operator_name}")
                # print(f"{ self.node_dict[target_name]}  ")
                # yanlw 26/8/2024
                if self.node_dict.get(target_name):
                    self.add_edge(self.node_dict[operator_name], self.node_dict[target_name], label =1)
        if self.ast_branch.startswith("Conc") or self.ast_branch.startswith("Lconc"):
            self.con_label += 1
            self.add_edge(self.node_dict[tnode.name],self.node_dict[self.conc[0]], label = self.con_label)
        if self.ast_branch.startswith("Operator1"):
            self.add_edge(self.node_dict[tnode.name], self.node_dict[self.operators[0]], label = 1)
            self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.target[0]], label = 1)
        return
        self.generic_visit(node)

    def visit_Default(self,node):
        node = copy_without_level(node)
        self.add_node(node)
        self.label_set.add(node.name)
        self.add_edge(self.node_dict[node.name], self.node_dict[self.casez[0]], label = self.case_num)
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_Var(self, node):
        if self.fundec == 1:
            node.children[0].name = self.func_list[0] + node.children[0].name
        if not self.in_nodes(node.children[0]):
            # print(f"node : {node.children[0].name}")
            tnode = copy_without_level(node.children[0])
            self.add_node(tnode)
            # self.wire_set.add(node.children[0].name)
            self.sigset_set.add(node.children[0].name)  # Intermediate variable.
        # hzq
        if self.ast_branch.startswith("Lconc") or (self.ast_branch.startswith("PartAssign") and node.parent.name == "Ele"):
            if node.children[0]:
                rang = self.vars[node.children[0].name].Range
                tnode1 = AnyNode( name = "PartSelect_" + str(self.part))
                self.partselects.append("PartSelect_" + str(self.part))
                self.add_node(tnode1)
                self.comb_set.add(tnode1.name)
                self.add_edge(self.node_dict[tnode1.name],self.node_dict[node.children[0].name], label = 1)

                tnode2 = AnyNode(name = "Constant_" + str(self.left))
                self.add_node(tnode2)
                self.add_edge(self.node_dict[tnode2.name],self.node_dict[tnode1.name], label = 3)

                self.left = self.left + rang[0] - rang[1]
                # print (self.left + int(tnode.children[0].children[0].name))
                tnode3 = AnyNode(name = "Constant_" + str(self.left))
                self.add_node(tnode3)
                self.add_edge(self.node_dict[tnode3.name],self.node_dict[tnode1.name], label = 2)
                
                # self.add_edge(self.node_dict[self.conc[0]], self.node_dict[tnode1.name], label = self.part + 1)

                self.left = self.left + 1
                self.part = self.part + 1
        # yanlw
        if node.parent.name == "TpEqAssign" and self.target:
            target_name = self.target[0]
            if self.func_list and target_name.startswith(self.func_list[0]):
                self.add_edge(self.node_dict[node.children[0].name], self.node_dict[self.casez[0]], label = self.case_num)
            elif self.ast_branch.startswith("PartAssign"):
                self.add_edge(self.node_dict[node.children[0].name], self.node_dict[target_name], label =1)
            else:
                self.add_edge(self.node_dict[node.children[0].name], self.node_dict[target_name], label =1)
        if self.ast_branch.startswith("TpVarTp"):
            if self.target:
                target_name = self.target[0]
                self.add_edge(self.node_dict[node.children[0].name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("Lname"):
            # if node.parent.parent.children[1] and node.parent.parent.children[1].name == "Cond":
            #     self.add_edge(self.node_dict["Cond"], self.node_dict[node.children[0].name], label = 1)
            # else:
            # print(node.children[0].name)
            if node.parent.name == "Lname" or node.parent.name == "Lconc":
            # if self.target:
                self.target.insert(0,node.children[0].name)
                target_name = node.children[0].name
                if self.if_scope == "TrueBlock" :
                    # print("TrueBlock")
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 2)
                if self.if_scope == "FalseBlock" :
                    # print("FalseBlock")
                    self.add_edge(self.node_dict[target_name],self.node_dict[self.conditions[0]], label = 3)
        if self.ast_branch.startswith("TpParaList") or node.parent.name == "Pexp":
            function_name = self.func_call
            inputnumber = self.current_scope[function_name][0]
            inputlist = self.current_scope[function_name][1]
            # print(self.parameter)
            # print(inputlist)
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[inputlist[self.parameter]], label = 1)
            self.parameter += 1
        if self.ast_branch.startswith("Cond"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[node_name], label =1)
        if self.ast_branch.startswith("TrueBlock"):
            node_name = self.conditions[0]
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[node_name], label =2)
        if self.ast_branch.startswith("FalseBlock"):
            node_name = self.conditions[0]
            # print(f"node_name:{node_name}")
            # print(node.children[0].name)
            print(self.conditions[0])
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[node_name], label =3)
        if self.ast_branch.startswith("Left"):
            self.Source1.insert(0, node.children[0].name)
        if self.ast_branch.startswith("Right"):
            # print(self.Source1[0])
            self.Source2.insert(0, node.children[0].name)
            if self.target and self.Source1 and self.Source2 and self.operators:
                # Pop stack.
                target_name = self.target[0]
                operator_name = self.operators[0]
                self.add_edge(self.node_dict[self.Source1[0]], self.node_dict[operator_name], label =1)
                self.add_edge(self.node_dict[self.Source2[0]], self.node_dict[operator_name], label =2)
                if self.if_scope == "If":
                    self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
                else:
                    self.add_edge(self.node_dict[operator_name], self.node_dict[target_name], label = 1)
        if self.ast_branch.startswith("Conc"):
            self.con_label += 1
            self.add_edge(self.node_dict[node.children[0].name],self.node_dict[self.conc[0]], label = self.con_label)
        if self.ast_branch.startswith("Operator1"):
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[self.operators[0]], label = 1)
            if self.if_scope == "If":
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.conditions[0]], label = 1)
            else:
                self.add_edge(self.node_dict[self.operators[0]], self.node_dict[self.target[0]], label = 1)
        if self.ast_branch.startswith("Casez"):
            self.case_num += 1
            tnode = copy_without_level(node.children[0])
            self.add_edge(self.node_dict[node.children[0].name], self.node_dict[self.casez[0]], label = self.case_num)
        if self.ast_branch == "Ufname":
            self.func_call = "Function" + node.children[0].name
            self.dfg.remove_node(self.node_dict[node.children[0].name])
            # print("call:" + self.func_call)
        self.generic_visit(node)

    def visit_Sfname(self,node):
        self.func_call = "Function" + node.children[0].name
        assert False,"Exist SysFunName!"

    def visit_Casez(self,node):
        node = copy_without_level(node)
        self.add_node(node)
        self.comb_set.add(node.name)
        self.casez.insert(0, node.name)
        self.case_num = 0
        if self.func_list:
            function_name = self.func_list[0]
            self.func_dict[function_name] = node.name
        #     inum = self.current_scope[function_name][0]
        #     self.case_num += inum
        #     for i in range(inum):
        #         node_name = self.current_scope[function_name][1][i]
        #         self.add_edge(self.node_dict[node_name], node, label = i+1)
        if self.if_scope:
            if node.parent.name == "TrueBlock":
                tlabel = 2
            if node.parent.name == "FalseBlock":
                tlabel = 3
            self.add_edge(self.node_dict[node.name],self.node_dict[self.conditions[0]], label = tlabel)
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_TpCase(self,node):
        self.ast_branch = node.name
        self.case_num += 1
        self.generic_visit(node)

    def visit_Funcall(self,node):
        self.func_left = self.ast_branch
        self.ast_branch = node.name
        self.generic_visit(node)
        self.parameter = 0

    def visit_TpParaList(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_Ufname(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)

    def visit_TpParaList(self,node):
        self.ast_branch = node.name
        self.generic_visit(node)
        # print(self.func_left)
        if self.ast_branch.startswith("Lconc") or self.func_left.startswith("Lconc"):
            # if self.target:
            target_name = self.target[0]
            # print("target:" + target_name)
            if target_name.startswith("Lconc"):
                # print(self.partselects)
                # print(self.partselects)
                for partsel in self.partselects :
                    # print(node.name + " " + partsel + "\n")
                    self.add_edge(self.node_dict[self.func_dict[self.func_call]], self.node_dict[partsel], label = 1)
        else:
            self.add_edge(self.node_dict[self.func_dict[self.func_call]], self.node_dict[self.target[0]], label = 1)
        self.parameter = 0

    def find_node(self, node):
        if self.dfg.has_node(node):
            return self.dfg.nodes[node]
        else:
            return None

    # Find the specified edge.
    def find_edge(self, u, v):
        if self.dfg.has_edge(u, v):
            return self.dfg[u][v]
        else:
            return None
        
    def predecessor(self,node):
        predecessors = list(self.dfg.predecessors(node))
        # print(f"All nodes pointing to {node}: {predecessors}")
        return predecessors
    
    def successor(self,node):
        successors = list(self.dfg.successors(node))
        # print(f"All nodes reachable from {node}: {successors}")
        return successors
    
    def is_function_var(self,node):
        if node.name.startswith("Function"):
            return True
        else:
            return False
    
    # Extra handling for functions generated from case.
    def case_function(self):
        for node_name in self.comb_set:
            if node_name.startswith("Casez"):
                # Find all nodes pointing to this node.
                predecessors = self.predecessor(self.node_dict[node_name])
                for node in predecessors:
                    # print(predecessors)
                    if node.name.startswith("Const") or node.name.startswith("Idx"):
                        self.delete_node(node)
                tpredecessors = self.predecessor(self.node_dict[node_name])
                for node in tpredecessors:
                    if node.name.startswith("Const"):
                        self.dfg.remove_edge(node,self.node_dict[node_name])
                predecessors = self.predecessor(self.node_dict[node_name])
                for node in predecessors:
                    if node.name == "Default":
                        self.delete_node(node,False)
                filter_predecessors = list(filter(self.is_function_var,predecessors))
                # print(filter_predecessors)
                s_num = 0
                b_num = 0
                for node in filter_predecessors:
                    if node.name[-1] == 'a':
                        # print("a")
                        a_predecessors = self.predecessor(node)
                        self.delete_node(node,False,"a")
                    if node.name[-1] == 's':
                        # print("s")
                        s_predecessors = self.predecessor(node)
                        if len(s_predecessors) == 1 and s_predecessors[0].name.startswith("Conc"):
                            Conc_node = s_predecessors[0]
                            Conc_predecessors = self.predecessor(Conc_node)
                            for predecessor in Conc_predecessors:
                                # print(predecessor)
                                s_num = s_num + self.number_of_edges(predecessor,Conc_node)
                            self.delete_node(node,False,"s")
                            self.delete_node(Conc_node,True,"s")
                    if node.name[-1] == 'b':
                        # print("b")
                        b_predecessors = self.predecessor(node)
                        if len(b_predecessors) == 1 and b_predecessors[0].name.startswith("Conc"):
                            Conc_node = b_predecessors[0]
                            Conc_predecessors = self.predecessor(Conc_node)
                            for predecessor in Conc_predecessors:
                                b_num = b_num + self.number_of_edges(predecessor,Conc_node)
                            # print(s_num)
                            # print(b_num)
                            # assert b_num == s_num,"Case Error!"
                            self.delete_node(node,False,"b")
                            self.delete_node(Conc_node,True,"b")
                        else:
                            assert False,"Error, Case_b unable to process"
                predecessors = self.predecessor(self.node_dict[node_name])
    
    def number_of_edges(self,node1,node2):
        return self.dfg.number_of_edges(node1,node2)
                        
    # Return the edge label.
    def edge_label(self, node1, node2):
        # Iterate over the set and check for a matching edge.
        for edge in self.edge_set:
            if edge[0] == node1.name and edge[1].startswith(node2.name):
                return edge[2]
        return None

    
    # Delete a node and reconnect a single edge.
    def delete_node(self,node,tag = True,parameter = None):
        if node:
            predecessors = self.predecessor(node)
            if len(self.successor(node)) == 1:
                target_node = self.successor(node)[0]
                for subnode in predecessors:
                    if parameter == "s":
                        if tag == True:
                            edges_number_list = self.edge_dict[(subnode.name.split(',')[-1], node.name.split(',')[-1])]
                            for edges_number in edges_number_list:
                                # print(f"edge:{edges_number}")
                                label = edges_number * 2 - 1
                                self.add_edge(subnode,target_node,label)  
                        else:
                            label = self.edge_label(node,target_node) * 2 - 1
                            self.add_edge(subnode,target_node,label)
                    elif parameter == "b":
                        if tag == True:
                            # edges_ab = list(self.dfg.edges(subnode, node, keys=True))
                            # edges_ab = [ edge for edge in edges_ab if edge[0] == subnode and edge[1] == node ]
                            # print(self.edge_dict[(subnode.name, node.name)])
                            edges_number_list = self.edge_dict[(subnode.name.split(',')[-1], node.name.split(',')[-1])]
                            for edges_number in edges_number_list:
                                # print(f"edge:{edges_number}")
                                label = edges_number * 2
                                self.add_edge(subnode,target_node,label) 
                        else:
                            label = self.edge_label(node,target_node) * 2 
                            self.add_edge(subnode,target_node,label)
                    elif parameter == "a":
                        if tag == True:
                            label = self.edge_label(subnode,node) * 2
                            self.add_edge(subnode,target_node,label)
                        else:
                            label = self.edge_label(node,target_node) * 2 
                            self.add_edge(subnode,target_node,label)
                    else:
                        if tag == True:
                            label = self.edge_label(subnode,node)
                            self.add_edge(subnode,target_node,label)
                        else:
                            label = self.edge_label(node,target_node)
                            self.add_edge(subnode,target_node,label)
            # print(f"delete node : {node.name}")
            self.dfg.remove_node(node)
            # print(f"Node {node} has been removed")

    def remove_isolated_nodes(self):
        isolated_nodes = [node for node in self.dfg.nodes if self.dfg.degree(node) == 0]
        for node in isolated_nodes:
            self.dfg.remove_node(node)
            # print(f"Isolated node {node} has been removed")

    
    # Add simplification strategy.
    def simplify_Idx(self):
        u = self.node_dict["Add_1"]
        v = self.node_dict["_00_"]
        if self.find_edge(u,v):
            print("find edge!")

    def add_node(self,node):
        if not self.in_nodes(node):
            self.dfg.add_node(self.node_dict[node.name])

    def add_edge(self,node1,node2,label):
        if not (node1.name,node2.name,label) in self.edge_set:
            self.dfg.add_edge(node1,node2,label = label)
            # Use setdefault to simplify dict initialization.
            self.edge_dict.setdefault((node1.name.split(',')[-1], node2.name.split(',')[-1]), []).append(label)
            # print(f"{(node1.name,node2.name)} : {label}")
        self.edge_set.add((node1.name,node2.name,label))

    def remove_same_node(self):
        nodelist = []
        removelist = []
        for node in self.dfg.nodes:
            if self.in_nodes:
                removelist.append(node)
            else:
                nodelist.append(node)
    
    def remain_Idx(self):
        if self.partselects:
            exist_partselect = self.partselects[-1]
            index = int(exist_partselect[-1]) + 1
        else:
            index = 0
        for node in self.dfg.nodes:
            if node.name.startswith("Idx"):
                node.name = "PartSelect_" + str(index)
                self.node_dict[node.name] = node
                self.comb_set.add(node.name)
                index += 1
                # self.edge_dict[(node.name,)]


        # for node_name in removelist:
        #     print(self.dfg.nodes)
        #     self.dfg.remove_node(node)

    def node_name(self):
        inset= set()
        outset = set()
        sigset = set()
        combset = set()
        self.remove_same_node()
        for node in self.dfg.nodes:
            node_name = node.name
            if node_name.startswith("Conc"):
                self.node_dict[node_name].name = node_name.replace('Conc_', 'Concat_')
            if node_name.startswith("Casez"):
                self.node_dict[node_name].name = node_name.replace('Casez_', 'Case_')
            if node_name in self.in_set:
                left = self.vars[node_name].Range[0]
                right = self.vars[node_name].Range[1]
                Range = left - right + 1
                newname = "Input," + str(Range) + "," + self.node_dict[node_name].name
                self.node_dict[node_name].name = newname
                inset.add(self.node_dict[node_name].name)
            if node_name in self.out_set:
                left = self.vars[node_name].Range[0]
                right = self.vars[node_name].Range[1]
                Range = left - right + 1
                if self.vars[node_name].type:
                    newname = "Output" + "_" + self.vars[node_name].type + "," + str(Range) + "," + self.node_dict[node_name].name
                else:
                    newname = "Output," + str(Range) + "," + self.node_dict[node_name].name
                self.node_dict[node_name].name = newname
                outset.add(self.node_dict[node_name].name)
            if node_name in self.sigset_set and not node_name.startswith("Function_"):
                left = self.vars[node_name].Range[0]
                right = self.vars[node_name].Range[1]
                Range = left - right + 1
                newname = self.vars[node_name].type + "," +  str(Range) + "," + self.node_dict[node_name].name
                self.node_dict[node_name].name = newname
                sigset.add(self.node_dict[node_name].name)
            if node_name in self.comb_set:
                operator_type = re.sub(r'_\d+$', '', self.node_dict[node_name].name)
                newname = operator_type + ",Null," + self.node_dict[node_name].name
                self.node_dict[node_name].name = newname
                combset.add(self.node_dict[node_name].name)
            if node_name.startswith("Constant_"):
                newname = "Const,Null," + node.name
                node.name = newname
        self.in_set = inset
        self.out_set = outset
        self.sigset_set = sigset
        self.comb_set = combset

    # Merge multiple nodes with the same name; keep the first node name.
    def node_merge(self,node_name_list,type):
        if type == "PartAssign":
            merge_node_name = node_name_list[0]
            exist_edge_num = 3
            # print(node_name_list)
            for node_name in node_name_list[1:]:
                predecessors = self.predecessor(self.node_dict[node_name])
                predecessors_len = len(predecessors)
                # print(predecessors_len)
                for node in predecessors:
                    # print("--------------------------------")
                    # print((node.name,node_name))
                    # print(self.edge_dict[(node.name.split(',')[-1],node_name.split(',')[-1])])
                    for edge_num in self.edge_dict[(node.name.split(',')[-1],node_name.split(',')[-1])]:
                        edge_label = edge_num
                        self.add_edge(node, self.node_dict[merge_node_name], label = exist_edge_num + edge_label)
                        # a[0] = b 
                exist_edge_num += 3

    def partassign_merge(self):
        remove_list = []
        for node in self.dfg.nodes:
            predecessors = self.predecessor(node)
            names = [node.name for node in predecessors]
            partassign_names = [name for name in names if name.startswith('PartAssign')]
            remove_list = remove_list + partassign_names[1:]
            # Check if there are multiple.
            if len(partassign_names) > 1:
                # print(f"{node.name} has multiple strings with 'partassign' prefix")
                self.node_merge(partassign_names,"PartAssign")
            else:
                ()
        remove_node_list = [self.node_dict[node_name] for node_name in remove_list]
        self.dfg.remove_nodes_from(remove_node_list)

    def partassign_function(self):
        remove_list = []
        add_list = []
        for node in self.dfg.nodes:
            if node.name.startswith("PartAssign"):
                predecessors = self.predecessor(node)
                names = [node.name for node in predecessors]
                tnode = AnyNode(name = "Conc_" + str(self.conc_num + 1))
                self.conc_num += 1
                self.comb_set.add(tnode.name)
                self.node_dict[tnode.name] = tnode
                partassign_class = [[] for _ in range(math.ceil(len(names) / 3))] 
                for name in names:
                    # print("--------------------------------")
                    # print(self.edge_dict[(name,node.name)])
                    for edge_label in self.edge_dict[(name.split(',')[-1],node.name.split(',')[-1])]:
                        # print(f"{(name,node.name)}:{edge_label}")
                        class_num = int((edge_label+2) / 3)
                        # Ensure the sublist for class_num exists.
                        if class_num >= len(partassign_class):
                            partassign_class.extend([[] for _ in range(class_num - len(partassign_class) + 1)])

                        # Ensure sublist length allows insertion at index 1 or 2.
                        while len(partassign_class[class_num]) < 3:
                            partassign_class[class_num].append(None)  # Pad with None.

                        if (edge_label+2) % 3 == 0:
                            partassign_class[class_num][0] = name
                        if (edge_label+2) % 3 == 1:
                            partassign_class[class_num][1] = name
                        if (edge_label+2) % 3 == 2:
                            partassign_class[class_num][1] = name
                sorted_partassign_class = sorted(partassign_class, key=lambda x: x[1] if len(x) > 1 else '')
                # print(sorted_partassign_class)
                new_edge_label = (len(sorted_partassign_class) - 1)
                # print(new_edge_label)
                for part_assign in sorted_partassign_class:
                    if part_assign:
                        # print((self.node_dict[part_assign[0]],tnode,new_edge_label))
                        add_list.append((self.node_dict[part_assign[0]],tnode,new_edge_label))
                        new_edge_label -= 1
                successors = self.successor(node)
                for suc_node in successors:
                    add_list.append((tnode,suc_node,1))
                remove_list.append(node.name)
        for add_edge in add_list:
            self.add_edge(add_edge[0],add_edge[1],int(add_edge[2]))
        remove_node_list = [self.node_dict[node_name] for node_name in remove_list]
        self.dfg.remove_nodes_from(remove_node_list)

    # Remove wire nodes.
    def remove_wires(self):
        # Find all wire variables to delete.
        wires_to_remove = [node for node in self.dfg.nodes if isinstance(node.name, str) and node.name.startswith("_") and node.name.endswith("_")]
        print(wires_to_remove)
        print(self.edge_dict)
        
        for wire in wires_to_remove:
            predecessors = list(self.dfg.predecessors(wire))
            successors = list(self.dfg.successors(wire))
            
            # Check wire predecessors and successors first.
            if predecessors and successors:
                # If wire has both predecessors and successors, connect them.
                for pred in predecessors:
                    for succ in successors:
                        label = self.edge_dict[(wire.name.split(',')[-1],succ.name.split(',')[-1])]
                        print(f"({wire.name},{succ.name}):{label[0]}")
                        self.add_edge(pred, succ, label[0])  # Connect predecessor and successor.
            
            # Delete the wire last.
            self.dfg.remove_node(wire)
    
    def extract_name(self, text):
        match = re.match(r"([^,]+),([^,]+),([^,]+)", text) 
        if match:
            part1 = match.group(1)  # Extract part 1.
            part2 = match.group(2)  # Extract part 2.
            part3 = match.group(3)  # Extract part 3.
            if part1 == "Const" and part2 != "Null":
                return text
            return part3
        else:
            raise ValueError("Input format mismatch; cannot extract parts.")
    
    def print_graph(self, G):
        # Print each edge and its attributes, including label.
        for edge in G.edges(data=True):
            print(edge)
        for node in G.nodes(data=True):
            print(node)

    # def load_dot_file(self, dot_file_path):
        # Read dot file and convert to a networkx graph.
        # A = pgv.AGraph(dot_file_path)
        # G = nx.nx_agraph.from_agraph(A)
        # return G

    # def find_boundary_nodes(self, G, link_node, subgraph_nodes):
    #     # Find boundary nodes that connect to external nodes in the current subgraph.
    #     boundary_nodes = set()
    #     for sub_node in subgraph_nodes :
    #         if sub_node != link_node:
    #             for neighbor in G.neighbors(sub_node):
    #                 if neighbor not in subgraph_nodes:
    #                     boundary_nodes.add(sub_node)
    #                     # break  # Exit loop after finding boundary node.
    #     return boundary_nodes
    def find_boundary_nodes(self, G, node, subgraph_nodes):
        # Find subgraph input/output boundary nodes.
        boundary_nodes = set()
        for n in subgraph_nodes:
            # If a node predecessor or successor is in the full graph but not in the subgraph, it is a boundary node.
            for pred in G.predecessors(n):
                if pred not in subgraph_nodes:
                    boundary_nodes.add(pred)
            for succ in G.successors(n):
                if succ not in subgraph_nodes:
                    boundary_nodes.add(succ)
        return boundary_nodes

    def add_predecessors(self, graph, subgraph, node):
        # Skip nodes already in the subgraph.
        if node in subgraph.G.nodes :
            return
        # Add current node.
        # print(node)
        subgraph.G.add_node(node)
        
        # Collect and traverse all predecessors.
        predecessors = list(graph.predecessors(node))
        for predecessor in predecessors:
            
            # Recursively traverse each predecessor's predecessors.
            self.add_predecessors(graph, subgraph, predecessor)
            pre_name = self.extract_name(predecessor.name)
            node_name = self.extract_name(node.name)
            subgraph.G.add_edge(predecessor, node, label = self.edge_dict[(pre_name.split(',')[-1],node_name.split(',')[-1])][0])  # Add edge.


    # def split_graph(self, G, rule_function):

    #     # Split graph into subgraphs by the given rule.
    #     subgraphs = {}
    #     for node in list(G.nodes):
    #         tag = rule_function(node)
    #         if tag is not None:
    #             node_list = self.predecessor(node)
    #             print(node_list)
    #             for node in node_list:
    #                 self.block_num = self.block_num + 1
    #                 key = tag + "_" + str(self.block_num)
    #                 subgraphs[key] = Subgraph(Graph = nx.DiGraph(),name = key)
                    
    #                 # print(list(G.predecessors(node)))
    #                 self.add_predecessors(G, subgraphs[key], node)


    #                 boundary_nodes = self.find_boundary_nodes(G, node, subgraphs[key].G.nodes)
    #                 print(boundary_nodes)

    #                 # Remove non-shared nodes from the original graph.
    #                 for node in subgraphs[key].G.nodes:
    #                     if not node in boundary_nodes:  # Delete nodes referenced by only one subgraph.
    #                         G.remove_node(node)
    #     print()
    #     subgraphs["BasicBlock_Remain"] = Subgraph(Graph = G,name = "BasicBlock_Remain")

    #     return subgraphs

    # Updated splitting algorithm.
    def split_graph(self, G, rule_function):
        # Split graph into subgraphs by the given rule.
        subgraphs = {}
        visited_nodes = set()  # Track processed nodes to avoid duplicates.

        for node in list(G.nodes):
            if node in visited_nodes:
                continue  # Skip processed nodes.

            tag = rule_function(node)  # Determine which subgraph the node belongs to.
            if tag is not None:
                self.block_num += 1
                block_name = f"{tag}_{self.block_num}"

                # Create a new subgraph.
                subgraphs[block_name] = Subgraph(Graph=nx.DiGraph(), name=block_name)

                # Add current node and its predecessors/successors.
                self.add_predecessors_and_successors(
                    G, subgraphs[block_name], node, visited_nodes
                )

                # Find subgraph boundary nodes (inputs/outputs).
                boundary_nodes = self.find_boundary_nodes(
                    G, node, subgraphs[block_name].G.nodes
                )
                print(f"Boundary nodes for {block_name}: {boundary_nodes}")

        # Put remaining nodes into a separate subgraph.
        subgraphs["BasicBlock_Remain"] = Subgraph(Graph=G, name="BasicBlock_Remain")
        return subgraphs
    
    def add_predecessors_and_successors(self, G, subgraph, node, visited_nodes):
        # Traverse node and its predecessors/successors and add them to subgraph.
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited_nodes:
                subgraph.G.add_node(current, **G.nodes[current])  # Add to subgraph.
                visited_nodes.add(current)

                # Traverse predecessors.
                for pred in G.predecessors(current):
                    label = self.edge_dict[(pred.name.split(',')[-1],current.name.split(',')[-1])]
                    subgraph.G.add_edge(pred, current,label=label)
                    # stack.append(pred)

                # Traverse successors.
                for succ in G.successors(current):
                    label = self.edge_dict[(current.name.split(',')[-1],succ.name.split(',')[-1])]
                    subgraph.G.add_edge(current, succ,label=label)
                    # stack.append(succ)


    def visualize_subgraphs(self, subgraphs, outpath):
        # Visualize each subgraph.
        for key, subgraph in subgraphs.items():  # Use items() iteration.
            os.makedirs(outpath+"/basicblock",exist_ok=True)
            out_path = os.path.join(outpath+"/basicblock", key + ".dot")
            self.show_graph(subgraph.G,out_path)

    # def basic_block(self,outpath):
    #     G = self.dfg.copy()
    #     # remove_wires(G)
    #     # Define a rule function, e.g., group by the first letter of the node name.
    #     def rule_function(node):
    #         if node is None or str(node.name).strip() == "":
    #             return None  # Or return a specific default name.
    #         if str(node.name).startswith("Cond"):
    #             return "BasicBlock"  # Use Cond nodes as grouping criteria.
    #         # successors = self.successor(node)
    #         # if len(successors) == 1 and successors[0].name.startswith("Cond"):
    #         #     self.block_num = self.block_num + 1
    #         #     block_name = "BasicBlock" + str(self.block_num)
    #         #     return block_name  # Use Cond nodes as grouping criteria.
    #         return None
        
    #     subgraphs = self.split_graph(G, rule_function)
    #     self.visualize_subgraphs(subgraphs,outpath)

    #     return subgraphs

    def basic_block(self,outpath):
        G = self.dfg.copy()
        # remove_wires(G)
        # Define a rule function, e.g., group by the first letter of the node name.
        def rule_function(node):
            if node is None or str(node.name).strip() == "":
                return None  # Or return a specific default name.
            # Part 2: operator-node grouping rule.
            # Check if the node is an operator (e.g., Add, Mul).
            divide_operators = ["ShiftLeft",
                        "ShiftRight",
                        "AshiftLeft",
                        "AshiftRight",
                        "LNot",
                        "And",
                        "Or",
                        "Not",
                        "BitAnd",
                        "BitOr",
                        "BitXor",
                        "BitNXor",
                        "URxor",
                        "URand",
                        "URor",
                        "URnand",
                        "URnor",
                        "URnxor",
                        "Cond", # cond is an operator too and can be included.
                        "Eq",
                        "Neq",
                        "Eeq",
                        "Neeq",
                        "Le",
                        "Ge",
                        "Lt",
                        "Gt",
                        "Add", 
                        "Sub", 
                        "Mul", 
                        "Div", 
                        "Mod",
                        "Power",
                        "Funcall" ]  # Extend the operator list as needed.
            for op in divide_operators:
                if str(node.name).startswith(op):
                    #if str(node.name).startswith("Add"):
                    return "BasicBlock"  # Use Cond nodes as grouping criteria.
                        
                    # successors = self.successor(node)
                    # if len(successors) == 1 and successors[0].name.startswith("Cond"):
                    #     self.block_num = self.block_num + 1
                    #     block_name = "BasicBlock" + str(self.block_num)
                    #     return block_name  # Use Cond nodes as grouping criteria.
            return None
        
        subgraphs = self.split_graph(G, rule_function)
        self.visualize_subgraphs(subgraphs,outpath)

        return subgraphs

    
    def Preprocessing(self):
        self.case_function()
        self.remove_isolated_nodes()
        self.partassign_merge()
        self.partassign_function()
        self.remain_Idx()
        # self.remove_wires()
        self.node_name()

    def show_basic_block_combine_graph(self, subgraphs = None, filename = None):
        top_name = 'Combined'
        with open (filename, 'w') as f:
            line = "digraph {0} ".format(top_name)
            line = line + "{\n"
            f.write(line)
            if subgraphs != None:
                for key,subgraph in subgraphs.items():
                    line = "    subgraph cluster_{0} ".format(subgraph.name)
                    line = line + "{\n"
                    f.write(line)
                    line = "        label=\"{0}\";\n".format(subgraph.name)
                    f.write(line)
                    # Write node to graph file
                    for node in subgraph.G.nodes:
                        if not node:
                            break
                        node_name = re.sub(r'\.|\[|\]|\\', r'_', node.name)
                        if node.name in self.in_set:
                            line = "        \"{0}\" [style=filled, color=yellow];\n".format(node_name)
                        elif node.name in self.out_set:
                            line = "        \"{0}\" [style=filled, color=green];\n".format(node_name)
                        elif node.name in self.sigset_set:
                            line = "        \"{0}\" [style=filled, color=orange];\n".format(node_name)
                        # elif node.name in self.seq_set:
                        #     line = "    {0} [style=filled, color=lightblue];\n".format(node_name)
                        elif node.name in self.wire_set:
                            line = "        \"{0}\" [style=filled, color=black, fillcolor=white];\n".format(node_name)
                        elif node.name.startswith("Const"):
                            if node.name in self.label_set:
                                line = "        \"{0}\" [style=filled, color=red];\n".format(node_name)
                            else:
                                line = "        \"{0}\" [style=filled, color=grey];\n".format(node_name)
                        elif node.name in self.label_set:
                            line = "        \"{0}\" [style=filled, color=red];\n".format(node_name)
                        elif node.name in self.comb_set:
                            line = "        \"{0}\" [style=filled, color=pink];\n".format(node_name)
                        else:
                            line = "        {0};\n".format(node_name)
                        f.write(line)

                    # # Write edge to graph file
                    # for vertice in subgraph.G.edges(data = True):
                    #     if vertice:
                    #         u = vertice[0]
                    #         v = vertice[1]
                    #         label = vertice[2]["label"]
                    #         node1_name = re.sub(r'\.|\[|\]|\\', r'_', u.name)
                    #         node2_name = re.sub(r'\.|\[|\]|\\', r'_', v.name)
                            
                    #         # if node1_name.startswith("Const"):
                    #         #     node1_name = "\"" + node1_name + "\""
                    #         # if node2_name.startswith("Const"):
                    #         #     node2_name = "\"" + node2_name + "\""
                    #         # pair = '{0} -> {1} [label=\"{2}\"]'.format(vertice, val, edge_id)
                                    
                    #         # reverse the ast to CDFG
                    #         pair = '\"{0}\" -> \"{1}\" [label=\"{2}\"]'.format(node1_name, node2_name, label)

                    #         line = "        {0};\n".format(pair)
                    #         f.write(line)

                    f.write("   }\n")

                # Write edge to graph file
                for vertice in self.dfg.edges(data = True):
                    if vertice:
                        u = vertice[0]
                        v = vertice[1]
                        label = vertice[2]["label"]
                        node1_name = re.sub(r'\.|\[|\]|\\', r'_', u.name)
                        node2_name = re.sub(r'\.|\[|\]|\\', r'_', v.name)
                            
                        # if node1_name.startswith("Const"):
                        #     node1_name = "\"" + node1_name + "\""
                        # if node2_name.startswith("Const"):
                        #     node2_name = "\"" + node2_name + "\""
                        # pair = '{0} -> {1} [label=\"{2}\"]'.format(vertice, val, edge_id)
                                    
                        # reverse the ast to CDFG
                        pair = '\"{0}\" -> \"{1}\" [label=\"{2}\"]'.format(node1_name, node2_name, label)

                        line = "    {0};\n".format(pair)
                        f.write(line)

                f.write("}\n")

                    
    def resolve_var_exp(self):
        resolved = self.var_exp.copy()

        def replace_in_value(value, resolved_dict):
            # Traverse resolved_dict to perform replacements.
            for key, replacement in resolved_dict.items():
                if key in value:
                    # Wrap in parentheses to preserve precedence.
                    value = value.replace(key, f"{replacement}")
            return value

        while True:
            progress = False
            for key in list(resolved.keys()):
                original_value = resolved[key]
                resolved[key] = replace_in_value(original_value, resolved)
                if resolved[key] != original_value:
                    progress = True
            # No more progress means the maximal resolution is done.
            if not progress:
                break
        
        return resolved
    # def write_node_edge_to_dot(self,G):

    def show_graph(self, G = None, filename = None):
        # self.get_stat()
        print('----- Writting Graph Visialization File -----')
        # outfile_path = "./png/"
        # outfile = outfile_path+"AST_graph.dot"
        top_name = 'test'
        # print(self.dfg.nodes)

        if G == None:
            G = self.dfg

        with open (filename, 'w') as f:
            line = "digraph {0} ".format(top_name)
            line = line + "{\n"
            f.write(line)
            reg_set = set()

            
            # Write node to graph file
            for node in G.nodes:
                if not node:
                    break
                node_name = re.sub(r'\.|\[|\]|\\', r'_', node.name)
                if node.name in self.in_set:
                    line = "    \"{0}\" [style=filled, color=yellow];\n".format(node_name)
                elif node.name in self.out_set:
                    line = "    \"{0}\" [style=filled, color=green];\n".format(node_name)
                elif node.name in self.sigset_set:
                    line = "    \"{0}\" [style=filled, color=orange];\n".format(node_name)
                # elif node.name in self.seq_set:
                #     line = "    {0} [style=filled, color=lightblue];\n".format(node_name)
                elif node.name in self.wire_set:
                    line = "    \"{0}\" [style=filled, color=black, fillcolor=white];\n".format(node_name)
                elif node.name.startswith("Const"):
                    if node.name in self.label_set:
                        line = "    \"{0}\" [style=filled, color=red];\n".format(node_name)
                    else:
                        line = "    \"{0}\" [style=filled, color=grey];\n".format(node_name)
                elif node.name in self.label_set:
                    line = "    \"{0}\" [style=filled, color=red];\n".format(node_name)
                elif node.name in self.comb_set:
                    line = "    \"{0}\" [style=filled, color=pink];\n".format(node_name)
                else:
                    line = "    {0};\n".format(node_name)
                f.write(line)

            # Write edge to graph file
            for vertice in G.edges(data = True):
                if vertice:
                    u = vertice[0]
                    v = vertice[1]
                    label = vertice[2]["label"]
                    node1_name = re.sub(r'\.|\[|\]|\\', r'_', u.name)
                    node2_name = re.sub(r'\.|\[|\]|\\', r'_', v.name)
                    
                    # if node1_name.startswith("Const"):
                    #     node1_name = "\"" + node1_name + "\""
                    # if node2_name.startswith("Const"):
                    #     node2_name = "\"" + node2_name + "\""
                    # pair = '{0} -> {1} [label=\"{2}\"]'.format(vertice, val, edge_id)
                            
                    # reverse the ast to CDFG
                    pair = '\"{0}\" -> \"{1}\" [label=\"{2}\"]'.format(node1_name, node2_name, label)

                    line = "    {0};\n".format(pair)
                    f.write(line)
            
            f.write("}\n")
        
        print('Finish!\n')

    def Operator_simple(self, operator_name):
        if operator_name.startswith("Add"):
            return "+"
        elif operator_name.startswith("Sub"):
            return "-"
        elif operator_name.startswith("Mul"):
            return "*"
        elif operator_name.startswith("Div"):
            return "/"
        elif operator_name.startswith("Mod"):
            return "%"
        
        elif operator_name.startswith("Eq"):
            return "=="
        elif operator_name.startswith("Neq"):
            return "!="
        elif operator_name.startswith("Gt"):
            return ">"
        elif operator_name.startswith("Lt"):
            return "<"
        elif operator_name.startswith("Ge"):
            return ">="
        elif operator_name.startswith("Le"):
            return "<="
        elif operator_name.startswith("Eeq"):
            return "==="
        elif operator_name.startswith("Neeq"):
            return "!=="
        
        elif operator_name.startswith("And"):
            return "&&"
        elif operator_name.startswith("BitAnd"):
            return "&"
        elif operator_name.startswith("Or"):
            return "||"
        elif operator_name.startswith("BitOr"):
            return "|"
        elif operator_name.startswith("BitXor"):
            return "^"
        elif operator_name.startswith("BitNXor"):
            return "~^"
        elif operator_name.startswith("Not"):
            return "~"
        
        elif operator_name.startswith("ShiftLeft"):
            return "<<"
        elif operator_name.startswith("ShiftRight"):
            return ">>"
        elif operator_name.startswith("AshiftLeft"):
            return "<<<"
        elif operator_name.startswith("AshiftRight"):
            return ">>>"
        
        elif operator_name.startswith("LogicAnd"):
            return "&&"
        elif operator_name.startswith("LogicOr"):
            return "||"
        elif operator_name.startswith("LNot"):
            return "!"
        
        elif operator_name.startswith("URxor"):
            return "^"
        elif operator_name.startswith("URand"):
            return "&"
        elif operator_name.startswith("URor"):
            return "|"
        elif operator_name.startswith("URnand"):
            return "~&"
        elif operator_name.startswith("URnor"):
            return "~|"
        
        # If no match found, return None or raise an exception as appropriate
        return None


    def variable_exp(self,node_name):
        print(self.node_dict[node_name])
        pre_list = self.predecessor(self.node_dict[node_name])


if __name__ == "__main__":
    # node1 = AnyNode(name = "parent", parent = None, level = 6)
    # tnode1 = AnyNode(name = "sss",parent = node1, level = 7)
    # tnode1 = copy_without_level(tnode1)
    # print(tnode1.parent)
    # exit()

    data = read_json_file("./data/75_ast_clean.json")
    ast = build_tree_from_json(data)

    # Preprocess: add suffix to identical keywords.
    pre(ast)
    
    # Extract CDFG; dfg is the extracted graph.
    dfg_extractor = DFGExtractor()
    dfg_extractor.visit(ast)
    dfg_extractor.Preprocessing()
    dfg_extractor.show_graph(filename = "rrrrr.dot")
    exit()
    Subgraphs = dfg_extractor.basic_block(".")
    print((Subgraphs))
    dfg_extractor.show_basic_block_combine_graph(Subgraphs,"control_example_combined")
    # output_directory = "."
    # for key, subgraph in subgraphs.items():
    #     out_path = os.path.join(output_directory, key+"_cdfg.dot")
    #     sub_dfg_extractor = DFGExtractor(subgraph.G)
    #     sub_dfg_extractor.show_graph(out_path)
    dfg = dfg_extractor.dfg

    # print(RenderTree(ast))

    # Extract DFG.
    # dfg = extract_dfg_from_ast(ast)
    input_directory = "./data"
    output_directory = "./png"
    # input_path = os.path.join(input_directory, file_name)
    out_path1 = os.path.join(output_directory, "comparator_ast_clean_cdfg.dot")
    out_path2 = os.path.join(output_directory, "comparator_ast_clean_cdfg.png")

    # print(dfg.nodes)
    # print(dfg.edges(data=True))
    dfg_extractor.show_graph(out_path1)

    # for edg in dfg.edges :
    #     print(edg[0])

    out_path = "./png"
    # print(dfg_extractor.out_set)
    # print(dfg_extractor.node_dict)
    os.system("dot.exe -Tpng {} -o {}".format(out_path1,out_path2))

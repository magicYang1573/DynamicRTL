import utils.CDFG_utils as CDFG_utils
from typing import Union
import os
from op_expression import op_expression
import json
import re

# op_expression = {
#     'Cond': '({0}) ? ({1}) : ({2})',
#     'Add': '({0}) + ({1})',
#     'Lt': '({0}) < ({1})',
#     'Eq': '({0}) == ({1})',
#     'And': '({0}) && ({1})',
#     'PartSelect': '{0}[{1}:{2}]'
# }
def write_json(content, path):
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)

class Cdfg2rtler:
    def __init__(self, dot_path, module_name, original_design_path=None):
        def get_clk():
            with open(original_design_path, 'r') as f:
                content_list = f.read().split('\n')
            for line in content_list:
                if 'always' in line:
                    clk_name = line.split('(')[-1].split(')')[0].split(' ')[-1]
                    return clk_name
                
        def is_invalid_dot(dot_path):
            with open(dot_path, 'r') as f:
                content = f.read()
            if not 'digraph' in content:
                return True
        
        self.dot_path = dot_path
        
        if is_invalid_dot(dot_path):
            raise Exception('Invalid dot file')
        
        self.nx_graph = CDFG_utils.dot_to_nxgraph(self.dot_path)
        
        self.module_name = module_name
        
        self.node_list = list(set(self.nx_graph.nodes()))
        self.edge_list, self.edge_type = self.get_edge_index_type_list()
        
        self.fanin_dict, self.fanout_dict = self.get_fanin_fanout_dict()
        
        self.reg_list = self.get_nodesWithTypes_list(['Reg', 'Output_Reg'])
        self.has_clk = len(self.reg_list) > 0
        self.clk_name = get_clk() if original_design_path != None else 'clk'
        
        self.input_list = self.get_nodesWithTypes_list('Input')
        if self.has_clk and original_design_path == None:
            self.input_list.append('Input,1,clk')
        elif self.has_clk and original_design_path != None:
            self.input_list.append(f'Input,1,{self.clk_name}')
            
        self.output_list = self.get_nodesWithTypes_list(['Output', 'Output_Reg'])
        self.wire_list = self.get_nodesWithTypes_list('Wire')
        self.var_types = ['Input', 'Output', 'Reg', 'Wire', 'Output_Reg']
        
        
        self.verilog_code = ''
        
    def get_edge_index_type_list(self):
        edge_list = []
        edge_type = []
        for u, v, d in self.nx_graph.edges(data=True):
            edge_list.append([u, v])
            edge_type.append(int(''.join(filter(str.isdigit, d['label']))))
            # print(f'{(u, v)}: {d}')
        return edge_list, edge_type
        
    def get_type(self, node):
        return node.split(',')[0]
    
    def get_width(self, node):
        return node.split(',')[1]
    
    def get_name(self, node):
        return node.split(',')[2]
    
    def get_fanin_fanout_dict(self):
        """
        The order of element in fanin_dict[node] is determined by the number of edge_type
        """
            
        def sort_fanin_dict(fanin_dict, edge_index, edge_type):
            edge_with_type = []
            for i in range(len(edge_index)):
                edge_with_type.append([edge_index[i], edge_type[i]])
            
            for key in fanin_dict:
                fanin_list_with_type = [[x[0][0], x[1]] for x in edge_with_type if x[0][1] == key]
                fanin_list_with_type = sorted(fanin_list_with_type, key=lambda x: x[1])
                fanin_dict[key] = [x[0] for x in fanin_list_with_type]
                
        
        nodes = self.node_list
        edges = self.edge_list
        fanin_dict = {node: [] for node in nodes}
        fanout_dict = {node: [] for node in nodes}
        for edge in edges:
            fanin_dict[edge[1]].append(edge[0])
            fanout_dict[edge[0]].append(edge[1])
            
        sort_fanin_dict(fanin_dict, self.edge_list, self.edge_type)
        
        # edge_type_dict = {(self.edge_list[id][0], self.edge_list[id][1]): self.edge_type[id] for id in range(len(self.edge_list))}
        
        ######################
        # write_json(edge_type_dict, './my_output/edge_type_dict.json')
        ######################
        
        # for key in fanin_dict:
        #     fanin_dict[key] = sorted(fanin_dict[key], key=lambda x: edge_type_dict[(x, key)])
            
        return fanin_dict, fanout_dict
    
    def get_nodesWithTypes_list(self, type: Union[str, list]):
        """
        Returns a list of nodes with the given type
        type can be a single type or a list of types
        """
        if not isinstance(type, list):
            type = [type]
        return [node for node in self.node_list if self.get_type(node) in type]
    
    def get_module_header(self):
        def nameof(node):
            return node.split(',')[-1]
        
        input_output_list = self.input_list + self.output_list
            
        var_statement = ', '.join(map(nameof, input_output_list))
        return f'module {self.module_name}({var_statement});\n'
    
    def get_definition_var(self, node):
        var_type = self.get_type(node).lower()
        # if var_type == 'output_reg':
        #     var_width = int(self.get_width(node))
        #     var_bits_vec = f'[{var_width-1}:0]' if var_width > 1 else ''
        #     var_name = self.get_name(node)
        #     return f'reg {var_bits_vec} {var_name};\noutput {var_bits_vec} {var_name};\n'
        var_width = int(self.get_width(node))
        var_bits_vec = f'[{var_width-1}:0]' if var_width > 1 else ''
        var_name = self.get_name(node)
        
        return f'{var_type} {var_bits_vec} {var_name};\n'
    
    def get_definition_batch(self):
        output_list = []
        for node in self.output_list:
            if 'Output_Reg' in node:
                output_list.append(node.replace('Output_Reg', 'Output'))
            else:
                output_list.append(node)
                
        reg_list = []
        for node in self.reg_list:
            if 'Output_Reg' in node:
                reg_list.append(node.replace('Output_Reg', 'Reg'))
            else:
                reg_list.append(node)
        def_vars = self.input_list + output_list + reg_list + self.wire_list
        
        return ''.join(map(self.get_definition_var, def_vars))
    
    # def get_always_var(self, node):
    #     node_name = node.split(',')[-1]
    #     input_node_name = self.fanin_dict[node][0].split(',')[-1]
    #     always_block = f'always @(posedge {self.clk_name})\n    {node_name} <= {input_node_name};\n'
    #     return always_block
    
    def get_always_var(self, node):
        node_name = node.split(',')[-1]
        expression = self.get_varinput_expression(node)
        always_block = f'always @(posedge {self.clk_name})\n    {node_name} <= {expression};\n'
        return always_block
    
    def get_always_batch(self):
        return ''.join(map(self.get_always_var, self.reg_list))
    
    def get_assign_var(self, node):
        expression = self.get_varinput_expression(node)
        return f'assign {self.get_name(node)} = {expression};\n'
        
    def get_assign_batch(self):
        def find_output_reg(nodes):
            for i in range(len(nodes)):
                if self.get_type(nodes[i]) == 'Output_Reg':
                    return i
            return -1
        def erase_output_reg(nodes):
            while 1:
                output_reg_id = find_output_reg(nodes)
                if output_reg_id == -1:
                    break
                nodes.pop(output_reg_id)
            
        nodes = self.wire_list + self.output_list
        
        erase_output_reg(nodes)
        
        return ''.join(map(self.get_assign_var, nodes))
        
    def get_varinput_expression(self, node):
        input_node = self.fanin_dict[node][0]
        return self.get_expression(input_node)
    
    # def get_expression(self, node):
        
    #     if self.get_type(node) in self.var_types:
    #         return self.get_name(node)
        
    #     if self.get_type(node) == 'Const':
    #         return self.get_name(node).split('_')[-1].lower()
        
    #     input_nodes = self.fanin_dict[node]
        
    #     if self.get_type(node) in op_expression:
    #         expression_format = op_expression[self.get_type(node)]
    #     else:
    #         expression_format = f'(not known op type: {self.get_type(node)}, its vars: '
    #         for i in range(len(input_nodes)):
    #             expression_format += '{},'
    #         expression_format += ')'
            
        
    #     return expression_format.format(*map(self.get_expression, input_nodes))
    
    def get_expression_format(self, node):
        
        expression_format = op_expression[self.get_type(node)]
        if not expression_format == 'TODO':
            return expression_format
        else:
            if self.get_type(node) == 'Concat':
                return self.get_Concat_expression_format(node)
            elif self.get_type(node) == 'Case':
                return self.get_Case_expression_format(node)
            else:
                raise Exception(f'Unknown node type: {self.get_type(node)}')
    
    def get_Concat_expression_format(self, node):
        len_input_nodes = len(self.fanin_dict[node])
        var_list = list(range(len_input_nodes))
        var_list = ['{' + str(i) + '}' for i in list(range(len_input_nodes))]
        var_statement = ', '.join(var_list)
        return '{{' + var_statement + '}}'
    
    def get_Case_expression_format(self, node):
        len_input_nodes = len(self.fanin_dict[node])
        sel_id = len_input_nodes - 3
        expression = f'({{{sel_id}}} ? {{{sel_id + 1}}} : {{{sel_id + 2}}})'
        sel_id -= 2
        while sel_id >= 0:
            expression = f'({{{sel_id}}} ? {{{sel_id + 1}}} : {expression})'
            sel_id -= 2
        return expression
    
    def get_expression(self, node):
        
        if self.get_type(node) in self.var_types:
            return self.get_name(node)
        
        if self.get_type(node) == 'Const':
            return self.get_name(node).split('_')[-1].lower()
        
        input_nodes = self.fanin_dict[node]
        
        if self.get_type(node) in op_expression:
            # expression_format = op_expression[self.get_type(node)]
            expression_format = self.get_expression_format(node)
        else:
            expression_format = f'(not known op type: {self.get_type(node)}, its vars: '
            for i in range(len(input_nodes)):
                expression_format += '{},'
            expression_format += ')'
            
        
        return expression_format.format(*map(self.get_expression, input_nodes))
    
    def get_module_end(self):
        return 'endmodule'
    
    def get_rtl_content(self):
        rtl_content = ''
        rtl_content += self.get_module_header()
        rtl_content += self.get_definition_batch()
        rtl_content += self.get_always_batch()
        rtl_content += self.get_assign_batch()
        rtl_content += self.get_module_end()
        
        ############### tackle plus1 bug
        rtl_content = rtl_content.replace(':plus', '+:1')
        ###############
        
        rtl_content = self.remove_Parentheses_for_syntax_error(rtl_content)
        
        return rtl_content
    
    def remove_Parentheses_for_syntax_error(self, rtl_content):
        pattern = r'\(([a-zA-Z0-9_]+(?:\[[0-9]+\:[0-9]+\])?)\)'
        rtl_content = re.sub(pattern, r"\1", rtl_content)
        
        pattern = r'\(([0-9]+\'[hdboHDBO][0-9a-fA-FxX]+)\)'
        rtl_content = re.sub(pattern, r"\1", rtl_content)
        
        pattern = r'\((\{[^\(\)]*\})\)'
        rtl_content = re.sub(pattern, r"\1", rtl_content)
        
        return rtl_content
    
    def write_rtl(self, output_path):
        with open(output_path, 'w') as f:
            f.write(self.get_rtl_content())


if __name__ == '__main__':
    dot_path = 'dataset/rawdesign/504/cdfg/module_soc_system_led_pio_504_ast_clean_cdfg.dot'
    module_name = 'soc_system_led_pio'
    cdfg2rtler = Cdfg2rtler(dot_path, module_name)
    # cdfg2rtler.gen_fanin_fanout_dict()
    # print(cdfg2rtler.fanin_dict)
    # print(cdfg2rtler.reg_list)
    # print(cdfg2rtler.get_nodesWithTypes_list(['Input', 'Output']))
    # print(len(cdfg2rtler.edge_list))
    # print(len(cdfg2rtler.edge_type))
    # print(cdfg2rtler.get_module_header() + cdfg2rtler.get_definition_batch() + cdfg2rtler.get_always_batch() + cdfg2rtler.get_assign_bath())
    print(cdfg2rtler.get_rtl_content())
    regen_rtl_path = os.path.join(os.path.dirname(os.path.dirname(dot_path)), f'module_{module_name}_genFromDot.v')
    cdfg2rtler.write_rtl(regen_rtl_path)

# if __name__ == '__main__':
#     dot_path = 'dataset-VerilogEval/Prob057/cdfg/Prob057_kmap2_ref_Prob057_ast_clean_cdfg.dot'
#     module_name = 'TopModule'
#     cdfg2rtler = Cdfg2rtler(dot_path, module_name)
#     # cdfg2rtler.gen_fanin_fanout_dict()
#     # print(cdfg2rtler.fanin_dict)
#     # print(cdfg2rtler.reg_list)
#     # print(cdfg2rtler.get_nodesWithTypes_list(['Input', 'Output']))
#     # print(len(cdfg2rtler.edge_list))
#     # print(len(cdfg2rtler.edge_type))
#     # print(cdfg2rtler.get_module_header() + cdfg2rtler.get_definition_batch() + cdfg2rtler.get_always_batch() + cdfg2rtler.get_assign_bath())
#     print(cdfg2rtler.get_rtl_content())
#     regen_rtl_path = '/home/user/projects/CDFG2RTL/dataset-VerilogEval/Prob057/Prob057_genFromCDFG.v'
#     cdfg2rtler.write_rtl(regen_rtl_path)

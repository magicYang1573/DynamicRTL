'''
Utils for read CDFG in .dot format and transfer it into networkx graph and numpy format
Author: Yunhao Zhou
'''

import networkx as nx
import pydot
import matplotlib.pyplot as plt
import os
from queue import Queue
from glob import glob
import numpy as np
import json
import re

def dot_to_nxgraph(dot_file):
    # read .dot file
    (graph,) = pydot.graph_from_dot_file(dot_file)
    # transform pydot into networkx graph
    nx_graph = nx.nx_pydot.from_pydot(graph)
    return nx_graph

def draw_graph(graph: nx.graph, output_file):
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')  # set layout
    edge_labels = {(u, v): d['label'] for u, v, d in graph.edges(data=True) if 'label' in d}
    
    nx.draw(graph, pos, with_labels=True, node_size=70, node_color="skyblue", font_size=2, font_color="black", edge_color="gray")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=3)
    
    plt.savefig(output_file, dpi=500)
    plt.close()

def find_fanin_is_zero(nodes: list[str], edges: list[tuple], remove = False) -> list[str]:
    fanin_zero_nodes = []
    for node in nodes:
        fanin_is_zero = True
        for edge in edges:
            if edge[1] == node:
                fanin_is_zero = False
                break
        if fanin_is_zero:
            fanin_zero_nodes.append(node)

    # remove fanin_zero_node
    if remove:
        for fanin_zero_node in fanin_zero_nodes:
            # remove relevant nodes
            nodes.remove(fanin_zero_node)
            # collect relevant edges
            edges_to_remove = [edge for edge in edges if edge[0] == fanin_zero_node]
            # remove edges
            for edge_to_remove in edges_to_remove:
                edges.remove(edge_to_remove)
            
    return fanin_zero_nodes

# can't process graph with cycle
def get_level(nx_graph, remove = False):
    # get the level of each node with bf topology sort
    level_list = []
    nodes = list(set(nx_graph.nodes()))
    edges = list(set(nx_graph.edges()))
    while len(nodes) > 0:
        level_list.append(find_fanin_is_zero(nodes, edges,remove=remove))
    return level_list

def get_edge_dict(nx_graph):
    nodes = list(set(nx_graph.nodes()))
    edges = list(set(nx_graph.edges()))
    edge_dict = {node: [] for node in nodes}
    for edge in edges:
        edge_dict[edge[0]].append(edge[1])
    return edge_dict

def get_input_nodes(nodes: list[str]):
    return [node for node in nodes if node.split(',')[0] == 'Input']

def get_constant_nodes(nodes: list[str]):
    return [node for node in nodes if node.split(',')[0] == 'Const']

def get_name2idx(nx_graph):
    nodes = list(set(nx_graph.nodes()))
    edges = list(set(nx_graph.edges()))
    # treat Constant nodes as input nodes
    input_nodes = sorted(get_input_nodes(nodes), key=lambda x: x.split(',')[-1])
    constant_nodes = sorted(get_constant_nodes(nodes), key=lambda x: x.split(',')[-1])
    input_nodes = input_nodes + constant_nodes
    edge_dict = get_edge_dict(nx_graph)
    vis = {node: False for node in nodes}
    name2idx = {}
    #bfs
    bfs_q = Queue()
    for input_node in input_nodes:
        bfs_q.put(input_node)
        name2idx[input_node] = len(name2idx)
        vis[input_node] = True
    while not bfs_q.empty():
        len_bfs_q = bfs_q.qsize()
        for _ in range(len_bfs_q):
            node = bfs_q.get()
            for fanout in edge_dict[node]:
                if not vis[fanout]:
                    bfs_q.put(fanout)
                    name2idx[fanout] = len(name2idx)
                    vis[fanout] = True
    return name2idx

def is_valid(nodes, name2idx):
    for idx, node in enumerate(nodes):
        if idx != name2idx[node]:
            return False
    return True

def get_node_type(name, op_to_index):
    keys = op_to_index.keys()
    type_of_name = name.split(',')[0]
    op_index = -1
    for key in keys:
        if key == type_of_name:
            op_index = op_to_index[key]
    if op_index == -1:
        print(f'[error] unknown node type: {type_of_name}')
        assert(False)
    return op_index

def get_node_width(name):
    splits = name.split(',')
    return int(splits[1]) if not splits[1] == 'Null' else 0

# This func should only be called after getting name2idx and sorting nodes
def gen_node_feature(nodes, op_to_index):
    x_data = []
    for idx, node in enumerate(nodes):
        x_data.append([idx, get_node_type(node, op_to_index), get_node_width(node)])
    return x_data

def build_graph(nx_graph, name2idx, op_to_index):
    x_data = []
    edge_index = []
    edge_type = []
    nodes = list(set(nx_graph.nodes()))
    nodes = sorted(nodes, key=lambda x: name2idx[x])

    assert is_valid(nodes, name2idx)

    x_data = gen_node_feature(nodes, op_to_index)
    for u, v, d in nx_graph.edges(data=True):
        edge_index.append([name2idx[u], name2idx[v]])
        edge_type.append(int(''.join(filter(str.isdigit, d['label']))))
        # print(f'{(u, v)}: {d}')
    return x_data, edge_index, edge_type

def fanin_list_is_valid(x_data, edge_index, edge_type_dict, fanin_list):
    for idx, fanins in enumerate(fanin_list):
        if len(fanins) > 0:
            temp = edge_type_dict[(fanins[0], idx)]
            for fanin in fanins:
                if temp <= edge_type_dict[(fanin, idx)]:
                    temp = edge_type_dict[(fanin, idx)]
                else:
                    return False
    return True

def get_fanin_fanout(x_data, edge_index, edge_type):
    """
    The fanin_list, fanout_list type: list[list[int]]
    The fanin list of node idx, is fanin_list[idx]
    The order of element in fanin_list[idx] is determined by the number of edge_type

    example:
    edge_index = [[1, 20], [1, 11], [1, 31]]
    edge_type = [2, 3, 1]
    fanin_list[1] = [31, 20, 11]
    """
    fanin_list = [[] for _ in range(len(x_data))]
    fanout_list = [[] for _ in range(len(x_data))]

    # collect elements in fanin_list and fanout_list
    for edge in edge_index:
        fanin_list[edge[1]].append(edge[0])
        fanout_list[edge[0]].append(edge[1])
    
    edge_type_dict = {(edge_index[id][0], edge_index[id][1]): edge_type[id] for id in range(len(edge_index))}

    # sort elements in fanin_list to obey the order of edge_type
    for i in range(len(fanin_list)):
        fanin_list[i] = sorted(fanin_list[i], key=lambda x: edge_type_dict[(x, i)])
    
    assert fanin_list_is_valid(x_data, edge_index, edge_type_dict, fanin_list)

    return fanin_list, fanout_list

def check_node_name(names):
    def report_error(name):
        print(f'[error] invalid node name: {name}')
        assert(False)
        
    for name in names:
        splits = name.split(',')
        if not len(splits) == 3:
            report_error(name)

def parse_dot(dot_file, op_to_index = {}, MAX_LENGTH = -1):
    nx_graph = dot_to_nxgraph(dot_file)
    if(MAX_LENGTH > 0 and len(nx_graph.nodes()) > MAX_LENGTH):
        return [], [], [], [], []
    check_node_name(list(nx_graph.nodes()))
    # set idx of node based on bfs order
    name2idx = get_name2idx(nx_graph)
    # build graph after getting name2idx
    x_data, edge_index, edge_type = build_graph(nx_graph, name2idx, op_to_index)
    fanin_list, fanout_list = get_fanin_fanout(x_data, edge_index, edge_type)
    return x_data, edge_index, edge_type, name2idx

    # can't process graph with cycle
    # level_list = get_level(nx_graph)
    
def conststr2hexstr(conststr):
    if not "\'" in conststr:
        constval = int(conststr)
    else:
        system = conststr[conststr.find("\'") + 1].lower()
        value_part = conststr[conststr.find("\'") + 2:].lower().replace("x", "0")
        if system == 'h':
            constval = int(value_part, 16)
            pass
        elif system == 'b':
            constval = int(value_part, 2)
            pass
        elif system == 'o':
            constval = int(value_part, 8)
            pass
        elif system == 'd':
            constval = int(value_part)
        else:
            print(f'[error] unknown system: {system}, value: {conststr}')
            assert(False)
    return str(hex(constval))

def get_simulation_results(simRes_dir, name2idx, cycles = 10, traces = 6):
    def check_conststr(conststr):
        pattern = r'[^0-9a-fhox\']'
        if re.search(pattern, conststr):
            return False
        else:
            return True
        
    trace_list = sorted(glob(os.path.join(simRes_dir, '*')), key=lambda x: int(os.path.basename(x).split('-')[-1]))[:traces]
    sim_res = []
    
    for trace in trace_list:
        trace_res = [[] for _ in range(len(name2idx))]
        # get results in simulation output
        with open(trace, 'r') as f:
            trace_content_list = f.read().splitlines()[:cycles]
        for cycle_content in trace_content_list:
            cycle_content_list = cycle_content.split(':')[-1].split(',')  # remove time
            for statements in cycle_content_list:
                variable = statements.split('=')[0].strip()
                value = statements.split('=')[1].strip()
                
                # considering that some input is not used in the module, so it is not in CDFG, but in simulation results
                if variable in name2idx:
                    trace_res[name2idx[variable]].append(value)
        # add results for const
        for name in name2idx.keys():
            if 'Constant' in name:
                if not check_conststr(name.split('_')[-1].lower()):
                    print(f"[WARNING] Invalid char exits in simulation result, The string is: {name.split('_')[-1]}")
                    return None
                const_value = conststr2hexstr(name.split('_')[-1])
                trace_res[name2idx[name]] = [const_value for _ in range(len(trace_content_list))]
        sim_res.append(trace_res)
        
    return sim_res

def remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index):

    def find_redundant_node():
        """
        Can edit this function determine the rule of finding redundant nodes
        Current rule is just find redundant nodes with exactly one fanin
        However, this rule will also remove ops with only one fanin, like unot
        """
        fanin_list = [[] for _ in range(len(x_data))]
        # collect elements in fanin_list and fanout_list
        for edge in edge_index:
            fanin_list[edge[1]].append(edge[0])
        
        for idx, fanins in enumerate(fanin_list):
            if x_data[idx][1] == op_to_index['Wire']:
                redundant_node = idx
                
                assert(len(fanin_list[idx]) == 1)
                
                fanin_of_redundant_node = fanin_list[idx][0]
                return redundant_node, fanin_of_redundant_node

        return None, None

    def merge_node(rm_node, exist_node):
        """
        The rm_node will be removed, and the exist_node will inherit the value and fanout of the rm_node.
        """
        # transfer sim_res
        for i in range(len(sim_res)):
            sim_res[i][exist_node] = sim_res[i][rm_node]
            
        # transfor width of x_data
        x_data[exist_node][2] = x_data[rm_node][2]

        # remove rm_node
        ## remove from x_data
        x_data.pop(rm_node)
        
        ## remove edges
        i = 0
        while i < len(edge_index):
            if edge_index[i][0] == rm_node:
                fanout = edge_index[i][1]  # get fanout
                edge_index.pop(i)  # remove rm_node
                edge_index.append([exist_node, fanout])  # ransfer fanout
                edge_type.append(edge_type.pop(i))
            elif edge_index[i][1] == rm_node:
                edge_index.pop(i)  # remove rm_node
                edge_type.pop(i)
            else:
                i += 1
        
        ## remove from name2idx
        for name in name2idx.keys():
            if name2idx[name] == rm_node:
                name2idx.pop(name)  # zhou comment
                removed_node_name = name  # zhou add for testing
                break
            
        ## remove from sim_res
        for i in range(len(sim_res)):
            sim_res[i].pop(rm_node)
            
        #update
        ## update idx of x_data
        for i in range(len(x_data)):
            if x_data[i][0] > rm_node:
                x_data[i][0] -= 1
            
        ## update edge_index and edge_type
        for i in range(len(edge_index)):
            if edge_index[i][0] > rm_node:
                edge_index[i][0] -= 1
            if edge_index[i][1] > rm_node:
                edge_index[i][1] -= 1
                
        ## update name2idx
        for name in name2idx.keys():
            if name2idx[name] > rm_node:
                name2idx[name] -= 1
                
        return removed_node_name  # zhou add for testing
                
    num_before_merge = len(x_data)
    num_removed = 0
    
    removed_nodes_list = []  # zhou add for testing
    
    while 1:
        redundant_node, fanin_of_redundant_node = find_redundant_node()
        if redundant_node is None:
            break
        # merge_node(redundant_node, fanin_of_redundant_node)  # zhou comment
        removed_nodes_list.append(merge_node(redundant_node, fanin_of_redundant_node))  # zhou add for testing
        num_removed += 1

    # print(f'number of nodes before marge: {num_before_merge}\n number of nodes after merge: {len(x_data)}\n number of nodes removed: {num_removed}')
    # print('removed nodes:\n')
    # for item in removed_nodes_list:
    #     print(item)

def write_json(content, file_name, output_dir = './my_output'):
    output_path = os.path.join(output_dir, f'{file_name}.json')
    with open(output_path, 'w') as f:
        json.dump(content, f, indent=4)
        
def rebuild_nxGraph(x_data, edge_index, edge_type, name2idx):
    graph = nx.DiGraph()
    for name in name2idx:
        graph.add_node(name)
    idx2name = {v: k for k, v in name2idx.items()}
    for id, edge in enumerate(edge_index):
        graph.add_edge(idx2name[edge[0]], idx2name[edge[1]], label=edge_type[id])
    return graph

def get_op_to_index(path='operator_types/op_to_index.json'):
    with open(path, 'r') as f:
        op_to_index = json.load(f)
    return op_to_index

def get_name2idx_without_type_width(name2idx):
    return {name.split(',')[-1]: idx for name, idx in name2idx.items()}


def check_sim_res(x_data, sim_res, name2idx):
    idx2name = {v: k for k, v in name2idx.items()}
    
    trace = sim_res[0]
    cycle = len(trace[0])
    for idx, node_res in enumerate(trace):
        if not len(node_res) == cycle:
            print(f'node {idx2name[idx]} has wrong length of sim_res: {len(node_res)}')

def hex2binary(hex_str, width):

    if hex_str.startswith('0x'):
        hex_str = hex_str[2:]
    hex_str = hex_str.lower().replace('x', '0')
    try:
        int_value = int(hex_str, 16)
    except ValueError:
        print(f"this is invalid: {hex_str}")
        raise ValueError(f"Invalid hexadecimal string")
    
    max_value = (1 << width) - 1
    overflow = int_value > max_value
    binary_str = format(int_value, '0{}b'.format(width))
    
    return binary_str, overflow

def str2array(num: str) -> np.array:
    num_array = []
    for char in num:
        num_array.append(int(char))
    return np.array(num_array)

def regulate_sim_res(sim_res, cycles, num_bits = 64) -> list[list[list[np.array]]]:
    """
    Transfer sim_res from Hex string to binary array
    The num_bits regulate the max of binary value, if overflow occurs, return None
    """
    has_sim_res = [1 for _ in range(len(sim_res[0]))]
    for trace_id, trace in enumerate(sim_res):
        for idx, node_res in enumerate(trace):
            if node_res == []:
                has_sim_res[idx] = 0
                sim_res[trace_id][idx] = [str2array(hex2binary('0', num_bits)[0]) for _ in range(cycles)]
            else:
                bin_vals = []
                for value in node_res:
                    bin_val, overflow = hex2binary(value, num_bits)
                    if overflow:
                        print('[WARNING] Overflow exists, should discard this design')
                        return None, None
                    bin_vals.append(str2array(bin_val))
                sim_res[trace_id][idx] = bin_vals
    return sim_res, has_sim_res

def check_regulated_sim_res(sim_res, has_sim_res, num_traces, num_cycles, num_bits):
    assert len(sim_res) == num_traces
    for trace_id, trace in enumerate(sim_res):
        for idx, node_res in enumerate(trace):
            assert len(node_res) == num_cycles
            if has_sim_res[idx] == 0:
                for i in range(num_cycles):
                    assert node_res[i].size == num_bits
                    assert node_res[i].sum() == 0
            else:
                for i in range(num_cycles):
                    assert node_res[i].size == num_bits
                    
def check_all_data(x_data, edge_index, edge_type, sim_res, name2idx):
    num_nodes = len(x_data)
    num_edges = len(edge_index)
    assert(len(edge_type) == num_edges)
    assert(len(name2idx) == num_nodes)
    assert(len(sim_res[0]) == num_nodes)

def parse_design(CDFG_path, sim_path, op_to_index, args):
    num_cycles = args.num_cycles
    num_traces = args.num_traces
    num_bits = args.num_bits
    rm_redundant_nodes = args.rm_redundant_nodes
    
    x_data, edge_index, edge_type, name2idx = parse_dot(CDFG_path, op_to_index)
    
    name2idx = get_name2idx_without_type_width(name2idx)
    
    sim_res = get_simulation_results(sim_path, name2idx, cycles = num_cycles, traces = num_traces)
    if sim_res == None:
        return None, None, None, None, None

    if rm_redundant_nodes:
        remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index)
        
    sim_res, has_sim_res = regulate_sim_res(sim_res, cycles = num_cycles, num_bits = num_bits)
    if sim_res == None:
        return None, None, None, None, None
    
    check_regulated_sim_res(sim_res, has_sim_res, num_traces, num_cycles, num_bits)
    check_all_data(x_data, edge_index, edge_type, sim_res, name2idx)
    
    return x_data, edge_index, edge_type, sim_res, has_sim_res


# if __name__ == '__main__':
#     hexstr = input('input hexstr: ')
#     print(hex2binary(hexstr, 32))

# gen cdfg and simulation res for check
if __name__ == '__main__':

    op_to_index = get_op_to_index()
    dir = 'dataset/rawdesign'
    designs = os.listdir(dir)
    failed_list = []
    rm_redundant_nodes = True
    num_cycles = 10
    num_traces = 6
    num_bits = 64
    outputdir = './my_output'
    
    for design in designs:
        dot_path = glob(os.path.join(dir, design, 'cdfg', '*.dot'))
        if len(dot_path) == 0:
            print(f'design{design} does not have cdfg')
            continue
        dot_path = dot_path[0]
        sim_path = os.path.join(dir, design, 'traces')
        design_outputpath = os.path.join(outputdir, design)
        if not os.path.exists(design_outputpath):
            os.makedirs(design_outputpath)
        
        x_data, edge_index, edge_type, name2idx = parse_dot(dot_path, op_to_index)
        print(f'num_nodes: {len(x_data)}')

        origin_graph_path =  os.path.join(design_outputpath, f'module_origin_{design}.png')
        if not os.path.exists(origin_graph_path):
            draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), origin_graph_path)
        
        name2idx = get_name2idx_without_type_width(name2idx)

        sim_res = get_simulation_results(sim_path, name2idx, cycles = num_cycles, traces = num_traces)
        if sim_res == None:
            print(f'[WARNING] design {design} is descarded')
            continue
        
        if rm_redundant_nodes:
            remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index)
            print(f'redundant removed num_nodes: {len(x_data)}')
        
        
        without_redun_graph_path = os.path.join(design_outputpath, f'module_removeRedun_{design}.png')
        if not os.path.exists(without_redun_graph_path):
            draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), without_redun_graph_path)
        
        idx2name = {v:k for k, v in name2idx.items()}
        sim_res_name_res = []
        for trace in sim_res:
            sim_res_in_trace = {idx2name[idx]:res for idx, res in enumerate(trace)}
            sim_res_name_res.append(sim_res_in_trace)
        
        total_info = {"x_data": x_data, "edge_index": edge_index, "edge_type": edge_type, "name2idx": name2idx, "sim_res": sim_res_name_res}
        write_json(content=total_info, file_name=f'info_{design}.json', output_dir=design_outputpath)

# import traceback
# if __name__ == '__main__':

#     op_to_index = get_op_to_index()
#     dir = 'dataset/rawdesign'
#     designs = os.listdir(dir)
#     failed_list = []
#     rm_redundant_nodes = True
#     num_cycles = 10
#     num_traces = 6
#     num_bits = 64
    
#     for design in designs:
#         dot_path = glob(os.path.join(dir, design, 'cdfg', '*.dot'))
#         if len(dot_path) == 0:
#             print(f'design{design} does not have cdfg')
#             continue
#         dot_path = dot_path[0]
#         sim_path = os.path.join(dir, design, 'traces')
        
#         x_data, edge_index, edge_type, name2idx = parse_dot(dot_path, op_to_index)
#         print(f'num_nodes: {len(x_data)}')

#         origin_graph_path =  f"./my_output/graph_{dot_path.split('/')[-1].split('.')[0]}.png"
#         if not os.path.exists(origin_graph_path):
#             draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), origin_graph_path)
        
#         name2idx = get_name2idx_without_type_width(name2idx)

#         sim_res = get_simulation_results(sim_path, name2idx, cycles = num_cycles, traces = num_traces)
#         if sim_res == None:
#             print(f'[WARNING] design {design} is descarded')
#             continue
        
#         if rm_redundant_nodes:
#             remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index)
#             print(f'redundant removed num_nodes: {len(x_data)}')
        
#         sim_res, has_sim_res = regulate_sim_res(sim_res, cycles = num_cycles, num_bits = 64)
#         if sim_res == None:
#             print(f'[WARNING] design {design} is descarded')
#             continue
#         check_regulated_sim_res(sim_res, has_sim_res, num_traces, num_cycles, num_bits)
#         check_all_data(x_data, edge_index, edge_type, sim_res, name2idx)
        
#         without_redun_graph_path = f"./my_output/graph_{dot_path.split('/')[-1].split('.')[0]}_removedRedun.png"
#         if not os.path.exists(without_redun_graph_path):
#             draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), without_redun_graph_path)


#         error_message = traceback.format_exc()
#         failed_list.append([dot_path, str(error_message)])
    
#     with open('./my_output/failed_dot.json', 'w') as f:
#         json.dump(failed_list, f, indent=4)


# if __name__ == '__main__':

#     op_to_index = get_op_to_index()
#     rm_redundant_nodes = True

#     dot_path = "dataset/rawdesign/405/cdfg/module_Booth_Multiplier_1_405_ast_clean_cdfg.dot"
#     sim_path = os.path.join(dot_path.split('cdfg')[0], 'traces')
#     x_data, edge_index, edge_type, name2idx = parse_dot(dot_path, op_to_index)
#     print(f'num_nodes: {len(x_data)}')

#     origin_graph_path =  f"./my_output/graph_{dot_path.split('/')[-1].split('.')[0]}.png"
#     if not os.path.exists(origin_graph_path):
#         draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), origin_graph_path)
    
#     name2idx = get_name2idx_without_type_width(name2idx)

#     sim_res = get_simulation_results(sim_path, name2idx)
    
#     if rm_redundant_nodes:
#         remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index)
#         print(f'redundant removed num_nodes: {len(x_data)}')
#     check_sim_res(x_data, sim_res)
    
#     without_redun_graph_path = f"./my_output/graph_{dot_path.split('/')[-1].split('.')[0]}_removedRedun.png"
#     if not os.path.exists(without_redun_graph_path):
#         draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), without_redun_graph_path)

# if __name__ == '__main__':
#     rm_redundant_nodes = True
#     dot_path = '/rshome/yunhao.zhou/projects/GraphDeepRTL/my_temp_data/new_dataset_8_27/dataset_8_27/1/cdfg/module_vga_text_1_ast_clean_cdfg.dot'
#     draw_graph(dot_to_nxgraph(dot_path), f"./my_output/graph_{dot_path.split('/')[-1].split('.')[0]}.png")
#     # op_to_index = {'Input': 0, 'Constant': 1, 'Eq': 2, 'Plus': 3, 'Cond': 4, 'Other': 5}
#     op_to_index = get_op_to_index()
#     x_data, edge_index, edge_type, name2idx = parse_dot(dot_path, op_to_index)
#     name2idx = get_name2idx_without_type_width(name2idx)

#     print(f'num_nodes: {len(x_data)}')
    
#     sim_res = get_simulation_results('/rshome/yunhao.zhou/projects/GraphDeepRTL/my_temp_data/new_dataset_8_27/dataset_8_27/1/traces', name2idx)
    
#     if rm_redundant_nodes:
#         remove_redundant_nodes(x_data, edge_index, edge_type, name2idx, sim_res, op_to_index)
#         print(f'redundant removed num_nodes: {len(x_data)}')
#     draw_graph(rebuild_nxGraph(x_data, edge_index, edge_type, name2idx), './my_output/graph_fsm_rebuild.png')
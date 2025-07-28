from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_geometric.data import Data

from utils.data_utils import construct_node_feature

class OrderedData(Data):
    def __init__(self, x=None, x_width=None, edge_index=None, edge_type=None, sim_res=None, has_sim_res=None, y=None, power=None, slack=None, area=None):
        super().__init__()
        self.x = x
        self.x_width = x_width
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.sim_res = sim_res
        self.has_sim_res = has_sim_res
        
        self.power = power
        self.slack = slack
        self.area = area

        if x is not None:
            self.node_num = x.size(0)      # used for graph-level task
        else:
            self.node_num = 0
            
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return 1
        else:
            return 0

def parse_pyg_mlpgate(x, edge_index, edge_type, sim_res, has_sim_res, y, power, slack, area, num_node_types=39):
    x_torch = construct_node_feature(x, num_node_types)     # node type
    x_width = torch.tensor(x[:, 2])     # node width

    # tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    # tt_pair_index = tt_pair_index.t().contiguous()
    # rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    # rc_pair_index = rc_pair_index.t().contiguous()
    # tt_dis = torch.tensor(tt_dis)
    # is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    sim_res = torch.tensor(sim_res)
    # sim_res = sim_res.transpose(0, 1)
    has_sim_res = torch.tensor(has_sim_res)

    # x_toggle_rate = cal_node_toggle_rate(x_width, sim_res)
    
    y = torch.tensor(y, dtype=torch.long)
    
    power=torch.tensor(power)
    slack=torch.tensor(slack)
    area=torch.tensor(area)
    
    # if len(edge_index) == 0:
    #     edge_index = edge_index.t().contiguous()
    #     forward_index = torch.LongTensor([i for i in range(len(x))])
    #     backward_index = torch.LongTensor([i for i in range(len(x))])
    #     forward_level = torch.zeros(len(x))
    #     backward_level = torch.zeros(len(x))
    # else:
    #     edge_index = edge_index.t().contiguous()
    #     forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    graph = OrderedData(x=x_torch, x_width=x_width, edge_index=edge_index, edge_type=edge_type, sim_res=sim_res, has_sim_res=has_sim_res, y=y, power=power, slack=slack, area=area)
    graph.use_edge_attr = False

    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    # graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    # graph.prob = torch.tensor(y).reshape((len(x), 1))

    return graph


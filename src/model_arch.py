from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import GRU

from gdrtl_arch.encoder_decoder import *
from test_gdrtl_arch.TFMLP.tfmlp_pos_op_Encoding import TfMlpPosEncAggr
from utils.CDFG_utils import get_op_to_index

OP_ID_INPUT = 0
OP_ID_CONST = 1
OP_ID_WIRE = 2
OP_ID_REG = 3
OP_ID_OUTPUT = 38

class Model_shared(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 enable_encode = True,
                 enable_reverse = False
                ):
        super(Model_shared, self).__init__()

        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse
        self.op_to_index = get_op_to_index()
        # self.op_to_index = {k: v.cuda() for k, v in get_op_to_index().items()}

        # dimensions
        self.dim_hidden = HID_DIM * N_LAYERS
        self.dim_mlp = 32

        # number encoder decoder
        self.seq2seq_model = Seq2Seq(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DROPOUT, retrain=False)  

        # operator aggrs
        self.aggr = TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden)   # input_channel, output_channel
        
        # operator GRUs
        self.gru = GRU(self.dim_hidden, self.dim_hidden)   # input_channel, output_channel

        # predict branch hit mlp
        self.branch_hit_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        # predict assert mlp
        self.assert_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.toggle_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    '''
        input: graph (one graph has one trace)
            - x: node type (one-hot)    [N, type_num]
            - edge_index: (out-edge node, in-edge node)     [2, E] 
            - edge_type: edge order id      [E]
            - sim_res:      [N, seq_len, value_width]
                - input and constant nodes
            - has_sim_res   [N]
        
        output:
            - hit of value
            - 
                -- except input and constant nodes
    '''
    def forward(self, G, seq_len):
        device = next(self.parameters()).device
        # print(G.name)
        num_nodes = len(G.x)
        
        # initialize the embedding of nodes to zero
        node_emb = torch.zeros(num_nodes, self.dim_hidden).to(device)

        # initialize the embedding of input and const node
        input_mask = G.x[:, self.op_to_index['Input']] == 1
        node_emb[input_mask] = self.seq2seq_model.encode(G.sim_res[input_mask, :seq_len, :].float()).permute(1, 0, 2).flatten(start_dim=1)

        const_mask = G.x[:, self.op_to_index['Const']] == 1
        node_emb[const_mask] = self.seq2seq_model.encode(G.sim_res[const_mask, :seq_len, :].float()).permute(1, 0, 2).flatten(start_dim=1)
        
        # edge proeprty = target node type (operator type) + order id
        edge_op_type = torch.argmax(G.x[G.edge_index[1]], dim=1)

        # get mask of operator nodes
        # mask = torch.zeros(num_nodes).bool().to(self.device)
        # for op, op_id in self.op_to_index.items():
        #     if op not in ['Input', 'Const', 'Wire', 'Output']:
        #         op_mask = G.x[:, op_id] == 1
        #         mask |= op_mask

        # propogate
        for _ in range(self.num_rounds):
            # aggregate
            aggr_emb = self.aggr(node_emb, G.edge_index, G.edge_type, edge_op_type)
            aggr_emb = aggr_emb.unsqueeze(0)
            # update node emb with GRU
            # Input: 
            ##  input emb [seq_len=1, batch_size, emb_dim]
            ##  hidden emb [layer=1, batch_size, emb_dim]
            # Output:
            ## output, hidden [layer=1, batch_size, emb_dim]
            _ , update_emb = self.gru(aggr_emb, node_emb.unsqueeze(0))

            # node_emb[mask, :] = update_emb.squeeze(0)[mask, :]
            node_emb = update_emb.squeeze(0)

        return node_emb
    

    def pred_seq(self, node_emb, src_shape, gt_sim_res):
        node_emb = node_emb.view(node_emb.size(0), N_LAYERS, int(node_emb.size(1)/N_LAYERS)).permute(1, 0, 2)
        seq = self.seq2seq_model.decode_test(node_emb.contiguous(), src_shape, gt_sim_res)
        return seq

    def pred_branch_hit(self, node_emb):
        prob = self.branch_hit_mlp(node_emb)
        return prob

    def pred_toggle_rate(self, node_emb):
        rate = self.toggle_mlp(node_emb)
        return rate

    # assert whether a variable will be active
    def pred_assert_zero(self, node_emb):
        prob = self.assert_mlp(node_emb)
        return prob


class Model_default(nn.Module):
    '''
    partly shared weight aggregator
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 enable_encode = True,
                 enable_reverse = False
                ):
        super(Model_default, self).__init__()

        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse
        self.op_to_index = get_op_to_index()
        # self.op_to_index = {k: v.cuda() for k, v in get_op_to_index().items()}

        # dimensions
        self.dim_hidden = HID_DIM * N_LAYERS
        self.dim_mlp = 32

        # number encoder decoder
        self.seq2seq_model = Seq2Seq(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DROPOUT, retrain=False)  

        # operator aggrs
        # self.aggr = TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden)   # input_channel, output_channel
        
        # operator GRUs
        self.gru = GRU(self.dim_hidden, self.dim_hidden)   # input_channel, output_channel

        # predict branch hit mlp
        self.branch_hit_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        # predict assert mlp
        self.assert_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.toggle_mlp = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
        self.node_groups = {
            "Reg": torch.tensor([self.op_to_index['Reg']]),
            "Bop": torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),  # binary operator
            "Sop": torch.tensor([26, 27, 28, 29, 30, 31, 32]),  # single operator
            "Mop": torch.tensor([34, 36]),  # multiple operator
            "Cond": torch.tensor([self.op_to_index['Cond']]),  # condition operator
            "Output": torch.tensor([self.op_to_index['Output']]),  # output operator
        }
        
        self.aggr_dict = {
            "Reg": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),
            "Bop": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),  # binary operator
            "Sop": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),  # single operator
            "Mop": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),  # multiple operator
            "Cond": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),  # condition operator
            "Output": TfMlpPosEncAggr(in_channels=self.dim_hidden, ouput_channels=self.dim_hidden),  # output operator
        }

    '''
        input: graph (one graph has one trace)
            - x: node type (one-hot)    [N, type_num]
            - edge_index: (out-edge node, in-edge node)     [2, E] 
            - edge_type: edge order id      [E]
            - sim_res:      [N, seq_len, value_width]
                - input and constant nodes
            - has_sim_res   [N]
        
        output:
            - hit of value
            - 
                -- except input and constant nodes
    '''
    def forward(self, G, seq_len):
        device = next(self.parameters()).device
        # print(G.name)
        num_nodes = len(G.x)
        
        # initialize the embedding of nodes to zero
        node_emb = torch.zeros(num_nodes, self.dim_hidden).to(device)

        # initialize the embedding of input and const node
        input_mask = G.x[:, self.op_to_index['Input']] == 1
        node_emb[input_mask] = self.seq2seq_model.encode(G.sim_res[input_mask, :seq_len, :].float()).permute(1, 0, 2).flatten(start_dim=1)

        const_mask = G.x[:, self.op_to_index['Const']] == 1
        node_emb[const_mask] = self.seq2seq_model.encode(G.sim_res[const_mask, :seq_len, :].float()).permute(1, 0, 2).flatten(start_dim=1)
        
        # edge proeprty = target node type (operator type) + order id
        edge_op_type = torch.argmax(G.x[G.edge_index[1]], dim=1)

        # get mask of operator nodes
        # mask = torch.zeros(num_nodes).bool().to(self.device)
        # for op, op_id in self.op_to_index.items():
        #     if op not in ['Input', 'Const', 'Wire', 'Output']:
        #         op_mask = G.x[:, op_id] == 1
        #         mask |= op_mask

        # propogate
        for _ in range(self.num_rounds):
            # aggregate
            # aggr_emb = self.aggr(node_emb, G.edge_index, G.edge_type, edge_op_type)
            # aggr_emb = aggr_emb.unsqueeze(0)
            
            # aggr_emb = self.aggr(node_emb, G.edge_index, G.edge_type, edge_op_type)
            aggr_emb = torch.zeros(node_emb.shape[0], self.dim_hidden).to(next(self.parameters()).device)
            
            for group_name in self.node_groups:
                # print('group_name:', group_name)
                group_mask = torch.isin(torch.argmax(G.x, dim=1), self.node_groups[group_name])
                # print('group_nodes:', aggr_emb[group_mask].shape[0])
                # print('--------------')
                aggr_emb[group_mask] = self.aggr_dict[group_name](node_emb, G.edge_index, G.edge_type, edge_op_type)[group_mask]
                
            aggr_emb = aggr_emb.unsqueeze(0)
                
            # update node emb with GRU
            # Input: 
            ##  input emb [seq_len=1, batch_size, emb_dim]
            ##  hidden emb [layer=1, batch_size, emb_dim]
            # Output:
            ## output, hidden [layer=1, batch_size, emb_dim]
            _ , update_emb = self.gru(aggr_emb, node_emb.unsqueeze(0))

            # node_emb[mask, :] = update_emb.squeeze(0)[mask, :]
            node_emb = update_emb.squeeze(0)

        return node_emb
    

    def pred_seq(self, node_emb, src_shape, gt_sim_res):
        node_emb = node_emb.view(node_emb.size(0), N_LAYERS, int(node_emb.size(1)/N_LAYERS)).permute(1, 0, 2)
        seq = self.seq2seq_model.decode_test(node_emb.contiguous(), src_shape, gt_sim_res)
        return seq

    def pred_branch_hit(self, node_emb):
        prob = self.branch_hit_mlp(node_emb)
        return prob

    def pred_toggle_rate(self, node_emb):
        rate = self.toggle_mlp(node_emb)
        return rate

    # assert whether a variable will be active
    def pred_assert_zero(self, node_emb):
        prob = self.assert_mlp(node_emb)
        return prob
    
    
    def to_device(self):
        device = next(self.parameters()).device

        for key, tensor in self.node_groups.items():
            self.node_groups[key] = tensor.to(device)
        
        for key, aggr in self.aggr_dict.items():

            self.aggr_dict[key] = aggr.to(device)
    
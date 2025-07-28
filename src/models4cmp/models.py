from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from gdrtl_arch.encoder_decoder import *
from utils.CDFG_utils import get_op_to_index
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

OP_ID_INPUT = 0
OP_ID_CONST = 1
OP_ID_WIRE = 2
OP_ID_REG = 3
OP_ID_OUTPUT = 38

class GCN(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 enable_encode = True,
                 enable_reverse = False
                ):
        super(GCN, self).__init__()

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
        self.aggr = GCNConv(in_channels=self.dim_hidden, out_channels=self.dim_hidden)
        
        # operator GRUs
        # self.gru = GRU(self.dim_hidden, self.dim_hidden)   # input_channel, output_channel

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
        
        other_mask = ~(input_mask | const_mask)
        node_emb[other_mask, :G.x.shape[1]] = G.x[other_mask]
        # edge proeprty = target node type (operator type) + order id
        # edge_op_type = torch.argmax(G.x[G.edge_index[1]], dim=1)

        # get mask of operator nodes
        # mask = torch.zeros(num_nodes).bool().to(self.device)
        # for op, op_id in self.op_to_index.items():
        #     if op not in ['Input', 'Const', 'Wire', 'Output']:
        #         op_mask = G.x[:, op_id] == 1
        #         mask |= op_mask

        # propogate
        for _ in range(self.num_rounds):
            # aggregate
            aggr_emb = self.aggr(node_emb, G.edge_index)
            aggr_emb = aggr_emb.unsqueeze(0)
            # update node emb with GRU
            # Input: 
            ##  input emb [seq_len=1, batch_size, emb_dim]
            ##  hidden emb [layer=1, batch_size, emb_dim]
            # Output:
            ## output, hidden [layer=1, batch_size, emb_dim]

            # node_emb[mask, :] = update_emb.squeeze(0)[mask, :]
            node_emb = aggr_emb.squeeze(0)

        return node_emb
    

    def pred_seq(self, node_emb, src_shape, gt_sim_res):
        node_emb = node_emb.view(node_emb.size(0), N_LAYERS, int(node_emb.size(1)/N_LAYERS)).permute(1, 0, 2)
        seq = self.seq2seq_model.decode_test(node_emb.contiguous(), src_shape, gt_sim_res)
        return seq

    def pred_branch_hit(self, node_emb):
        prob = self.branch_hit_mlp(node_emb)
        return prob

    # assert whether a variable will be active
    def pred_assert_zero(self, node_emb):
        prob = self.assert_mlp(node_emb)
        return prob

    def pred_toggle_rate(self, node_emb):
        rate = self.toggle_mlp(node_emb)
        return rate


class GAT(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 enable_encode = True,
                 enable_reverse = False
                ):
        super(GAT, self).__init__()

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
        self.aggr = GATConv(in_channels=self.dim_hidden, out_channels=self.dim_hidden)
        
        # operator GRUs
        # self.gru = GRU(self.dim_hidden, self.dim_hidden)   # input_channel, output_channel

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
        
        other_mask = ~(input_mask | const_mask)
        node_emb[other_mask, :G.x.shape[1]] = G.x[other_mask]
        # edge proeprty = target node type (operator type) + order id
        # edge_op_type = torch.argmax(G.x[G.edge_index[1]], dim=1)

        # get mask of operator nodes
        # mask = torch.zeros(num_nodes).bool().to(self.device)
        # for op, op_id in self.op_to_index.items():
        #     if op not in ['Input', 'Const', 'Wire', 'Output']:
        #         op_mask = G.x[:, op_id] == 1
        #         mask |= op_mask

        # propogate
        for _ in range(self.num_rounds):
            # aggregate
            aggr_emb = self.aggr(node_emb, G.edge_index)
            aggr_emb = aggr_emb.unsqueeze(0)
            # update node emb with GRU
            # Input: 
            ##  input emb [seq_len=1, batch_size, emb_dim]
            ##  hidden emb [layer=1, batch_size, emb_dim]
            # Output:
            ## output, hidden [layer=1, batch_size, emb_dim]

            # node_emb[mask, :] = update_emb.squeeze(0)[mask, :]
            node_emb = aggr_emb.squeeze(0)

        return node_emb
    

    def pred_seq(self, node_emb, src_shape, gt_sim_res):
        node_emb = node_emb.view(node_emb.size(0), N_LAYERS, int(node_emb.size(1)/N_LAYERS)).permute(1, 0, 2)
        seq = self.seq2seq_model.decode_test(node_emb.contiguous(), src_shape, gt_sim_res)
        return seq

    def pred_branch_hit(self, node_emb):
        prob = self.branch_hit_mlp(node_emb)
        return prob

    # assert whether a variable will be active
    def pred_assert_zero(self, node_emb):
        prob = self.assert_mlp(node_emb)
        return prob

    def pred_toggle_rate(self, node_emb):
        rate = self.toggle_mlp(node_emb)
        return rate

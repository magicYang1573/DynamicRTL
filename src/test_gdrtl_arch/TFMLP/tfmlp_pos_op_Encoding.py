import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from typing import Optional
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor

class PositionalEncoding_learnable(torch.nn.Module):
    def __init__(self, d_model, max_len=10):
        # max_len should be larger than the maximum of edge_types
        super(PositionalEncoding_learnable, self).__init__()
        # learnable positional encoding size (num_edge_types, d_model)
        self.position_embedding = torch.nn.Embedding(max_len, d_model)

    def forward(self, edge_type):
        # get position embedding
        return self.position_embedding(edge_type)
    
class PositionalEncoding_SineCosine(torch.nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding_SineCosine, self).__init__()
        # 创建一个(max_len, d_model)大小的矩阵用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos函数
        self.register_buffer('pe', pe)
    def forward(self, edge_type):
        # 将edge_type对应到pe的行
        return self.pe[edge_type]

class OperatorEncoding_learnable(torch.nn.Module):
    def __init__(self, d_model, max_len=40):
        # max_len should be larger than the maximum of operator number
        super(OperatorEncoding_learnable, self).__init__()
        # learnable operator encoding size (num_opertor_types, d_model)
        self.op_embedding = torch.nn.Embedding(max_len, d_model)

    def forward(self, edge_type):
        # get position embedding
        return self.op_embedding(edge_type)


class TfMlpPosEncAggr(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node's neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, ouput_channels=64, reverse=False, mlp_post=None, learnable_pos_enc=False, num_edge_numbers=11, op_number=40):
        super(TfMlpPosEncAggr, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        # pos and op emb
        pos_emb_dim = 32
        op_emb_dim = 32
        if learnable_pos_enc:
            self.pos_encoder = PositionalEncoding_learnable(d_model=pos_emb_dim, max_len=num_edge_numbers)
        else:
            self.pos_encoder = PositionalEncoding_SineCosine(d_model=pos_emb_dim, max_len=num_edge_numbers)
        
        self.op_encoder = OperatorEncoding_learnable(d_model=op_emb_dim, max_len=op_number)

        # attention
        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1)

        attn_in_dim = in_channels + pos_emb_dim + op_emb_dim
        self.msg_q = nn.Linear(in_channels, ouput_channels)
        self.msg_k = nn.Linear(attn_in_dim, ouput_channels)
        self.msg_v = nn.Linear(attn_in_dim, ouput_channels)
        

    def forward(self, x, edge_index, edge_order, edge_op_type, edge_attr=None, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        edge_pos_enc = self.pos_encoder(edge_order)
        edge_op_enc = self.op_encoder(edge_op_type)

        return self.propagate(edge_index, x=x, edge_pos_enc=edge_pos_enc, edge_op_enc=edge_op_enc, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_pos_enc, edge_op_enc, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key
        # edge: [j->i]
        # x_i: emb of node i
        # x_j: emb of node j 

        # pos and op embedding by XXXX-5
        x_j = torch.cat((x_j, edge_pos_enc, edge_op_enc), dim=1)
        # x_i = torch.cat((x_i, edge_pos_enc, edge_op_enc), dim=1)
        
        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        # x_j -> value 
        t = self.msg_v(x_j) * a_j

        return t



    # def message(self, x_i, x_j, edge_pos_enc, edge_op_enc, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
    #     # h_i: query, h_j: key
    #     # edge: [j->i]
    #     # x_i: emb of node i
    #     # x_j: emb of node j 

    #     # pos and op embedding by XXXX-5
    #     x_j = torch.cat((x_j, edge_pos_enc, edge_op_enc), dim=1)
    #     x_i = torch.cat((x_i, edge_pos_enc, edge_op_enc), dim=1)
        
    #     h_attn_q_i = self.msg_q(x_i)
    #     h_attn = self.msg_k(x_j)
    #     # see comment in above self attention why this is done here and not in forward
    #     a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
    #     a_j = softmax(a_j, index, ptr, size_i)
    #     # x_j -> value 
    #     t = self.msg_v(x_j) * a_j

    #     return t
    
    # def message(self, x_i, x_j, edge_pos_enc, edge_op_enc, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
    #     # h_i: query, h_j: key
    #     # edge: [j->i]
    #     # x_i: emb of node i
    #     # x_j: message emb of node j to node i

    #     # pos and op embedding by XXXX-5
    #     x_j = torch.cat((x_j, edge_pos_enc, edge_op_enc), dim=1)
    #     # x_i = torch.cat((x_i, torch.zeros_like(torch.cat((edge_pos_enc, edge_op_enc), dim=1))), dim=1)
    #     return x_j

    #     # h_attn_q_j = self.msg_q(x_j)
    #     # h_attn = self.msg_k(x_j)
    #     # # see comment in above self attention why this is done here and not in forward
    #     # a_j = self.attn_lin(torch.cat([h_attn_q_j, h_attn], dim=-1))
    #     # a_j = softmax(a_j, index, ptr, size_i)
    #     # # x_j -> value 
    #     # t = self.msg_v(x_j) * a_j
    #     # return t
    
    # def aggregate(self, x_j, index, x):
    #     # index: the connected node_id of x_j
    #     q = self.msg_q(x_j)
    #     k = self.msg_k(x_j)
    #     v = self.msg_v(x_j)

    #     edge_mask = index.unsqueeze(0) == index.unsqueeze(1)
    #     # print(edge_mask.size())
    #     attn_weight = torch.bmm(q.unsqueeze(0), k.transpose(0,1).unsqueeze(0))
    #     # print(attn_weight.size())
    #     attn_weight = attn_weight.squeeze(0) * edge_mask
    #     scale_factor = edge_mask.sum(dim=1) ** 0.5
    #     scaled_weight = attn_weight / scale_factor
    #     attn_weight = self.attn_softmax(scaled_weight)
    #     attn_output = torch.bmm(attn_weight.unsqueeze(0), v.unsqueeze(0))

    #     # aggr_emb = x
    #     # aggr_emb[index, :] = 0
    #     aggr_emb = torch.zeros_like(x)
    #     aggr_emb[index, :] += attn_output.squeeze(0)

    #     return aggr_emb

    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out
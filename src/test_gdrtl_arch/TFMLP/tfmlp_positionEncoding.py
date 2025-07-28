import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from typing import Optional
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor

class PositionalEncoding_learnable(torch.nn.Module):
    def __init__(self, d_model, max_len=100, device='cpu'):
        # max_len should be larger than the maximum of edge_types
        super(PositionalEncoding_learnable, self).__init__()
        # learnable positional encoding size (num_edge_types, d_model)
        self.position_embedding = torch.nn.Embedding(max_len, d_model)

    def forward(self, edge_type):
        # get position embedding
        return self.position_embedding(edge_type)
    
class PositionalEncoding_SineCosine(torch.nn.Module):
    def __init__(self, d_model, max_len=100, device='cpu'):
        super(PositionalEncoding_SineCosine, self).__init__()
        # 创建一个(max_len, d_model)大小的矩阵用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe = torch.zeros(max_len, d_model).to(device)
        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin函数
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos函数

    def forward(self, edge_type):
        # 将edge_type对应到pe的行
        return self.pe[edge_type]

class TfMlpPosEncAggr(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node's neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, ouput_channels=64, reverse=False, mlp_post=None, learnable_pos_enc=False, num_edge_types=100, device='cpu'):
        super(TfMlpPosEncAggr, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        
        self.device = device

        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1).to(self.device)

        self.msg_q = nn.Linear(in_channels, ouput_channels).to(self.device)
        self.msg_k = nn.Linear(in_channels, ouput_channels).to(self.device)
        self.msg_v = nn.Linear(in_channels, ouput_channels).to(self.device)
        
        
        if learnable_pos_enc:
            self.pos_encoder = PositionalEncoding_learnable(d_model=in_channels, max_len=num_edge_types, device=self.device)
        else:
            self.pos_encoder = PositionalEncoding_SineCosine(d_model=in_channels, max_len=num_edge_types, device=self.device)
        


    def forward(self, x, edge_index, edge_type, edge_attr=None, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_pos_enc = self.pos_encoder(edge_type)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_pos_enc=edge_pos_enc)

    def message(self, x_i, x_j, edge_pos_enc, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key 
        
        # positional encoding by XXXX-1
        x_j = x_j + edge_pos_enc

        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        # x_j -> value 
        t = self.msg_v(x_j) * a_j
        return t
    
    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out
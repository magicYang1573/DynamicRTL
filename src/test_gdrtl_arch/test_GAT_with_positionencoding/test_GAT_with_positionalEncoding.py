import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class PositionalEncoding_learnable(torch.nn.Module):
    def __init__(self, d_model, max_len=100):
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
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin函数
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos函数

    def forward(self, edge_type):
        # 将edge_type对应到pe的行
        return self.pe[edge_type]

class GATWithPositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, num_edge_types=10, learnable_pos_enc=False):
        super(GATWithPositionalEncoding, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
        
        if learnable_pos_enc:
            self.pos_encoder = PositionalEncoding_learnable(d_model=in_channels, max_len=num_edge_types)
        else:
            self.pos_encoder = PositionalEncoding_SineCosine(d_model=in_channels, max_len=num_edge_types)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        # get position embedding
        position_embedding = self.pos_encoder(edge_type)
        
        # 对于每条边，将position embedding加到对应的输入节点特征上
        x_with_pos = x.clone()
        for i in range(edge_index.shape[1]):
            source_node = edge_index[0, i]  # 边的起始节点
            x_with_pos[source_node] += position_embedding[i]  # 将对应的position_embedding加到起始节点的特征上
        
        # 执行带位置编码的GAT卷积
        x = self.gat_conv(x_with_pos, edge_index)
        return F.elu(x)


x = torch.rand((4, 16))  
edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)  # 边索引
edge_type = torch.tensor([0, 1, 2, 1, 3])  

data = Data(x=x, edge_index=edge_index, edge_type=edge_type)


model = GATWithPositionalEncoding(in_channels=16, out_channels=8, heads=2, num_edge_types=5)
output = model(data)
print(output.shape)

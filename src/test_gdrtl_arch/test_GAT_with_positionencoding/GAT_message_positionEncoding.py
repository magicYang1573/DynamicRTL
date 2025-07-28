import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
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

class GATWithEdgeFeatures(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, num_edge_types=10, learnable_pos_enc=False):
        super(GATWithEdgeFeatures, self).__init__(aggr='add')  # "add" 聚合消息
        self.heads = heads
        self.lin = torch.nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.edge_encoder = torch.nn.Embedding(num_edge_types, out_channels * heads)
        torch.nn.init.xavier_uniform_(self.att)
        if learnable_pos_enc:
            self.pos_encoder = PositionalEncoding_learnable(d_model=in_channels, max_len=num_edge_types)
        else:
            self.pos_encoder = PositionalEncoding_SineCosine(d_model=in_channels, max_len=num_edge_types)
    
    def forward(self, x, edge_index, edge_type):
        # 线性变换输入特征
        x = self.lin(x).view(-1, self.heads, self.lin.out_features // self.heads)
        
        # 获取edge feature
        edge_feature = self.edge_encoder(edge_type)

        # 开始消息传递，调用内部的message和aggregate函数
        return self.propagate(edge_index, x=x, edge_feature=edge_feature)

    def message(self, x_j, edge_feature):
        # x_j 是邻居节点的特征，edge_feature 是边的特征
        # 计算注意力权重：节点特征和边特征相结合
        attention_scores = (x_j + edge_feature).sum(dim=-1, keepdim=True)
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights * (x_j + edge_feature)

    def aggregate(self, inputs, index):
        # 按节点聚合消息
        return torch.scatter_add(inputs, 0, index, dim_size=index.max().item() + 1)
    
    def update(self, aggr_out):
        # 对聚合后的特征做非线性激活
        return F.elu(aggr_out)

# 创建示例数据
x = torch.rand((4, 16))  # 4个节点，每个节点16维特征
edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)  # 边索引
edge_type = torch.tensor([0, 1, 2, 1, 3])  # 边类型

data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

# 定义模型
model = GATWithEdgeFeatures(in_channels=16, out_channels=8, heads=2, num_edge_types=5)

# 前向传播
output = model(data.x, data.edge_index, data.edge_type)
print(output.shape)

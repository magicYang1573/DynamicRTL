import torch
from torch import nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import degree

from gdrtl_arch.encoder_decoder import *

class DownstreamModel(nn.Module):
    '''
    Downstream Task Models
    '''
    def __init__(self, gnn_round):
        super(DownstreamModel, self).__init__()

        self.dim_hidden = HID_DIM * N_LAYERS

        # downstream task for specific assertion (1 variable)
        self.assert_mlp_1 = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        
        # downstream task for specific assertion (2 variable)
        self.assert_mlp_2 = nn.Sequential(
            nn.Linear(self.dim_hidden*2, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()        
        )

        # downstream task for power prediction
        # alignment layer
        node_structure_dim = 41     # node_width (1), type (39), neighbor count (1)
        self.reshape_linear_1 = nn.Linear(node_structure_dim, self.dim_hidden)
        self.reshape_linear_2 = nn.Linear(node_structure_dim + self.dim_hidden, self.dim_hidden)

        # power prediction GNN layer
        num_layers = gnn_round
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(self.dim_hidden, self.dim_hidden))
        # linear layer
        self.fc = nn.Sequential(
            nn.Linear(self.dim_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def predict_downstream_value_assert(self, node_emb):
        prob = self.assert_mlp_1(node_emb)
        return prob

    def predict_downstream_cmp_assert(self, node_emb_1, node_emb_2):
        prob = self.assert_mlp_2(torch.cat([node_emb_1, node_emb_2], dim=1))
        return prob

    # no dynamic: use 'how good' method
    # - node feature: node width, node type, number of neighboring nodes
    # dynamic: 'how good' method + learned dynamic embedding
    def predict_downstream_power(self, batch, node_emb, power_model):
        edge_index = batch.edge_index
        node_width = batch.x_width
        node_type = batch.x

        node_numbers = batch.node_num

        neighbor_counts = degree(edge_index[0], node_width.size(0), dtype=torch.float)
        neighbor_counts = neighbor_counts.view(-1, 1)
        node_width = node_width.view(-1, 1)
        node_structure_features = torch.cat((node_width, node_type, neighbor_counts), dim=1)  # [node_number, (1+39+1)]

        if power_model=='no_dynamic': 
            node_emb = self.reshape_linear_1(node_structure_features)
        elif power_model=='dynamic':
            node_emb = self.reshape_linear_2(torch.cat((node_structure_features, node_emb), dim=1))

        for conv in self.convs:
            node_emb = conv(node_emb, edge_index)
            node_emb = torch.relu(node_emb)
        
        # sum pooling according to each graph, then use linear layer to get the value
        start_idx = 0
        aggregated_features = torch.zeros(len(node_numbers), node_emb.size(1), device=node_emb.device)
        for i, num_nodes in enumerate(node_numbers):
            graph_node_emb = node_emb[start_idx:start_idx + num_nodes]
            graph_aggregated_features = torch.sum(graph_node_emb, dim=0, keepdim=True)
            aggregated_features[i] = graph_aggregated_features
            start_idx += num_nodes

        pred_power = self.fc(aggregated_features).squeeze(1)


        # # use linear layer to get the power value of each node, then sum
        # node_power = self.fc(node_emb).squeeze(1)

        # pred_power = torch.zeros(len(node_numbers), device=node_emb.device)
        # start_idx = 0
        # for i, num_nodes in enumerate(node_numbers):
        #     graph_power = node_power[start_idx:start_idx + num_nodes]
        #     pred_power[i] = torch.sum(graph_power, dim=0, keepdim=True)
        #     start_idx += num_nodes

        return pred_power



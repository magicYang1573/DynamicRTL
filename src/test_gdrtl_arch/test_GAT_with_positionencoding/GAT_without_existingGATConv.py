import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, concat=True, dropout=0.6, alpha=0.2):
        super(CustomGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features // num_heads
        self.num_heads = num_heads
        self.concat = concat  # True if we concatenate the heads, False if we average them

        # Linear transformation for each head
        self.linear = nn.Linear(in_features, self.out_features * num_heads, bias=False)
        # Attention mechanism
        self.attention = nn.Parameter(torch.zeros(size=(num_heads, 2 * self.out_features)))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)
        
        # LeakyReLU for attention coefficients
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Dropout for attention scores
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: [N, in_features], the input node features.
        adj: [N, N], adjacency matrix indicating edges between nodes.
        """
        N = h.size(0)

        # 1. Linear transformation (for each head)
        h_prime = self.linear(h)  # Shape: [N, out_features * num_heads]
        h_prime = h_prime.view(N, self.num_heads, self.out_features)  # Shape: [N, num_heads, out_features]

        # 2. Compute attention coefficients for each edge (i,j)
        # Prepare attention scores
        h_i = h_prime.unsqueeze(1).repeat(1, N, 1, 1)  # Shape: [N, N, num_heads, out_features]
        h_j = h_prime.unsqueeze(0).repeat(N, 1, 1, 1)  # Shape: [N, N, num_heads, out_features]

        attention_input = torch.cat([h_i, h_j], dim=-1)  # Shape: [N, N, num_heads, 2 * out_features]
        attention_input = attention_input.permute(2, 0, 1, 3)  # Shape: [num_heads, N, N, 2 * out_features]

        # Attention score for each head
        e = self.leakyrelu(torch.einsum('hij,kj->hij', attention_input, self.attention))  # Shape: [num_heads, N, N]

        # Mask non-edges by setting attention scores to -infinity (adj == 0)
        attention = torch.where(adj > 0, e, torch.full_like(e, -float('inf')))

        # Softmax to get normalized attention coefficients
        attention = F.softmax(attention, dim=2)  # Shape: [num_heads, N, N]
        attention = self.dropout(attention)

        # 3. Apply attention to node features (message passing)
        h_prime = torch.einsum('hij,hjk->hik', attention, h_prime)  # Shape: [num_heads, N, out_features]

        # 4. Concatenate or average over the heads
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).reshape(N, -1)  # Concatenate heads (Shape: [N, num_heads * out_features])
        else:
            h_prime = h_prime.mean(dim=0)  # Average heads (Shape: [N, out_features])

        return F.elu(h_prime)

# Example usage:
N = 4  # Number of nodes
in_features = 5  # Input feature dimension
out_features = 8  # Output feature dimension

# Node feature matrix (random for demonstration)
h = torch.rand(N, in_features)

# Adjacency matrix (example)
adj = torch.Tensor([[1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [0, 1, 1, 1],
                    [0, 0, 1, 1]])

# Define and apply the custom GAT layer
gat_layer = CustomGATLayer(in_features, out_features, num_heads=2, concat=True)
output = gat_layer(h, adj)
print(output)

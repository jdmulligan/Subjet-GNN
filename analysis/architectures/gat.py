'''
Graph Attention Network using PyTorch via PyTorch Geometric
See: https://pytorch-geometric.readthedocs.io/en/latest

This class defines the PyTorch architecture: __init__(), forward()
'''

import torch
import torch_geometric

##################################################################    
class GAT(torch.nn.Module):
    def __init__(self, n_input_features, hidden_dim, n_output_classes, n_heads, edge_dimension, dropout_rate=0.):
        super(GAT,self).__init__()

        # GNN layers
        self.conv1 = torch_geometric.nn.GATConv(n_input_features, hidden_dim, n_heads, edge_dim=edge_dimension)
        self.conv2 = torch_geometric.nn.GATConv(hidden_dim*n_heads, hidden_dim, n_heads, edge_dim=edge_dimension)

        # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_dim*n_heads, n_output_classes)

    def forward(self, x, edge_index, batch, edge_attributes=None):

        # GNN layers
        x = self.conv1(x, edge_index, edge_attr=edge_attributes)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attributes)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        # Global mean pooling (i.e. avg node features across each graph) to get a graph-level representation for graph classification
        # This requires the batch tensor, which keeps track of which nodes belong to which graphs in the batch.
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        # Note: For now, we don't apply dropout here, since the dimension is small
        x = self.fc(x)
        return x
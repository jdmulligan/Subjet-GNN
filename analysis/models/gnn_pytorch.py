'''
Graph Neural Network using PyTorch via PyTorch Geometric
See: https://pytorch-geometric.readthedocs.io/en/latest

This class defines a model interface to handle initialization, data loading, and training:
 - init_model()
 - init_data()
 - train()

'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

import torch
import torch_geometric
import networkx
from analysis.architectures import gcn_pytorch, gat_pytorch

##################################################################
class GNN_PyTorch():

    #---------------------------------------------------------------
    def __init__(self, model_info):
        '''
        :param model_info: Dictionary of model info, containing the following keys:
                                'model': model name ('particle_gnn', 'subjet_gnn')
                                'model_settings': dictionary of model settings
                                'n_total': total number of training+val+test examples
                                'n_train': total number of training examples
                                'torch_device': torch device
                                'output_dir': output directory
                                'graph_type': type of graph ('fully_connected', 'laman_naive', 'laman_1N', 'laman_1N2N')
                           In the case of subjet GNNs, the following keys are also required, originating from the graph_constructor:
                                'r': subjet radius
                                'n_subjets_total': total number of subjets per jet
                                'subjet_graphs_dict': dictionary of subjet graphs
        '''
        self.model_info = model_info

        self.train_loader, self.test_loader = self.init_data()
        self.model = self.init_model()

    #---------------------------------------------------------------
    def init_data(self):
        '''
        Construct a Dataset and DataLoader for our graph data using PyGraph

        We use sparse COO format: (n_edges, 2) array storing pairs of indices that are connected
        '''

        # Load PyG graphs from file
        graph_filename = os.path.join(self.model_info['output_dir'], f"graphs_pyg_{self.model_info['graph_key']}.pt")
        graph_list = torch.load(graph_filename)
        print(f'Loaded {len(graph_list)} graphs from file: {graph_filename}')

        # Construct DataLoader objects
        # PyG implements its own DataLoader that creates batches with a block-diagonal adjacency matrix,
        #   which allows different numbers of nodes/edges within each batch 
        # See: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches
        print('Constructing DataLoader...')
        train_dataset = graph_list[:self.model_info['n_train']]
        test_dataset = graph_list[self.model_info['n_train']:self.model_info['n_total']]
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=self.model_info['model_settings']['batch_size'], shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=self.model_info['model_settings']['batch_size'], shuffle=True)
        print('Done.')
        print()
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')
        example_graph = graph_list[0]
        print(f'Number of features: {example_graph.num_features}')
        print(f'Number of node features: {example_graph.num_node_features}')
        print(f'Number of edge features: {example_graph.num_edge_features}')
        print(f'  Has self-loops: {example_graph.has_self_loops()}')
        print(f'  Is undirected: {example_graph.is_undirected()}')
        print(f'Example graph:')
        print(f'  Number of nodes: {example_graph.num_nodes}')
        print(f'  Number of edges: {example_graph.num_edges}')
        print(f'  Adjacency: {example_graph.edge_index.shape}')
        print()

        # Store the number of input features
        self.n_input_features = example_graph.num_node_features

        # Check that the number of edges is as expected
        N = example_graph.num_nodes
        if self.model_info['graph_type'] == 'fully_connected':
            n_expected_edges = N*(N-1)
        elif 'laman' in self.model_info['graph_type']:
            n_expected_edges = 2*N-3
        assert example_graph.num_edges == n_expected_edges

        # Visualize one of the jet graphs as an example
        vis = torch_geometric.utils.convert.to_networkx(example_graph, to_undirected=True)
        plt.figure(1,figsize=(10,10))
        networkx.draw(vis, node_size=10, linewidths=6)
        plt.savefig(os.path.join(self.model_info['output_dir'], f"{self.model_info['model_key']}.png"))
        plt.close()

        return train_loader, test_loader

    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''
        hidden_dim = self.model_info['model_settings']['hidden_dim']
        n_output_classes = 2
        if self.model_info['model'] in ['particle_gcn_pytorch', 'subjet_gcn_pytorch']:
            return gcn_pytorch.GCN(self.n_input_features, hidden_dim, n_output_classes)
        if self.model_info['model'] in ['particle_gat', 'subjet_gat']:
            n_heads = self.model_info['model_settings']['n_heads']
            edge_dimension = 1
            return gat_pytorch.GAT(self.n_input_features, hidden_dim, n_output_classes, n_heads, edge_dimension)

    #---------------------------------------------------------------
    def train(self):
        print(f'Training...')
        start_time = time.time()
        
        self.model.to(self.model_info['torch_device'])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_info['model_settings']['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()

        loss_list = []
        for epoch in range(1, self.model_info['model_settings']['epochs']+1):
            for batch in self.train_loader:
                batch = batch.to(self.model_info['torch_device'])
                out = self._forward(batch)              # Forward pass
                loss = criterion(out, batch.y)          # Compute loss
                loss_list.append(loss.item())           # Store loss for plotting
                loss.backward()                         # Compute gradients
                optimizer.step()                        # Update model parameters
                optimizer.zero_grad()                   # Clear gradients.

            train_acc = self._accuracy(self.train_loader)
            test_acc = self._accuracy(self.test_loader)
            print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.model_info['output_dir'], f"model_{self.model_info['model_key']}.pt"))
        print(f'--- runtime: {time.time() - start_time} seconds ---')
        print()

        # Evaluate model on test set
        pred_graphs_list = []
        label_graphs_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                pred_graph = self._forward(batch.to(self.model_info['torch_device']))
                pred_graphs_list.append(pred_graph.cpu().data.numpy())
                label_graphs_list.append(batch.y.cpu().data.numpy())
            pred_graphs = np.concatenate(pred_graphs_list, axis=0)
            label_graphs = np.concatenate(label_graphs_list, axis=0)
            auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
            roc_curve = sklearn.metrics.roc_curve(label_graphs, pred_graphs[:,1])
            return auc, roc_curve

    #---------------------------------------------------------------
    def _forward(self, batch):
        '''
        Forward pass of a model on a batch
        (includes edge features or not, depending on whether they are requested in config)

        Note: edge features may not be supported for GCN, it seems
        '''
        if self.model_info['model_settings']['edge_features']:
            out = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        else:
            out = self.model(batch.x, batch.edge_index, batch.batch)
        return out 

    #---------------------------------------------------------------
    def _accuracy(self, loader):
        correct = 0
        for batch in loader:
            batch = batch.to(self.model_info['torch_device'])
            out = self._forward(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
        accuracy = correct / len(loader.dataset)
        return accuracy
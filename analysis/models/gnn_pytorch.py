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
import pickle
import tqdm
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

import energyflow
import torch
import torch_geometric
import networkx
from analysis.architectures import gcn, gat

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

        # Construct list of PyG graph objects
        print('Looping through jets to construct DataLoader...')
        if self.model_info['model'] in ['particle_gcn', 'particle_gat']:
            graph_list = self._init_particle_graphs()
        elif self.model_info['model'] in ['subjet_gcn', 'subjet_gat']:
            graph_list = self._init_subjet_graphs()

        # Construct DataLoader objects
        # PyG implements its own DataLoader that creates batches with a block-diagonal adjacency matrix,
        #   which allows different numbers of nodes/edges within each batch 
        # See: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches
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
    def _init_particle_graphs(self):
        '''
        Construct a list of PyG graphs for the particle-based GNNs from the energyflow dataset

        Graph structure:
         - Nodes: particle four-vectors
         - Edges: no edge features
         - Connectivity: fully connected (TODO: implement other connectivities)
        '''
        assert self.model_info['graph_type'] == 'fully_connected'

        # Load dataset; ignore pid information for now
        X, y = energyflow.qg_jets.load(self.model_info['n_total'])
        X = X[:,:,:3]
        print(f'  (n_jets, n_particles, features): {X.shape}')

        # Preprocess by centering jets and normalizing pts
        for x in tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X)):    
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()

        # TODO: there seems to be some issue with multiprocessing and the PyG data structure
        #       that causes "too many files" error -- for now we just use a single for loop
        #n_processes = multiprocessing.cpu_count()
        #print(f'  Multiprocessing with {n_processes} processes...')
        #with multiprocessing.Pool(processes=n_processes) as pool:
        #    args = [(x, y[i]) for i,x in enumerate(X)]
        #    graph_list = pool.map(self._init_particle_graph, args)

        graph_list = []       
        args = [(x, y[i]) for i, x in enumerate(X)] 
        for arg in tqdm.tqdm(args, desc='  Constructing PyG graphs', total=len(args)):
            graph_list.append(self._init_particle_graph(arg))
        return graph_list

    #---------------------------------------------------------------
    @staticmethod
    def _init_particle_graph(args):
        '''
        Construct a single PyG graph for the particle-based GNNs from the energyflow dataset
        '''
        x, label = args

        # Node features -- remove the zero pads
        x = x[~np.all(x == 0, axis=1)]
        node_features = torch.tensor(x,dtype=torch.float)

        # Edge connectivity -- fully connected
        adj_matrix = np.ones((x.shape[0],x.shape[0])) - np.identity((x.shape[0]))
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row,col)))
        edge_indices = torch.tensor(coo)
        edge_indices_long = edge_indices.t().to(torch.long).view(2, -1)

        # Construct graph as PyG data object
        graph_label = torch.tensor(label, dtype=torch.int64)
        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label)
        return graph

    #---------------------------------------------------------------
    def _init_subjet_graphs(self):
        '''
        Construct a list of PyG graphs for the subjet-based GNNs, loading from JFN dataset
 
        Graph structure:
         - Nodes: particle four-vectors3
         - Edges: pairwise angles as edge features
         - Connectivity: various connectivity options (see graph_constructor.py)
        '''
        key_prefix = f"subjet_r{self.model_info['r']}_N{self.model_info['n_subjets_total']}"
        graph_type = self.model_info['graph_type']
        z = self.model_info['subjet_graphs_dict'][f'{key_prefix}_sub_z'][:self.model_info['n_total']]
        rap = self.model_info['subjet_graphs_dict'][f'{key_prefix}_sub_rap'][:self.model_info['n_total']]
        phi = self.model_info['subjet_graphs_dict'][f'{key_prefix}_sub_phi'][:self.model_info['n_total']]
        angles = self.model_info['subjet_graphs_dict'][f'{key_prefix}_{graph_type}_edge_values'][:self.model_info['n_total']]
        edge_connections = self.model_info['subjet_graphs_dict'][f'{key_prefix}_{graph_type}_edge_connections'][:self.model_info['n_total']]
        labels = self.model_info['subjet_graphs_dict']['labels'][:self.model_info['n_total']]
        n_jets = z.shape[0]

        print(f'  Number of jets: {n_jets}')
        graph_list = []       
        args = [(z[i], rap[i], phi[i], angles[i,:], edge_connections[i,:,:], labels[i]) for i in range(n_jets)]
        for arg in tqdm.tqdm(args, desc='  Constructing PyG graphs', total=len(args)):
            graph_list.append(self._init_subjet_graph(arg))
        return graph_list

    #---------------------------------------------------------------
    @staticmethod
    def _init_subjet_graph(args):
        '''
        Construct a single PyG graph for the subjet-based GNNs from the JFN dataset
        '''
        z_i, rap_i, phi_i, angles_i, edge_connections_i, label = args

        # Node features -- remove the zero pads
        # TODO: add rap,phi as node features, if pairwise angles are not used as edge features
        node_mask = np.logical_not((z_i == 0) & (rap_i == 0) & (phi_i == 0))
        z_i = z_i[node_mask]
        node_features = torch.tensor(z_i, dtype=torch.float)
        node_features = node_features.reshape(-1,1)

        # Edge connectivity and features -- remove the zero pads
        edge_mask = ~np.all(edge_connections_i == [-1, -1], axis=1)
        edge_connections_i = edge_connections_i[edge_mask]
        angles_i = angles_i[edge_mask]
        edge_indices = torch.tensor(edge_connections_i)
        edge_indices_long = edge_indices.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(angles_i, dtype=torch.float).reshape(-1,1)  

        # Construct graph as PyG data object
        graph_label = torch.tensor(label,dtype=torch.int64)
        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=edge_attr, y=graph_label) 
        return graph

    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''
        hidden_dim = self.model_info['model_settings']['hidden_dim']
        n_output_classes = 2
        if self.model_info['model'] in ['particle_gcn', 'subjet_gcn']:
            return gcn.GCN(self.n_input_features, hidden_dim, n_output_classes)
        if self.model_info['model'] in ['particle_gat', 'subjet_gat']:
            n_heads = self.model_info['model_settings']['n_heads']
            edge_dimension = 1
            return gat.GAT(self.n_input_features, hidden_dim, n_output_classes, n_heads, edge_dimension)

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
                pred_graph = self._forward(batch)
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
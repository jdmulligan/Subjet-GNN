#!/usr/bin/env python3

"""
Train GNNs to classify jets
"""

import os
import sys
import yaml
import pickle
import time
import energyflow
from collections import defaultdict

# Pytorch
import torch
import torch_geometric
import networkx

# sklearn
import sklearn
from sklearn import metrics

# Data analysis and plotting
import numpy as np
from matplotlib import pyplot as plt

# Base class
sys.path.append('.')
from base import common_base

################################################################
class MLAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()
            
        # Set torch device
        os.environ['TORCH'] = torch.__version__
        print()
        print(f'pytorch version: {torch.__version__}')
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.torch_device)
        if self.torch_device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

        print(self)
        print()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac = 1. * self.n_val / (self.n_train + self.n_val)

        self.label_0 = config['label_0']
        self.label_1 = config['label_1']

        # Subjet Basis
        self.r_list = config['r']
        self.subjet_basis = config['subjet_basis']
        # For 'exclusive' we need to make sure we don't lose information so we need r=0.4
        if self.subjet_basis == 'exclusive':
            if self.r_list != [self.R]:
                print(f'ERROR: Wrong subjet radius r. For exlusive basis we need r = {self.R}')
                print()
                print(f'Changing radius to r = {self.R}')
                self.r_list = [self.R]

        # Initialize model-specific settings
        self.models = config['models']
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = config[model]
            
    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self, subjet_graphs_dict):
    
        # Clear variables
        self.AUC, self.AUC_av, self.AUC_std = defaultdict(list), defaultdict(list), defaultdict(list)
        self.roc_curve_dict = self.recursive_defaultdict()

        # Train ML models
        for model in self.models:
            print()
        
            model_settings = self.model_settings[model]

            # Particle GNN
            if model == 'particle_gnn':
                for graph_type in model_settings['graph_types']:
                    self.fit_particle_gnn(model, model_settings, graph_type)

            # Subjet GNNs
            if model == 'subjet_gnn':
                for r in self.r_list:
                    for n_subjets_total in subjet_graphs_dict['n_subjets_total']:
                        for graph_type in model_settings['graph_types']:
                            self.fit_subjet_gnn(subjet_graphs_dict, model, model_settings, r, n_subjets_total, graph_type)

        # Save ROC curves to file
        if self.models:
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)

    #---------------------------------------------------------------
    # Fit Graph Neural Network with particle four-vectors
    #---------------------------------------------------------------
    def fit_particle_gnn(self, model, model_settings, graph_type):
        print(f'--- Fitting particle GNN {graph_type} ---')
        start_time = time.time()

        # load data
        X, y = energyflow.qg_jets.load(self.n_total)

        # ignore pid information for now
        X = X[:,:,:3]

        # preprocess by centering jets and normalizing pts
        for x in X:
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()
        
        print(f'(n_jets, n_particles, features): {X.shape}')

        # Loop over all jets and construct graphs
        graph_list = []
        for i, xp in enumerate(X):
            
            # 1. Get node feature vector 
            #    First need to remove zero padding
            xp = xp[~np.all(xp == 0, axis=1)]
            node_features = torch.tensor(xp,dtype=torch.float)

            # 2. Get adjacency matrix / edge indices
            
            # Fully connected graph 
            adj_matrix = np.ones((xp.shape[0],xp.shape[0])) - np.identity((xp.shape[0]))
            row, col = np.where(adj_matrix)
            
            # Use sparse COO format: (n_edges, 2) array storing pairs of indices that are connected
            coo = np.array(list(zip(row,col)))

            #    Switch format
            edge_indices = torch.tensor(coo)
            edge_indices_long = edge_indices.t().to(torch.long).view(2, -1) #long .. ?!

            #    or can use this directly: edge_indices_full_conn = torch.tensor([row,col],dtype=torch.long) 

            # 3. Can add edge features later on ...

            # 4. Get graph label
            graph_label = torch.tensor(y[i],dtype=torch.int64)

            # 5. Create PyG data object
            graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label).to(self.torch_device)

            # 6. Add to list of graphs
            graph_list.append(graph)
               
        # 7. Create PyG batch object that contains all the graphs and labels
        graph_batch = torch_geometric.data.Batch().from_data_list(graph_list)

        # Print
        print(f'Number of graphs in PyG batch object: {graph_batch.num_graphs}')
        print(f'Graph batch structure: {graph_batch}') # It says "DataDataBatch" .. correct?

        # [Check the format that is required by the GNN classifier!!]
        # [Fully connected edge index, ok: N*(N-1) = 18 * 17 = 306 ]

        # Visualize one of the jet graphs as an example ...
        # Are the positions adjusted if we include edge features?
        vis = torch_geometric.utils.convert.to_networkx(graph_batch[3], to_undirected=True) #... undirected graph?
        plt.figure(1,figsize=(10,10))
        networkx.draw(vis, node_size=10, linewidths=6)
        plt.savefig(os.path.join(self.output_dir, 'particle_gnn_graph.pdf'))
        plt.close()

        # Check adjacency of the first jet
        print()
        print(f'Adjacency of first jet: {graph_batch[0].edge_index}')
        print()

        # Check a few things .. 
        # 1. Graph batch
        print(f'Number of graphs: {graph_batch.num_graphs}') # correct number ...??
        print(f'Number of features: {graph_batch.num_features}') # ok
        print(f'Number of node features: {graph_batch.num_node_features}') # ok
        #print(f'Number of classes: {graph_batch.num_classes}') # .. labels?? print(graph_batch.y)

        # 2. A particular graph
        print(f'Number of nodes: {graph_batch[1].num_nodes}') # ok
        print(f'Number of edges: {graph_batch[1].num_edges}') # ok, 17*16=272
        print(f'Has self-loops: {graph_batch[1].has_self_loops()}') # ok
        print(f'Is undirected: {graph_batch[1].is_undirected()}') # ok

        # Shuffle and split into training and test set
        #graph_batch = graph_batch.shuffle() # seems like I have the wrong format ...

        train_dataset = graph_batch[:self.n_train]
        test_dataset = graph_batch[self.n_train:self.n_total]

        print(f'Number of training graphs: {len(train_dataset)}') # now ok, doesn't work for graph_batch ...?
        print(f'Number of test graphs: {len(test_dataset)}')

        # Group graphs into mini-batches for parallelization (..?)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=model_settings['batch_size'], shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_settings['batch_size'], shuffle=True)

        # Set up GNN structure
        # 1. Embed each node by performing multiple rounds of message passing
        # 2. Aggregate node embeddings into a unified graph embedding (readout layer)
        # 3. Train a final classifier on the graph embedding
        
        gnn_model = GCN_class(graph_batch, hidden_channels = 64)  # TO DO: In order to train this model with the Laman w/ edge_attr we need to add a new train_gnn -> 
        print(f'{gnn_model}')
        gnn_model = gnn_model.to(self.torch_device)
        #print(f'gnn_model is_cuda: {gnn_model.is_cuda}')

        # Now train the fully connected GNN
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, model_settings['epochs']):
            self.train_gnn(train_loader, gnn_model, optimizer, criterion, False)
            train_acc = self.test_gnn(train_loader, gnn_model, False)
            test_acc = self.test_gnn(test_loader, gnn_model, False)
            print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        # Get AUC & ROC curve for the fully connected GNN
        for i, datatest in enumerate(test_loader):  # Iterate in batches over the test dataset.
            pred_graph = gnn_model(datatest.x, datatest.edge_index, datatest.batch).data # '.data' removes the 'grad..' from the torch tensor
            pred_graph = pred_graph.cpu().data.numpy() # Convert predictions to np.array. Not: values not in [0,1]
            label_graph = datatest.y.cpu().data.numpy() # Get labels

            if i==0:
                pred_graphs = pred_graph
                label_graphs = label_graph
            else:
                pred_graphs = np.concatenate((pred_graphs,pred_graph),axis=0)
                label_graphs = np.concatenate((label_graphs,label_graph),axis=0)

        # Get AUC
        gnn_auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
        print(f'Fully connected GNN AUC based on particle four-vectors is: {gnn_auc}')
        self.AUC[model].append(gnn_auc)

        # Get ROC curve for the fully connected GNN
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(label_graphs, pred_graphs[:,1])

        print(f'--- runtime: {time.time() - start_time} seconds ---')
        print()

    #---------------------------------------------------------------
    # Fit Graph Neural Network with subjet four-vectors
    #---------------------------------------------------------------
    def fit_subjet_gnn(self, subjet_graphs_dict, model, model_settings, r, N, graph_type):
        print(f'--- Fitting subjet GNN {graph_type} ---')

        # Initialize node,edge information
        key_prefix = f'subjet_r{r}_N{N}'
        z = subjet_graphs_dict[f'{key_prefix}_sub_z'][:self.n_total]
        rap = subjet_graphs_dict[f'{key_prefix}_sub_rap'][:self.n_total]
        phi = subjet_graphs_dict[f'{key_prefix}_sub_phi'][:self.n_total]
        angles = subjet_graphs_dict[f'{key_prefix}_{graph_type}_edge_values'][:self.n_total]
        edge_connections = subjet_graphs_dict[f'{key_prefix}_{graph_type}_edge_connections'][:self.n_total]
        labels = subjet_graphs_dict['labels'][:self.n_total]

        n_jets = z.shape[0]
        print(f'Number of jets: {n_jets}')

        # Loop through jets and construct graphs
        graph_list = []
        for i in range(n_jets):

            # Edge information
            # Use sparse COO format: (n_edges, 2) array storing pairs of indices that are connected
            # Remove the zero pads
            edge_connections_i = edge_connections[i,:,:]
            angles_i = angles[i,:]

            edge_mask = ~np.all(edge_connections_i == [-1, -1], axis=1)
            edge_connections_i = edge_connections_i[edge_mask]
            angles_i = angles_i[edge_mask]

            # Node information
            # Remove the zero pads
            node_mask = np.logical_not((z[i] == 0) & (rap[i] == 0) & (phi[i] == 0))
            z_i = z[i][node_mask]

            # Construct graph
            node_features = torch.tensor(z_i,dtype=torch.float)
            node_features = node_features.reshape(-1,1)
            
            edge_indices = torch.tensor(edge_connections_i)
            edge_indices_long = edge_indices.t().to(torch.long).view(2, -1) #long .. ?!
            edge_attr = torch.tensor(angles_i,dtype=torch.float).reshape(-1,1)  

            graph_label = torch.tensor(labels[i],dtype=torch.int64)

            graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=edge_attr, y=graph_label).to(self.torch_device) 

            graph_list.append(graph)

        graph_batch = torch_geometric.data.Batch().from_data_list(graph_list)
        print(f'Number of graphs in PyG batch object: {graph_batch.num_graphs}')
        print(f'Graph batch structure: {graph_batch}') # It says "DataDataBatch" .. correct?

        # Visualize one of the jet graphs as an example ...
        vis = torch_geometric.utils.convert.to_networkx(graph_batch[3],to_undirected=True) #... undirected graph?
        plt.figure(1,figsize=(10,10))
        networkx.draw(vis, node_size=10, linewidths=6)
        plt.savefig(os.path.join(self.output_dir, f'subjet_gnn_{graph_type}_graph.pdf'))
        plt.close()

        # Check adjacency of the first jet
        print()
        print(f'adjacency of first jet: {graph_batch[0].edge_index}')
        print()

        # Check a few things .. 
        # 1. Graph batch
        print(f'Number of graphs: {graph_batch.num_graphs}') # ok
        print(f'Number of features: {graph_batch.num_features}') # ok
        print(f'Number of node features: {graph_batch.num_node_features}') # ok
        print(f'Number of edge features: {graph_batch.num_edge_features}') # ok

        train_dataset = graph_batch[:self.n_train]
        test_dataset = graph_batch[self.n_train:self.n_total]

        print(f'Number of training graphs: {len(train_dataset)}') # now ok, doesn't work for graph_batch ...?
        print(f'Number of test graphs: {len(test_dataset)}')

        # Group graphs into mini-batches for parallelization (..?)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=model_settings['batch_size'], shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_settings['batch_size'], shuffle=True)

        # Set up GNN structure
        # 1. Embed each node by performing multiple rounds of message passing
        # 2. Aggregate node embeddings into a unified graph embedding (readout layer)
        # 3. Train a final classifier on the graph embedding
        gnn_model = GAT_class(graph_batch, hidden_channels = 8, heads = 8, edge_dimension = 1, edge_attributes = graph_batch.edge_attr)
        print(gnn_model)
        gnn_model = gnn_model.to(self.torch_device)
        #print(f'gnn_model is_cuda: {gnn_model.is_cuda}')

        # Now train the GNN
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        self.best_valid_acc = 0
        last_epochs = 0
        self.patience_gnn = 4
        for epoch in range(1, model_settings['epochs'] + 1): # -> 171
            time_start = time.time()
            loss_train = self.train_gnn(train_loader, gnn_model, optimizer, criterion, True)
            train_acc = self.test_gnn(train_loader, gnn_model, True)
            test_acc = self.test_gnn(test_loader, gnn_model, True)
            time_end = time.time()
            print(f'Epoch: {epoch:02d}, Train Loss: {loss_train:.4f},  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Duration: {time_end-time_start}')

            if test_acc > self.best_valid_acc:
                last_epochs = 0
                self.best_valid_acc = test_acc
                # Save the best model
                torch.save(gnn_model.state_dict(), 'best-model-parameters.pt')

            if last_epochs >= self.patience_gnn:
                print(f"Ending training after {epoch} epochs due to performance saturation with a patience parameter of {self.patience_gnn} epochs")
                break

            last_epochs += 1
                    
        # Use the best model
        gnn_model.load_state_dict(torch.load('best-model-parameters.pt'))

        # Get AUC & ROC curve
        for i, datatest in enumerate(test_loader):  # Iterate in batches over the test dataset.
            pred_graph = gnn_model(datatest.x, datatest.edge_index, datatest.batch, datatest.edge_attr).data # '.data' removes the 'grad..' from the torch tensor
            pred_graph = pred_graph.cpu().data.numpy() # Convert predictions to np.array. Not: values not in [0,1]
            label_graph = datatest.y.cpu().data.numpy() # Get labels

            if i==0:
                pred_graphs = pred_graph
                label_graphs = label_graph
            else:
                pred_graphs = np.concatenate((pred_graphs,pred_graph),axis=0)
                label_graphs = np.concatenate((label_graphs,label_graph),axis=0)


        # get AUC
        gnn_auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
        print()
        print(f'GNN {graph_type} with r={r}, N={N} : AUC based on subjet four-vectors is: {gnn_auc}')
        self.AUC[f'{model}_{graph_type}'].append(gnn_auc)

        # get ROC curve
        self.roc_curve_dict[f'{model}_{graph_type}'] = sklearn.metrics.roc_curve(label_graphs, pred_graphs[:,1])



    #---------------------------------------------------------------
    def train_gnn(self, train_loader, gnn_model, optimizer, criterion, edge_attr_boolean):
        gnn_model.train()

        loss_cum=0

        for data in train_loader:  # Iterate in batches over the training dataset.

            data = data.to(self.torch_device)
            if edge_attr_boolean:
                out = gnn_model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            else: 
                out = gnn_model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)  # Compute the loss.
            loss_cum += loss.item() #Cumulative loss
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        return loss_cum/len(train_loader)

    #---------------------------------------------------------------
    def test_gnn(self, loader, gnn_model, edge_attr_boolean):
        gnn_model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(self.torch_device)
            if edge_attr_boolean:
                out = gnn_model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            else: 
                out = gnn_model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

##################################################################
class GCN_class(torch.nn.Module):
    def __init__(self,graph_batch, hidden_channels):
        super(GCN_class,self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(graph_batch.num_features, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels,2)

    def forward(self, x, edge_index, batch):

        #x = torch.nn.functional.dropout(x, p=0.5, training = self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        #x = torch.nn.functional.dropout(x, p=0.5, training = self.training)
        x = self.lin(x)

        return x 

##################################################################    
class GAT_class(torch.nn.Module):
    def __init__(self, graph_batch, hidden_channels, heads, edge_dimension, edge_attributes):
        super(GAT_class,self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(graph_batch.num_features, hidden_channels, heads, edge_dim = edge_dimension)
        self.conv2 = torch_geometric.nn.GATConv(hidden_channels*heads, hidden_channels, heads, edge_dim = edge_dimension)
        #self.conv3 = torch_geometric.nn.GATConv(hidden_channels*heads, hidden_channels, heads, edge_dim = edge_dimension)
        self.lin = torch.nn.Linear(hidden_channels*heads, 2)

    def forward(self, x, edge_index, batch, edge_attributes):

        #x = torch.nn.functional.dropout(x, p=0.4, training = self.training)
        x = self.conv1(x, edge_index, edge_attr = edge_attributes )
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr = edge_attributes )
        #x = x.relu()
        #x = self.conv3(x,edge_index,edge_attr = edge_attributes )

        x = torch_geometric.nn.global_mean_pool(x, batch)

        #x = torch.nn.functional.dropout(x, p=0.4, training = self.training)
        x = self.lin(x)

        return x
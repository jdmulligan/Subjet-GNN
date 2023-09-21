#!/usr/bin/env python3

"""
The graph_constructor module constructs the input graphs to the ML analysis:
    - graphs_numpy_subjet.h5: builds graphs from JFN output subjets_unshuffled.h5
    - graphs_pyg_subjet__{graph_key}.pt: builds PyG graphs from subjet_graphs_numpy.h5
    - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
"""

import os
import sys
import tqdm
import yaml
import numpy as np
import numba

import energyflow

import torch
import torch_geometric

import data_IO

#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def construct_graphs(input_data, config_file, output_dir, use_precomputed_graphs=False):
    '''
    Construct graphs:
      - Particle graphs are constructed from energyflow dataset
      - Subjet graphs are constructed from JFN dataset

    Several graph structures are generated:
      - Subjet graphs: Fully connected, Laman graphs (naive, 1N, 1N2N)
      - Particle graphs: Fully connected

    There are several different feature constructions as well:
      - Node features: 
          - Subjet graphs: (z)
          - Particle graphs: (z,y,phi)
      - Edge features:
          - Subjet graphs: pairwise angles
          - Particle graphs: no edge features
    TODO: implement more comprehensive options

    The graphs are saved in several formats:
      - graphs_numpy_subjet.h5: numpy arrays
      - graphs_pyg_subjet__{graph_key}.pt: PyG data objects
      - graphs_pyg_particle__{graph_key}.pt: PyG data objects
    '''
    #------------------------
    # Construct subjet graphs
    print('------------------- Subjet graphs --------------------')
    graph_structures = ['laman_naive', 'laman_1N', 'laman_1N2N']

    # Numpy format
    _construct_subjet_graphs_numpy(input_data, graph_structures, config_file, output_dir, use_precomputed_graphs)

    # PyG format
    _construct_subjet_graphs_pyg(output_dir, graph_structures)

    #------------------------
    # Construct particle graphs
    print()
    print('------------------- Particle graphs --------------------')
    graph_structures = ['fully_connected']

    # PyG format
    _construct_particle_graphs_pyg(output_dir, graph_structures, N=500000)

#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def _construct_subjet_graphs_numpy(input_data, graph_structures, config_file, output_dir, use_precomputed_graphs=False):
    '''
    input_data:  dict of ndarrays storing subjet kinematics
    '''

    # We will write a HDF5 file containing all the info we will need for the ML training
    output_dict = data_IO.recursive_defaultdict()

    # Loop through all configs
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    subjet_basis = config['subjet_basis']
    if subjet_basis == 'inclusive':
        N_cluster_list = input_data['N_max']
    elif subjet_basis == 'exclusive':
        N_cluster_list = input_data['njet']
    else:
        sys.exit(f'ERROR: Invalid choice for subjet_basis')
    r_list = input_data['r_list'] 
    for r in r_list:
        for n_subjets_total in N_cluster_list:
            key_prefix = f'subjet_r{r}_N{n_subjets_total}'

            # Get the (z,y,phi) of subjets from the input file
            subjet_z = input_data[f'{key_prefix}_z']
            subjet_rap = input_data[f'{key_prefix}_sub_rap']
            subjet_phi = input_data[f'{key_prefix}_sub_phi']
            labels = input_data['y']

            # Shuffle the data sets, to ensure that the first N entries are not from the same class
            idx = np.random.permutation(labels.shape[0])
            subjet_z = subjet_z[idx]
            subjet_rap = subjet_rap[idx]
            subjet_phi = subjet_phi[idx]
            labels = labels[idx]

            #-----------------------------------
            # Construct graphs
            n_jets = subjet_z.shape[0]
            print()
            print(f'Constructing numpy subjet graphs for {n_jets} jets: r={r}, n_subjets_total={n_subjets_total} ({subjet_basis} clustering) ...')
            for graph_type in graph_structures:
                print(f'  graph_type: {graph_type}')

                #-----------------------------------
                # Call the appropriate graph construction function for each set of subjets
                # Note: we need to define n_edges here in order to take advantage of numba's jit
                if graph_type == 'fully_connected':
                    n_edges = n_subjets_total*(n_subjets_total-1)
                    f = _fully_connected
                elif graph_type == 'laman_naive':
                    n_edges = 2*n_subjets_total-3
                    f = _laman_naive
                elif graph_type == 'laman_1N':
                    n_edges = 2*n_subjets_total-3
                    f = _laman_1N
                elif graph_type == 'laman_1N2N':
                    n_edges = 2*n_subjets_total-3
                    f = _laman_1N2N

                # Option: If the graphs have already been constructed in the input file, then get the graphs directly
                if use_precomputed_graphs:
                    precomputed_graph_type = input_data['Laman_construction']
                    if graph_type != f'laman_{precomputed_graph_type}':
                        continue
                    edge_connections = input_data[f'{key_prefix}_edges']
                    edge_values = input_data[f'{key_prefix}_angles']
                # Otherwise, construct the graphs by looping through the subjets
                else:
                    edge_connections, edge_values = _compute_edges(n_jets, n_subjets_total, n_edges, f, subjet_z, subjet_rap, subjet_phi, graph_type)
                print(f'    edge_connections: {edge_connections.shape}')

                # Check expected number of nonzero nodes/edges on an example graph
                node_mask = np.logical_not((subjet_z[0] == 0) & (subjet_rap[0] == 0) & (subjet_phi[0] == 0))
                edge_mask = ~np.all(edge_connections[0] == [-1, -1], axis=1)
                n_nodes = subjet_z[0][node_mask].shape[0]
                n_edges = edge_connections[0][edge_mask].shape[0]
                if graph_type == 'fully_connected':
                    n_expected_edges = n_nodes*(n_nodes-1)
                elif 'laman' in graph_type:
                    n_expected_edges = 2*n_nodes-3
                if n_nodes > 1:
                    assert n_edges == n_expected_edges, f'ERROR: n_edges != n_expected_edges: {n_edges} != {n_expected_edges} (n_nodes = {n_nodes})'
                #-----------------------------------

                # Populate output
                output_dict[f'{key_prefix}_{graph_type}_edge_connections'] = edge_connections
                output_dict[f'{key_prefix}_{graph_type}_edge_values'] = edge_values

                output_dict['labels'] = labels
                output_dict[f'{key_prefix}_sub_z'] = subjet_z
                output_dict[f'{key_prefix}_sub_rap'] = subjet_rap
                output_dict[f'{key_prefix}_sub_phi'] = subjet_phi

    # Write dict to HDF5
    output_dict['subjet_basis'] = subjet_basis
    output_dict['r_list'] = r_list
    output_dict['n_subjets_total'] = N_cluster_list
    print()
    data_IO.write_data(output_dict, output_dir, filename='graphs_numpy_subjet.h5')

#---------------------------------------------------------------
# We will return two numpy arrays
#  - edge_connections: shape (n_jets, n_edges, 2) that list set of (i,j) node connections for each jet
#  - edge_values: shape (n_jets, n_edges) that list set of edge values for each jet
# We will also propagate zero pads (i.e. subjet z=y=phi=0) as edges denoted [-1,-1], to keep fixed-size arrays
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False) 
def _compute_edges(n_jets, n_subjets_total, n_edges, f, subjet_z, subjet_rap, subjet_phi, graph_type):

    edge_connections = np.zeros((n_jets, n_edges, 2), dtype=np.int32)
    edge_values = np.zeros((n_jets, n_edges), dtype=np.float32)
    for i_jet in numba.prange(n_jets):

        z = subjet_z[i_jet]
        rap = subjet_rap[i_jet]
        phi = subjet_phi[i_jet]

        # Count the number of (non zero-padded) subjets
        n_subjets_nonzero = 0
        for i_subjet in numba.prange(n_subjets_total):
            if z[i_subjet] != 0 or rap[i_subjet] != 0 or phi[i_subjet] != 0:
                n_subjets_nonzero += 1

        # Construct the graph for this jet
        edge_connections[i_jet], edge_values[i_jet] = f(z, rap, phi,
                                                        n_subjets_total, 
                                                        n_subjets_nonzero, 
                                                        n_edges)
        
    return edge_connections, edge_values

#---------------------------------------------------------------
# Fully connected graph
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def _fully_connected(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):   

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for i in numba.prange(n_subjets_total):
        for j in numba.prange(n_subjets_total):
            if j > i:
                angle = _delta_R(subjet_rap[i], subjet_phi[i], subjet_rap[j], subjet_phi[j])
                edge_connections[edge_idx] = np.array([i, j])
                edge_values[edge_idx] = angle
                edge_idx += 1

    return edge_connections, edge_values

#---------------------------------------------------------------
# Henneberg construction using Type 1 connections
# To start, let's just build based on pt ordering
# A simple construction is to have each node N connected to nodes N+1,N+2
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def _laman_naive(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):   

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in numba.prange(n_subjets_total):

        if N < n_subjets_total-1: 
            if N < n_subjets_nonzero-1: 
                angle = _delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+1], subjet_phi[N+1])
                edge_connections[edge_idx] = np.array([N, N+1])
                edge_values[edge_idx] = angle
                edge_idx += 1

        if N < n_subjets_total-2:
            if N < n_subjets_nonzero-2:
                angle = _delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+2], subjet_phi[N+2])
                edge_connections[edge_idx] = np.array([N, N+2])
                edge_values[edge_idx] = angle
                edge_idx += 1

    return edge_connections, edge_values

#---------------------------------------------------------------
# 
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def _laman_1N(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in numba.prange(n_subjets_total):
        if N == 0:
            for i in range(n_subjets_total-1): # Because we want to start from i=1 
                if i < n_subjets_nonzero-1:
                    angle = _delta_R(subjet_rap[0], subjet_phi[0], subjet_rap[i+1], subjet_phi[i+1])
                    edge_connections[edge_idx] = np.array([0, i+1])
                    edge_values[edge_idx] = angle
                    edge_idx += 1
            
        elif N < n_subjets_total-1:
            if N < n_subjets_nonzero-1:
                angle = _delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+1], subjet_phi[N+1])
                edge_connections[edge_idx] = np.array([N, N+1])
                edge_values[edge_idx] = angle
                edge_idx += 1

    return edge_connections, edge_values

#---------------------------------------------------------------
# 
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def _laman_1N2N(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in range(n_subjets_total):

        if N == 0:
            for i in range(n_subjets_nonzero-1): # Because we want to start from i=1
                if i < n_subjets_nonzero-1:
                    angle = _delta_R(subjet_rap[0], subjet_phi[0], subjet_rap[i+1], subjet_phi[i+1])
                    edge_connections[edge_idx] = np.array([0, i+1])
                    edge_values[edge_idx] = angle
                    edge_idx += 1

        elif N == 1:
            for i in range(n_subjets_nonzero-2): # Because we want to start from i=2
                if i < n_subjets_nonzero-2:
                    angle = _delta_R(subjet_rap[1], subjet_phi[1], subjet_rap[i+2], subjet_phi[i+2])
                    edge_connections[edge_idx] = np.array([1, i+2])
                    edge_values[edge_idx] = angle
                    edge_idx += 1

    return edge_connections, edge_values

#---------------------------------------------------------------
# Compute delta_R = sqrt(delta_y^2 + delta_phi^2) between two subjets
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def _delta_R(y1, phi1, y2, phi2):

    delta_y = y1 - y2

    # We must shift phi to avoid boundary effect
    # We want to constrain delta_phi to be in the range [-pi, pi]
    delta_phi = phi1 - phi2
    if delta_phi > np.pi:
        delta_phi -= 2*np.pi
    elif delta_phi < -np.pi:
        delta_phi += 2*np.pi

    return np.sqrt(delta_y**2 + delta_phi**2)

#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def _construct_subjet_graphs_pyg(output_dir, graph_structures):
    '''
    Construct a list of PyG graphs for the subjet-based GNNs, loading from JFN dataset
    '''
    graph_filename = os.path.join(output_dir, "graphs_numpy_subjet.h5")
    subjet_graphs_dict = data_IO.read_data(graph_filename)

    for r in subjet_graphs_dict['r_list']:
        for n_subjets_total in subjet_graphs_dict['n_subjets_total']:
            key_prefix = f'subjet_r{r}_N{n_subjets_total}'
            z = subjet_graphs_dict[f'{key_prefix}_sub_z']
            rap = subjet_graphs_dict[f'{key_prefix}_sub_rap']
            phi = subjet_graphs_dict[f'{key_prefix}_sub_phi']
            n_jets = z.shape[0]
            for graph_structure in graph_structures:
                graph_key = f'subjet__{graph_structure}__r{r}__n{n_subjets_total}'

                angles = subjet_graphs_dict[f'{key_prefix}_{graph_structure}_edge_values']
                edge_connections = subjet_graphs_dict[f'{key_prefix}_{graph_structure}_edge_connections']
                labels = subjet_graphs_dict['labels']

                graph_list = []       
                args = [(z[i], rap[i], phi[i], angles[i,:], edge_connections[i,:,:], labels[i]) for i in range(n_jets)]
                for arg in tqdm.tqdm(args, desc=f'  Constructing PyG graphs: {graph_key}', total=len(args)):
                    graph_list.append(_construct_subjet_graph_pyg(arg))

                # Save to file using pytorch (separate files for each graph type, to limit memory)
                graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}.pt")
                torch.save(graph_list, graph_filename)
                print(f'  Saved PyG graphs to {graph_filename}.')

#---------------------------------------------------------------
def _construct_subjet_graph_pyg(args):
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
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def _construct_particle_graphs_pyg(output_dir, graph_structures, N=500000):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected (TODO: implement other connectivities)
    '''
    print(f'Constructing PyG particle graphs from energyflow dataset...')

    # Load dataset; ignore pid information for now
    X, y = energyflow.qg_jets.load(N)
    X = X[:,:,:3]
    print(f'  (n_jets, n_particles, features): {X.shape}')

    # Preprocess by centering jets and normalizing pts
    for x in tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X)):    
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()

    for graph_structure in graph_structures:
        graph_key = f'particle__{graph_structure}'

        # TODO: there seems to be some issue with multiprocessing and the PyG data structure
        #       that causes "too many files" error -- for now we just use a single for loop
        #n_processes = multiprocessing.cpu_count()
        #print(f'  Multiprocessing with {n_processes} processes...')
        #with multiprocessing.Pool(processes=n_processes) as pool:
        #    args = [(x, y[i]) for i,x in enumerate(X)]
        #    graph_list = pool.map(self._init_particle_graph, args)

        graph_list = []       
        args = [(x, y[i]) for i, x in enumerate(X)] 
        for arg in tqdm.tqdm(args, desc=f'  Constructing PyG graphs: {graph_key}', total=len(args)):
            graph_list.append(_construct_particle_graph_pyg(arg))

        # Save to file using pytorch
        graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}.pt")
        torch.save(graph_list, graph_filename)
        print(f'  Saved PyG graphs to {graph_filename}.')

#---------------------------------------------------------------
def _construct_particle_graph_pyg(args):
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
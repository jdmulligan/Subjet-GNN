#!/usr/bin/env python3

"""
This module constructs graphs from subjets and writes them to a new HDF5 file
"""

import sys
import yaml
import numpy as np
import numba

import data_IO

#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def construct_graphs(input_data, config_file, output_dir):
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
            k = 1000
            subjet_z = input_data[f'{key_prefix}_z'][:k]
            subjet_rap = input_data[f'{key_prefix}_sub_rap'][:k]
            subjet_phi = input_data[f'{key_prefix}_sub_phi'][:k]

            #-----------------------------------
            # Construct graphs
            n_jets = subjet_z.shape[0]
            print()
            print(f'Constructing graphs for {n_jets} jets: r={r}, n_subjets_total={n_subjets_total} ({subjet_basis} clustering) ...')
            graphs = ['laman_naive', 'laman_1N', 'laman_1N2N']
            for graph_type in graphs:

                #-----------------------------------
                # Call the appropriate graph construction function for each set of subjets
                # Note: we need to define n_edges here in order to take advantage of numba's jit
                if graph_type == 'laman_naive':
                    n_edges = 2*n_subjets_total-3
                    f = laman_naive
                elif graph_type == 'laman_1N':
                    n_edges = 2*n_subjets_total-3
                    f = laman_1N
                elif graph_type == 'laman_1N2N':
                    n_edges = 2*n_subjets_total-3
                    f = laman_1N2N

                edge_connections, edge_values = compute_edges(n_jets, n_subjets_total, n_edges, f, subjet_z, subjet_rap, subjet_phi)
                print(f'  {graph_type}')
                print(f'    edge_connections: {edge_connections.shape}')
                print(f'    edge_values: {edge_values.shape}')
                #-----------------------------------

                # Populate output
                output_dict[f'{key_prefix}_{graph_type}_edge_connections'] = edge_connections
                output_dict[f'{key_prefix}_{graph_type}_edge_values'] = edge_values

                output_dict[f'{key_prefix}_sub_z'] = subjet_z
                output_dict[f'{key_prefix}_sub_rap'] = subjet_rap
                output_dict[f'{key_prefix}_sub_phi'] = subjet_phi

    # Write dict to HDF5
    output_dict['subjet_basis'] = subjet_basis
    output_dict['r_list'] = r_list
    output_dict['n_subjets_total'] = n_subjets_total
    print()
    data_IO.write_data(output_dict, output_dir, filename='subjet_graphs.h5')

#---------------------------------------------------------------
# We will return two numpy arrays
#  - edge_connections: shape (n_jets, n_edges, 2) that list set of (i,j) node connections for each jet
#  - edge_values: shape (n_jets, n_edges) that list set of edge values for each jet
# We will also propagate zero pads (i.e. subjet z=y=phi=0) as edges denoted [-1,-1], to keep fixed-size arrays
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=True) 
def compute_edges(n_jets, n_subjets_total, n_edges, f, subjet_z, subjet_rap, subjet_phi):

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
# Henneberg construction using Type 1 connections
# To start, let's just build based on pt ordering
# A simple construction is to have each node N connected to nodes N+1,N+2
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=True)
def laman_naive(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):   

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in numba.prange(n_subjets_total):

        if N < n_subjets_total-1: 
            if N < n_subjets_nonzero-1: 
                angle = delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+1], subjet_phi[N+1])
                edge_connections[edge_idx] = np.array([N, N+1])
                edge_values[edge_idx] = angle
                edge_idx += 1

        if N < n_subjets_total-2:
            if N < n_subjets_nonzero-2:
                angle = delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+2], subjet_phi[N+2])
                edge_connections[edge_idx] = np.array([N, N+2])
                edge_values[edge_idx] = angle
                edge_idx += 1

    return edge_connections, edge_values

#---------------------------------------------------------------
# 
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=True)
def laman_1N(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in numba.prange(n_subjets_total):
        if N == 0:
            for i in range(n_subjets_total-1): # Because we want to start from i=1 
                if i < n_subjets_nonzero-1:
                    angle = delta_R(subjet_rap[0], subjet_phi[0], subjet_rap[i+1], subjet_phi[i+1])
                    edge_connections[edge_idx] = np.array([0, i+1])
                    edge_values[edge_idx] = angle
            
        elif N < n_subjets_total-1:
            if N < n_subjets_nonzero-1:
                angle = delta_R(subjet_rap[N], subjet_phi[N], subjet_rap[N+1], subjet_phi[N+1])
                edge_connections[edge_idx] = np.array([N, N+1])
                edge_values[edge_idx] = angle

    return edge_connections, edge_values

#---------------------------------------------------------------
# 
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=True)
def laman_1N2N(subjet_z, subjet_rap, subjet_phi, n_subjets_total, n_subjets_nonzero, n_edges):

    # Loop through subjets to construct the graph
    edge_connections = np.full((n_edges, 2), -1)
    edge_values = np.zeros((n_edges), dtype=np.float32)
    edge_idx = 0
    for N in range(n_subjets_total):

        if N == 0:
            for i in range(n_subjets_nonzero-1): # Because we want to start from i=1
                if i < n_subjets_nonzero-1:
                    angle = delta_R(subjet_rap[0], subjet_phi[0], subjet_rap[i+1], subjet_phi[i+1])
                    edge_connections[edge_idx] = np.array([0, i+1])
                    edge_values[edge_idx] = angle

        elif N == 1:
            for i in range(n_subjets_nonzero-2): # Because we want to start from i=2
                if i < n_subjets_nonzero-2:
                    angle = delta_R(subjet_rap[1], subjet_phi[1], subjet_rap[i+2], subjet_phi[i+2])
                    edge_connections[edge_idx] = np.array([1, i+2])
                    edge_values[edge_idx] = angle

    return edge_connections, edge_values

#---------------------------------------------------------------
# Compute delta_R = sqrt(delta_y^2 + delta_phi^2) between two subjets
#---------------------------------------------------------------
@numba.jit(nopython=True, parallel=False)
def delta_R(y1, phi1, y2, phi2):

    delta_y = y1 - y2

    # We must shift phi to avoid boundary effect
    # We want to constrain delta_phi to be in the range [-pi, pi]
    delta_phi = phi1 - phi2
    if delta_phi > np.pi:
        delta_phi -= 2*np.pi
    elif delta_phi < -np.pi:
        delta_phi += 2*np.pi

    return np.sqrt(delta_y**2 + delta_phi**2)
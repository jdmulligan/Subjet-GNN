#! /usr/bin/env python
'''
Main script to steer graph construction and GNN training
'''

import argparse
import os
import sys
import yaml
import time

import torch

import data_IO
import graph_constructor
import ml_analysis
import plot_results

# Base class
sys.path.append('.')
from base import common_base

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, use_precomputed_graphs=False, **kwargs):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs
        self.use_precomputed_graphs = use_precomputed_graphs

        self.initialize(config_file)

        print()
        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self, config_file):
        print('Initializing class objects')

        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        '''
        For now, we will assume that MC samples have been used to
        generate a dataset of particles and subjets (subjets_unshuffled.h5).
        
        Existing datasets are listed here:
        https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90/edit#gid=0
        
        The graph_constructor module constructs the input graphs to the ML analysis:
          - graphs_numpy_subjet.h5: builds graphs from JFN output subjets_unshuffled.h5
          - graphs_pyg_subjet__{graph_key}.pt: builds PyG graphs from subjet_graphs_numpy.h5
          - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
        '''

        # Check whether the graphs file has already been generated, and if not, generate it
        graph_numpy_subjet_file = os.path.join(self.output_dir, 'graphs_numpy_subjet.h5')
        print('========================================================================')
        if self.regenerate_graphs or not os.path.exists(graph_numpy_subjet_file):
            input_data = data_IO.read_data(self.input_file)
            graph_constructor.construct_graphs(input_data, self.config_file, self.output_dir, self.use_precomputed_graphs)
        else:
            print(f'Subjet numpy graphs found: {graph_numpy_subjet_file}')

        # Perform ML analysis, and write results to file
        print()
        print('========================================================================')
        print('Running ML analysis...')
        analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir)
        analysis.train_models()
        print()

        # Plot results
        print('========================================================================')
        print('Plotting results...')
        plot = plot_results.PlotResults(self.config_file, self.output_dir)
        plot.plot_results()
        print('Done!')

####################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Hadronization Analysis')
    parser.add_argument('-c', '--config_file', 
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='config.yaml', )
    parser.add_argument('-i' ,'--input_file', 
                        help='Path to subjets_unshuffled.h5 file with ndarrays for ML input',
                        action='store', type=str,
                        default='', )
    parser.add_argument('-o', '--output_dir',
                        help='Output directory for output to be written to',
                        action='store', type=str,
                        default='./TestOutput', )
    parser.add_argument('--regenerate_graphs', 
                        help='construct graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    parser.add_argument('--use_precomputed_graphs', 
                        help='use graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    args = parser.parse_args()

    # If invalid config_file or input_file is given, exit
    if not os.path.exists(args.config_file):
        print(f'File {args.config_file} does not exist! Exiting!')
        sys.exit(0)
    if not os.path.exists(args.input_file):
        print(f'File {args.input_file} does not exist! Exiting!')
        sys.exit(0)

    # If output dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start_time = time.time()

    analysis = SteerAnalysis(input_file=args.input_file, 
                             config_file=args.config_file, 
                             output_dir=args.output_dir, 
                             regenerate_graphs=args.regenerate_graphs,
                             use_precomputed_graphs=args.use_precomputed_graphs)
    analysis.run_analysis()

    print('--- {} minutes ---'.format((time.time() - start_time)/60.))

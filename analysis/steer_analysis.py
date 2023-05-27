#! /usr/bin/env python
'''
Main script to steer graph construction and GNN training
'''

import argparse
import os
import sys
import yaml
import time

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
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, **kwargs):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs

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
        
        The graph_constructor module takes subjets_unshuffled.h5 and constructs a new file
        subjet_graphs.h5, which serves as the sole input to the ML analysis.
        '''

        # Check whether the graphs file has already been generated, and if not, generate it
        graph_file = os.path.join(self.output_dir, 'subjet_graphs.h5')
        print('========================================================================')
        if self.regenerate_graphs or not os.path.exists(graph_file):
            input_data = data_IO.read_data(self.input_file)
            graph_constructor.construct_graphs(input_data, self.config_file, self.output_dir)
        else:
            print(f'Graphs found: {graph_file}')

        # Perform ML analysis, and write results to file
        print('========================================================================')
        print('Running ML analysis...')
        analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir)
        graphs = data_IO.read_data(graph_file)
        analysis.run_analysis(graphs)
        print()

        # Plot results
        print('========================================================================')
        print('Plotting results...')
        plot_results.plot_results()
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
                             regenerate_graphs=args.regenerate_graphs)
    analysis.run_analysis()

    print('--- {} minutes ---'.format((time.time() - start_time)/60.))

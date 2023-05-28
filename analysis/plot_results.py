#!/usr/bin/env python3

"""
Plot classification performance
"""

import os
import sys
import argparse
import yaml
import pickle

# Data analysis and plotting
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Base class
sys.path.append('.')
from base import common_base

################################################################
class PlotResults(common_base.CommonBase):

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

        # Suffix for plot outputfile names
        self.roc_plot_index = 0
        self.significance_plot_index = 0
        self.auc_plot_index = 0

        self.plot_title = False
                
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)

        self.models = config['models']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def plot_results(self):
    
        # Load ML results from file
        roc_filename = os.path.join(self.output_dir, f'ROC.pkl')
        with open(roc_filename, 'rb') as f:
            self.roc_curve_dict = pickle.load(f)
            self.AUC = pickle.load(f)

        # Plot models for a single setting
        self.plot_models()

    #---------------------------------------------------------------
    # Plot several versions of ROC curves and significance improvement
    #---------------------------------------------------------------
    def plot_models(self):

        # Plot Subjet GNNs and Particle GNN
        roc_list = {}
        for model in self.roc_curve_dict.keys():
            if model == 'particle_gnn' or 'subjet_gnn' in model:
                roc_list[model] = self.roc_curve_dict[model]
        self.plot_roc_curves(roc_list)

    #--------------------------------------------------------------- 
    # Plot ROC curves
    #--------------------------------------------------------------- 
    def plot_roc_curves(self, roc_list):
    
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])
        plt.title('q vs. g', fontsize=14)
        plt.xlabel('False q Rate', fontsize=16)
        plt.ylabel('True q Rate', fontsize=16)
        plt.grid(True)
        
        for label,value in roc_list.items():
            if 'subjet_gnn' in label:
                linewidth = 2
                alpha = 0.9
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
            else:
                linewidth = 2
                linestyle = 'solid'
                alpha = 0.9
                color = sns.xkcd_rgb['almost black']
                legend_fontsize = 12
  
            FPR = value[0]
            TPR = value[1]
            plt.plot(FPR, TPR, linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
                    
        plt.legend(loc='lower right', fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'ROC_{self.roc_plot_index}.pdf'))
        plt.close()

        self.roc_plot_index += 1

    #---------------------------------------------------------------
    # Get color for a given label
    #---------------------------------------------------------------
    def color(self, label):

        color = None
          
        if label in ['subjet_gnn_laman_naive']:
            color = sns.xkcd_rgb['faded purple'] 
        elif label in ['subjet_gnn_laman_1N']:
            color = sns.xkcd_rgb['medium green']
        elif label in ['subjet_gnn_laman_1N2N']:
            color = sns.xkcd_rgb['watermelon']
        else:
            color = sns.xkcd_rgb['almost black']

        return color

    #---------------------------------------------------------------
    # Get linestyle for a given label
    #---------------------------------------------------------------
    def linestyle(self, label):
 
        linestyle = None
        if 'subjet_gnn' in label:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'

        return linestyle
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Plot ROC curves')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='config/qg.yaml',
                        help='Path of config file for analysis')
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir',
                        default='./TestOutput',
                        help='Output directory for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('ouputDir: \'{0}\"'.format(args.outputDir))

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    analysis = PlotResults(config_file=args.configFile, output_dir=args.outputDir)
    analysis.plot_results()
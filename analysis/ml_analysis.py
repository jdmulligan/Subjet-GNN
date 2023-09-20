#!/usr/bin/env python3

"""
Train GNNs to classify jets
"""

import os
import sys
import yaml
import pickle
from collections import defaultdict

import torch

sys.path.append('.')
from base import common_base
from analysis.models import gnn_pytorch

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
        if self.subjet_basis == 'exclusive': # For 'exclusive' we need to make sure we don't lose information so we need r=R
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
    
        self.AUC = defaultdict(list)
        self.roc_curve_dict = self.recursive_defaultdict()
        for model in self.models:
            print()
            print(f'------------- Training model: {model} -------------')
            model_settings = self.model_settings[model]
            model_info = {'model': model,
                          'model_settings': model_settings,
                          'n_total': self.n_total,
                          'n_train': self.n_train,
                          'torch_device': self.torch_device,
                          'output_dir': self.output_dir}

            # ---------- Input: Particle four-vectors ----------
            if model in ['particle_gcn', 'particle_gat']:
                for graph_type in model_settings['graph_types']:
                    model_key = f'{model}__{graph_type}'
                    print(f'model_key: {model_key}')
                    print()
                    model_info_temp = model_info.copy()
                    model_info_temp['graph_type'] = graph_type
                    model_info_temp['model_key'] = model_key
                    self.AUC[model_key], self.roc_curve_dict[model_key] = gnn_pytorch.GNN_PyTorch(model_info_temp).train()

            # ---------- Input: Subjet four-vectors ----------
            if model in ['subjet_gcn', 'subjet_gat']:
                for r in self.r_list:
                    for n_subjets_total in subjet_graphs_dict['n_subjets_total']:
                        for graph_type in model_settings['graph_types']:
                            model_key = f'{model}__{graph_type}__r{r}__n{n_subjets_total}'
                            print(f'model_key: {model_key}')
                            print()
                            model_info_temp = model_info.copy()
                            model_info_temp['graph_type'] = graph_type
                            model_info_temp['r'] = r
                            model_info_temp['n_subjets_total'] = n_subjets_total
                            model_info_temp['subjet_graphs_dict'] = subjet_graphs_dict
                            model_info_temp['model_key'] = model_key
                            self.AUC[model_key], self.roc_curve_dict[model_key] = gnn_pytorch.GNN_PyTorch(model_info_temp).train()

            # TODO: GNNs (jax)

            # TODO: PFN (tensorflow)

            # TODO: PFN (pytorch)

            # TODO: ParticleNetLite

            # ---------- Write ROC curve dict to file ----------
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)
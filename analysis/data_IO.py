#!/usr/bin/env python
'''
This module contains functions to read and write data structures relevant for the analysis
'''

import os
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict
from collections import defaultdict

#---------------------------------------------------------------
# Write nested dictionary of ndarray to hdf5 file
# Note: all keys should be strings
#---------------------------------------------------------------
def write_data(results, output_dir, filename = 'results.h5'):
    print()

    print(f'Writing results to {output_dir}/{filename}...')
    dicttoh5(results, os.path.join(output_dir, filename), overwrite_data=True)

    print('done.')
    print()

#---------------------------------------------------------------
# Read dictionary of ndarrays from hdf5
# Note: all keys should be strings
#---------------------------------------------------------------
def read_data(input_file):
    print()
    print(f'Loading results from {input_file}...')

    results = h5todict(input_file)

    print('done.')
    print()

    return results
    
#---------------------------------------------------------------
# Create a nested defaultdict
#---------------------------------------------------------------
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

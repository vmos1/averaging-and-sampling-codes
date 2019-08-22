# Code to study Autocorrelation
## Computes autocorrelation time and block size for max_error
## Generates plots for data, autocorrelation and error variation with block size


### Needs functions from mean_corr_functions in  '/Users/vpa/Tools/my_custom_modules/mean_and_sampling/'


import numpy as np
import sys
import os
import subprocess
import argparse 
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/vpa/Tools/my_custom_modules/mean_and_sampling/')
from mean_corr_functions import *

parser = argparse.ArgumentParser(description="Code to study autocorrelation analysis. Computes autocorrelation time and block size for max error", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# This extra option formatter... is to include the default values in the output of help.

parser.add_argument('--fname','-f', required=True, type=str, help='File name')
parser.add_argument('--column',   type=int,   default=0,     help='Column number of file for multi-column')
parser.add_argument('--plot','-v',   action='store_true',   help='Plots for autocorrelation and block errors')
# Note, None is needed in end. If you use -1, you lose the last element of the array.

parg = parser.parse_args(sys.argv[1:])

fname,plot,col=parg.fname,parg.plot,parg.column

a1=np.loadtxt(fname,dtype=np.float64)
if len(a1.shape)>1 :        a1=a1[:,col] # For multi-column files 

auto=f_autocorr(a1)
blk_errors=f_block_errs(a1)
print "Auto-correlation time",f_find_autocorr(a1)
print "Max-block",f_max_block(a1)
print "\n"

if plot: 
    # Simple plot of variation of data
    plt.figure()
    plt.plot(a1,color='magenta')
    plt.title('Data plot')
    plt.xlabel('Simulation time')
    plt.ylabel('O')
    plt.show()

    # Autocorrelation plot
    plt.figure()
    plt.plot(auto,color='r')
    plt.title('Autocorrelation')
    plt.xlabel('Simulation time')
    plt.show()

    # Plot of variation of error with block size
    plt.figure()
    plt.plot(blk_errors,'b')
    plt.title('Error variation with block size')
    plt.xlabel('Block size')
    plt.show()

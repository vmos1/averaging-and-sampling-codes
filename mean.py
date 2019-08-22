#!/anaconda/bin/python
# coding: utf-8

# # This code gives the mean and error for a set of numbers

import numpy as np
import sys
import os
import subprocess
import argparse 

# Remember to change this path when hierarchy changes.
function_file_path= '/Users/vpa/Tools/my_custom_modules/mean_and_sampling/'
sys.path.insert(0, function_file_path)
from mean_corr_functions import *

parser = argparse.ArgumentParser(description="Compute the mean and error for a set of numbers. Also computes autocorrelation time and max block", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# This extra option formatter... is to include the default values in the output of help.

parser.add_argument('--fname','-f', required=True, type=str, help='File name')
parser.add_argument('--start',   type=int,   default=0,     help='Start point of input array.')
parser.add_argument('--block','-b', type=int,   default=0,     help='Block size.')
parser.add_argument('--end',   type=int,   default=None,     help='End point of input array')
parser.add_argument('--column',   type=int,   default=-1,     help='Column number of file')
parser.add_argument('--concise','-c',   action='store_true',    help='Hide details of mean calculation')
parser.add_argument('--verbose','-v',   action='store_true',    help='Show extra details of max block and autocorr calculation')
# Note, None is needed in end. If you use -1, you lose the last element of the array.

parg = parser.parse_args(sys.argv[1:])

start,end,blk,col=parg.start,parg.end,parg.block,parg.column
concise,verbose=parg.concise,parg.verbose
fname=parg.fname
if not concise: print fname

a1=np.loadtxt(fname,dtype=np.float64)
f_mean(a1,start,end,blk,col,verbose=verbose,concise=concise)



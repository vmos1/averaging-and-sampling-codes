#!/anaconda/bin/python
# This code gives the mean and error for the ratio of two observables.

import numpy as np
import sys
import os
import subprocess
import argparse 

# Remember to change this path when hierarchy changes.
function_file_path= '/Users/vpa/Tools/my_custom_modules/mean_and_sampling/'
sys.path.insert(0, function_file_path)
from mean_corr_functions import *

parser = argparse.ArgumentParser(description="Compute the mean and error for the ratio of two observables. Also computes autocorrelation time and max block", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# This extra option formatter... is to include the default values in the output of help.

parser.add_argument('--fname','-f', required=True, type=str, help='File name')
parser.add_argument('--start',   type=int,   default=0,     help='Start point of input array.')
parser.add_argument('--block', '-b',  type=int,   default=0,     help='Block size.')
parser.add_argument('--end',   type=int,   default=None,     help='End point of input array')
parser.add_argument('--column1','-c1',   type=int,   default=0,     help='Column number of array 1')
parser.add_argument('--column2','-c2',   type=int,   default=-1,     help='Column number of array 2')
parser.add_argument('--method',   type=str, choices=['bootstrap','jacknife'],  default='bootstrap',  help='Type of resampling')
parser.add_argument('--concise','-c',   action='store_true',    help='Hide details of mean calculation')
parser.add_argument('--verbose','-v',   action='store_true',    help='Show extra details of max block and autocorr calculation')

# Note, None is needed in end. If you use -1, you lose the last element of the array.

parg = parser.parse_args(sys.argv[1:])

start,end,blk,c1,c2,method=parg.start,parg.end,parg.block,parg.column1,parg.column2,parg.method
concise,verbose=parg.concise,parg.verbose
fname=parg.fname
if not concise: print fname

a=np.loadtxt(fname,dtype=np.float64)
assert(a.shape[1]>1),"File contains just one column"
a1,a2=a[:,c1],a[:,c2]
f_corr(a1,a2,c1,c2,start,end,blk,method,verbose,concise=concise)



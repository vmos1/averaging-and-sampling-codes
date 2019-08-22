
# coding: utf-8

# # This code stores important functions for mean and corr.

# In[1]:

import numpy as np
import sys
import os
import subprocess


# In[2]:

## Functions for writing in error form: value(error)

def frexp10(p):
    '''
    Function to express a positive floating point number as :   y=a 10^b
    log_10 y = log_10 (a) + b = x = dec + b
    for y>1, b is +ve, so integer part of x (=b) is power and 10**dec is a
    for y<0, modification required. b = b-1, dec = 1 + dec
    
    '''
    a=np.log10(p) # get log to the base 10
    b=np.modf(a) # split a float into integer part and rest
    
    f1,fpow=b[0],b[1]
    
    if p < 1.0 :
        f1=1.0+f1
        fpow=fpow-1
    fcoeff=10**f1
    return fcoeff,fpow

def f_value_error(mean,err):
    '''
    function that reads two numbers : value and error and gives a string of the form
    value(error) x 10^{exponent} 
    '''
    
    if np.isclose(err,0.0): # Exception for infinite precision
        return str(mean)
    
    a=frexp10(err)
    b=np.rint(mean/(10**a[1]))
    c=np.around(a[0])

    ans=[str(int(b))+'('+str(int(c))+')'+'x 10^{'+str(int(a[1]))+"}"]
    return ans[0]


# ## Formula for standard error
# $$ \sigma_{err} = \left\{ \sum_{i=1}^N  \frac{{(x_i-\overline{x})}^2}{(N-1)}\right\}/\sqrt(N)$$

# # Notes on blocking and resampling
# 
# ### Autocorrelation
# - If data is correlated, you can do 2 things : Autocorrelation analysis or find optimal block size.
# - Autocorrelation analysis gives you the autocorrelation time. But it needs a long string of data points.
# - Optimal block size can be found by trying different blocks to see where the error saturates.
# - Either way, you get a block size that you need to use to block your data.
# - After that you can perform other analysis.
# 
# ### Resampling:
# - After obtaining independent configurations by blocking your data, you can do resampling if you want to compute errors of derived quantities.
# - Bootstrap and jacknife assume your data in independent (unless you do then with a bin size).
# - The error calculation from them is tricky : Use the right formula.

# ## Formula for blocking without discard.
# - If you discard, the mean is not the same. 
# - If you don't discard, you need to correct the element in the last bin to get the right answer.
# - The correct formula for X is given below
# 
# $ \frac{\left(\frac{a_1+a_2+a_3}{3}\right) +\left(\frac{a_4+a_5+a_6}{3}\right) + X }{(2+1)} = \left(\frac{a_1+a_2+a_3 +a_4 + a_5 + a_6 +a_7 +a_8}{8}\right) $
# $ \implies $ 
# 
# $ X = ( a_1+a_2+a_3 +a_4 + a_5 + a_6)(\frac{3*(2+1)}{8}-1) +\frac{(2+1)}{8}(a_7 +a_8)$
# 
# 
# $ X=  \frac{a1+a2+a3+a4+a5+a6}{blksize} \times \left(\frac{(numblks+1)*blksize}{sz} -1 \right)+ (a7+a8)*\frac{(numblks+1)}{sz} $
# 

# In[3]:

# Block and sample mean functions

def f_block(a, blk_size=1,discard=True):
    '''
    Blocks the data given the array and block size and returns the blocked array.
    (Data is averaged inside the block). 
    discard=True by default.
    To get the the same without discard, you need to correct for the element in the last bin.
    '''
    assert a.ndim==1,"Array is not one-dimensional %s"%(a.shape,)
    sz=a.shape[0]
    assert sz>0, 'input list is empty'
    
    #Discarding last few numbers if block size is not a perfect divisor of total size.
    num_blks=int(sz/blk_size) # number of blocks if perfect divisor
    len_divisor=num_blks*blk_size # total size of new array excluding edge elements.
    
    a2=a[:len_divisor] # Array elements excluding edge elements if block size is not a perfect divisor
    a_edge=a[len_divisor:] # excluded edge elements.

    a3=a2.reshape(num_blks,blk_size)

    if (discard or sz%blk_size==0) :
        a_block=np.mean(a3,axis=1)
    else :
        #Computing the element in the last bin using a correction formula given in the description.
        # This is done to ensure the mean remains the same.
        a_end=(np.sum(a2)/(1.0*blk_size))*((num_blks+1)*blk_size*(1.0/sz) -1)+ np.sum(a_edge)*(num_blks+1)*1.0/sz
        # Can also compute the above expression directly using the actual answer.
#         a_end=np.mean(a)*(num_blks+1)-np.sum(a2)/(1.0*blk_size)
        a_block=np.append(np.mean(a3,axis=1),a_end)    

    return a_block




def f_sample_mean_err(a):
    '''
    Simple function to compute sample mean and standard error
    '''
    assert a.ndim==1,"Array is not one-dimensional %s"%(a.shape,)
    return np.mean(a),np.std(a,ddof=1)/np.sqrt(len(a)),len(a)


# In[4]:

# Functions to fix autocorrelation : Autocorrelation time and block size with max error.

def f_autocorr(a):
    '''
    General function to compute autocorrelation of an array.
    '''
    N=len(a)
    tmax=min(N/1,50)
    a_corr,acorr_error=np.ones(tmax),np.ones(tmax)
    O2_mean=np.sum(a**2)/N
    O_mean=(np.sum(a)/N)
    Dr=O2_mean-O_mean**2 # we simplify the expression given above to get O^2_mean-O_mean_square
    
    for i in range(1,tmax): # First auto corr element is 1 by definition, so we start from 2.
        a_corr[i]=np.mean((a[:-i]-O_mean)*(a[i:]-O_mean))/Dr 
    return a_corr

def f_find_autocorr(a,verbose=False):
    ''' Function that takes in the array, computes autocorrelation and gives the autocorrelation 
    time : The time to reach 0.1
    '''
    b1=f_autocorr(a)
    for i,j in enumerate(b1):
        if j<0.1:
            if verbose: print "Autocorelation array",b1
            return(i)
    
    print "Autocorrelation too large"
    
    return len(a)-1


def f_block_errs(a):
    '''
    Function that computes errors for different block sizes. Useful in autocorrelation plots.
    '''    
    assert a.ndim==1,"Array is not one-dimensional %s"%(a.shape)
    bmax=a.shape[0]/10 # Atleast 10 data points after blocking.
    err_list=np.ones(bmax)

    for i in np.arange(bmax):
        mean,err_list[i],N=f_sample_mean_err(f_block(a,i+1))#Plus one to correct for np array -> 0..N-1,but block size->1..N
    
    return err_list


def f_max_block(a,verbose=False,check_block=3):
    '''
    Function that computes the rough autocorrelation time for an array.
    It tries different size of blocks and picks the one showing a local maximum in error.
    if verbose, it prints the list with errors.
    check_block is the bin size within which we check if error is minimum
    '''
    # for example, if check_block =3, we compare with the last 3 elements if the error is the minimum.
    
    assert a.ndim==1,"Array is not one-dimensional %s"%(a.shape)
    bmax=a.shape[0]/10 # Atleast 10 data points after blocking.
    err_list=np.ones(bmax)

    for i in np.arange(bmax):
        mean,err_list[i],N=f_sample_mean_err(f_block(a,i+1))#Plus one to correct for np array -> 0..N-1,but block size->1..N
        if i >= check_block: # ignoring first few blocks until we can check. Need a few block calcs to compare.
            prev_list=err_list[(i-check_block):(i+1)]

            if np.min(prev_list)==err_list[i] :
                blk_max=np.argmax(prev_list)+(i-check_block) + 1 # Plus 1 because of plus one above. 
                
                if verbose: print "Blocking errors array", err_list[:i+1]
                return blk_max
    
    print "Max-block too large"
    return bmax



# # Functions for resampling

# In[5]:

def f_btstrap(a1,a2,n_btstrap=1000):    
    '''
    Generic code to compute the bootstrap of the ratio of two numpy arrays a1 and a2.
    The approach is : 
    - Make sample of a1 
    - Use corresponding values of a2.
    - Compute the mean and then take ratio of the means.
    - Repeat this a certain number of times.
    - Use these stored numbers as a sample and perform statistics.
    - Error is the Standard Deviation, no division by sqrt(N) (IMPORTANT)
    - Default number of bootstraps is 1000
    
    '''
    assert len(a1)==len(a2),"Array sizes are different %d, %d"%(len(a1),len(a2))
    n_smples=len(a1) # number of samples
    p=np.zeros(shape=(n_btstrap),dtype=np.float64)
    for j in range(n_btstrap):
        smple=np.random.randint(0,n_smples,n_smples)
        b1=np.mean(a1[smple])
        b2=np.mean(a2[smple])
        if b2==0:
            print "Encountered a zero"
            b2=1e-15
        p[j]=b1/b2
    p_bar=np.sum(p)/len(p) # average
    p_square_bar=np.sum(np.square(p))/n_btstrap # mean square value ( summmation x^2 /N )
    return p_bar,np.sqrt(p_square_bar-p_bar**2),n_smples    



# Jack-knife sampling formula:
# - Mean
# $$ \overline{x_i} = \frac{1}{n-1} \sum_{j\neq i}^{n} x_j $$
# - Variance
# \begin{eqnarray}
# Var &&= \frac{n-1}{n} \sum_{i=1}^n {\left( \overline{x_i} - \overline{x}\right) }^2 \nonumber \\
#  &&= (n-1) \sum_{i=1}^n {\left( \frac{\overline{x_i}^2}{n} - \overline{x}^2\right) }
# \end{eqnarray}
# where $ \overline{x} $ is the mean of all $ \overline{x_i}$
# 

# In[6]:

def f_jknife(a1,a2):    
    '''
    Generic code to compute the jacknife of the ratio of two numpy arrays a1 and a2.
    The approach is : 
    - Remove ith element of a1 
    - Use corresponding values of a2.
    - Compute the mean and then take ratio of the means.
    - Repeat this for all elements i.
    - Use these stored numbers as a sample and perform statistics.
    - Error is like Standard Deviation, no division by sqrt(N) (IMPORTANT). Formula given above.
    '''
    assert len(a1)==len(a2),"Array sizes are different %d, %d"%(len(a1),len(a2))
    n_smples=len(a1) # number of samples
    n_jknife=n_smples-1 # number of jack-knife tries.
    p=np.zeros(shape=(n_smples),dtype=np.float64)
    for j in range(n_smples):
        smple=np.delete(np.arange(n_smples),j)
        b1=np.mean(a1[smple])
        b2=np.mean(a2[smple])
        if b2==0:
            print "Encountered a zero"
            b2=1e-15
        p[j]=b1/b2
    p_bar=np.sum(p)/len(p) # average
    p_square_bar=np.sum(np.square(p))/n_smples # mean square value ( summmation x^2 /N )
    return p_bar,np.sqrt((n_smples-1)*(p_square_bar-p_bar**2)),n_smples   


# In[ ]:




# In[17]:

def f_mean(a1,start=0,end=None,blk=0,col=-1,verbose=False,concise=False):
    '''
    Function to compute the mean for a column of a given file.  
    If no default block size given, it uses autocorrelation time. if autocorrelation time is large, it uses max block.
    '''
    # For multi-column files, pick correct column
    a2=a1[:,col] if len(a1.shape)!=1 else a1 
    a3=a2[start:end]
    #Find max block and autocorr to compare
    b_max=f_max_block(a3,verbose=verbose)
    autocorr_time=f_find_autocorr(a3,verbose=verbose)

    # Blocking
    if blk==0: 
        blk=autocorr_time if autocorr_time < 100 else b_max

    a4=f_block(a3,blk) # Array used for mean
    mean,stderr,N=f_sample_mean_err(a4)
    
    #Print values in form
    
    if not concise:
        print 'start {}, block size {} column {}, end {}'.format(start,blk,col,end)
        print "maxblock\t",b_max,",\tAutocorr\t",autocorr_time
        print mean,',+/-,',stderr,',\tData pts',N,',\tblk =',blk
    print f_value_error(mean,stderr)

    
    

def f_corr(a1,a2,c1,c2,start=0,end=None,blk=0,mthd='bootstrap',verbose=False,concise=False):
    '''
    Function to compute the mean of the ratio of two columns of a given file.  
    If no default block size given, it uses the largest autocorrelation time. 
    If autocorrelation time is large, it uses max block.
    '''
    # For multi-column files, pick correct column
    a3,a4=a1[start:end],a2[start:end]
    
    #Find max block and autocorr to compare    
    b1_max=f_max_block(a3,verbose=verbose)
    autocorr_time1=f_find_autocorr(a3,verbose=verbose)
    b2_max=f_max_block(a4,verbose=verbose)
    autocorr_time2=f_find_autocorr(a4,verbose=verbose)
    
    #Blocking
    
    if blk==0: 
        autocorr_time=max(autocorr_time1,autocorr_time2)
        b_max=max(b1_max,b2_max)
        blk=autocorr_time if autocorr_time < 100 else b_max
    
    a5=f_block(a3,blk)
    a6=f_block(a4,blk)

    if mthd=='bootstrap':
        mean,stderr,N=f_btstrap(a5,a6) 
    elif mthd=='jacknife':
        mean,stderr,N=f_jknife(a5,a6)
    else :
        print "Invalid option",mthd
        raise SystemExit
        
        
    if not concise:
        print "Using",mthd
        print """start {},\t block size {},\t columns {},{},\t end {},\nAutocorrs = {},{};\tMax blocks {},{};\nMean {},'+/-',stderr {}, \tData pts {}            """.format(start,blk,c1,c2,end,autocorr_time1,autocorr_time2,b1_max,b2_max,mean,stderr,N)
    print f_value_error(mean,stderr)


# In[18]:

#Test functions
data_dir='/Users/vpa/Tools/my_custom_modules/mean_and_sampling/test_data_files/'
if __name__=='__main__':
    
    fname=data_dir+'1col_file.txt'
    start,blk,col,end=0,0,0,None
    a1=np.loadtxt(fname,dtype=np.float64)
    print fname
    f_mean(a1,start,end,blk,col,verbose=True)    


if __name__=='__main__':
    
    fname=data_dir+'mult_col_file.txt'
    print fname
    start,blk,c1,c2,end,mthd=0,0,-1,-2,None,'Bootstrap'
    a=np.loadtxt(fname,dtype=np.float64)
    assert(a.shape[1]>1),"File contains just one column"
    a1,a2=a[:,c1],a[:,c2]
    
    f_corr(a1,a2,c1,c2,start=0,end=None,blk=blk,mthd='bootstrap',verbose=False,concise=False)


# In[ ]:




# In[ ]:




# In[ ]:




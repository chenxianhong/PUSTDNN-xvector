#!/usr/bin/env python

########################################################################################
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot, Frantisek Grezl  #
#                         Brno Universioty of Technology                               #
#                         Faculty of information technology                            #
#                         Department of Computer Graphics and Multimedia               #
#  email             : {isilnova,matejkap,iplchot,grezl}@vut.cz                        #
########################################################################################
#                                                                                      #
#  This software and provided models can be used freely for research                   #
#  and educational purposes. For any other use, please contact BUT                     #
#  and / or LDC representatives.                                                       #
#                                                                                      #
########################################################################################

import sys, os, logging
import numpy as np
import scipy.io.wavfile as wav
import h5py

sys.path.insert(0, 'local/phoneme/')
import utils
import nn_def

sys.path.insert(0, 'local/tf/')
import kaldi_io

logging.basicConfig( format= '%(message)s',level=logging.INFO)


def load_keys(kaldi_trials):
    """Loads test data keys

    Parameters
    ----------
    trial_key_file : string
        Path to trial_key files (e.g. ivec14_sre_trial_key_release.tsv)

    Returns
    -------
    key : array
        a vector of 1(target) or 0(nontarget)

    """
    file_open = open(kaldi_trials)
    key = []
    for line in file_open.readlines():
        line=line.strip()
        if line.split(' ')[2] == 'target':
            key.append(1)
        else:
            key.append(0)

    file_open.close()
    

    #convert list  np.array, a vector
    key = np.array(key)
    return key

def load_scores(kaldi_scores):
    """Loads test data keys

    Parameters
    ----------
    trial_key_file : string
        Path to trial_key files (e.g. ivec14_sre_trial_key_release.tsv)

    Returns
    -------
    key : array
        a vector of 1(target) or 0(nontarget)

    """
    file_open = open(kaldi_scores)
    score = []
    for line in file_open.readlines():
        line=line.strip()
        score.append(float(line.split(' ')[2]))

        
    file_open.close()
    
    score = np.array(score)
    return score

    
def compute_eer(scores, labels):
    scores_indexes = np.argsort(scores)
    labels_sort=[]
    labels_sort_inver = []
    for i in scores_indexes:
        labels_sort.append(labels[i])
        if labels[i] == 1:
            labels_sort_inver.append(0)
        else:
            labels_sort_inver.append(1)
        
    labels_sort = np.array(labels_sort)
    labels_sort_inver = np.array(labels_sort_inver)
    
    FN = labels_sort.cumsum()/sum(labels_sort)
    TN = labels_sort_inver.cumsum()/sum(labels_sort_inver)
    FP = 1 - TN
    TP = 1 - FN
    
    difs = FN - FP
    for i in range(0,len(difs)-1):
        if difs[i+1] >= 0 and difs[i] < 0:
            idx1 = i
            idx2 = i+1
 
    x = [FN[idx1], FP[idx1]]
    y = [FN[idx2], FP[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = 100 * ( x[0] + a * ( y[0] - x[0] ) )
    
    
    DCF14 = FN + 100 * FP  # SRE-2014 performance parameters
    dcf14 = min(DCF14)

    Cmiss = 10; Cfa = 1; P_tgt = 0.01 # SRE-2008 performance parameters
    Cdet  = Cmiss * FN * P_tgt + Cfa * FP * ( 1 - P_tgt)
    # Cdefault = min(Cmiss * P_tgt, Cfa * ( 1 - P_tgt))
    dcf08 = 100 * min(Cdet) # note this is not percent


    Cmiss = 1; Cfa = 1; P_tgt = 0.001  # SRE-2010 performance parameters
    Cdet  = Cmiss * FN * P_tgt + Cfa * FP * ( 1 - P_tgt)
    #Cdefault = min(Cmiss * P_tgt, Cfa * ( 1 - P_tgt))
    dcf10 = 100 * min(Cdet)  # note this is not percent
    
    return eer, dcf08, dcf10, dcf14


    
    
if len(sys.argv)==3:
    trial_file, score_file=sys.argv[1:3]
else:
    print("Wrong number of input arguments. 2 are expected")
    sys.exit()

keys = load_keys(trial_file)
scores= load_scores(score_file)

    
eer, dcf08, dcf10, dcf14 = compute_eer(scores, keys)

print("eer: %f, dcf08: %f, dcf10: %f, dcf14: %f" %(eer, dcf08, dcf10, dcf14))
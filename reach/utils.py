"""Miscellaneous function module for reach
"""
from __future__ import division, print_function
import csv
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Utilities Functions
# -----------------------------------------------------------------------------
def summarise_sequences():
    """
    """
    # Read in each sequence
    bright_list_files = ["p99_bright.txt", "p101_bright.txt"]
    faint_list_files = ["p99_faint.txt", "p101_faint.txt"]
    period = [99, 101]

    target_list = []

    for p_i, bright_list_file in enumerate(bright_list_files):
        with open(bright_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "bright"))

    for p_i, faint_list_file in enumerate(faint_list_files):
        with open(faint_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "faint"))
        
    # Order each sequence
    sequences = OrderedDict()
    
    for tgt_i in xrange(0, len(target_list), 4):
        # All targets must share a sequence and period
        assert (target_list[tgt_i][::2] == target_list[tgt_i+1][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+2][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+3][::2])
        
        sequences[target_list[tgt_i]] = [target_list[tgt_i+1][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+2][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+3][1]]
    
    pkl_sequences = open("data/sequences.pkl", "wb")
    pickle.dump(sequences, pkl_sequences)
    pkl_sequences.close()
    
    return sequences
    
def load_target_information(filepath="data/target_info.tsv"):
    """Loads in the target information tsv (tab separated) as a pandas 
    dataframne with appropriate column labels and rows labelled as each star.
    
    https://pandas.pydata.org/pandas-docs/version/0.21/generated/
    pandas.read_csv.html#pandas.read_csv
    """
    # Import (TODO: specify dtypes)
    tgt_info = pd.read_csv(filepath, sep="\t", header=1, index_col=8, 
                              skiprows=0)
    
    # Organise dataframe by removing duplicates
    # Note that the tilde is a bitwise not operation on the mask
    tgt_info = tgt_info[~tgt_info.index.duplicated(keep="first")]
    
    # Force primary and Bayer IDs to standard no-space format
    tgt_info["Primary"] = [id.replace(" ", "").replace(".", "").replace("_","")
                           for id in tgt_info["Primary"]]
                           
    tgt_info["Bayer_ID"] = [id.replace(" ", "").replace("_","") 
                            if type(id)==str else None
                            for id in tgt_info["Bayer_ID"]]
                            
    # Use None as empty value for IDs
    # Note: this is possibly unnecessary since dataframes have a notnull method
    tgt_info["Ref_ID_1"].where(tgt_info["Ref_ID_1"].notnull(), None, 
                               inplace=True)
    tgt_info["Ref_ID_2"].where(tgt_info["Ref_ID_2"].notnull(), None, 
                               inplace=True)
    tgt_info["Ref_ID_3"].where(tgt_info["Ref_ID_3"].notnull(), None, 
                               inplace=True)
    # Return result
    return tgt_info
    
    
def combine_independent_boostrap_runs(pkl_list):
    """Function to combine LDD realisations from independent bootstrapping runs
    for the purpose of plotting histograms/estimating uncertainties.
    
    Parameters
    ----------
    pkl_list: list
        List of pickle files to combine.
        
    Returns
    -------
    n_ldd_fit_all: dict
        Dictionary of all LDD fits from each bootstrapping run. Key is the 
        science target, values stored in list.
    """
    n_ldd_fit_all = {}
    
    # Open each pickle and join together into n_ldd_fit_all
    for pkl_fn in pkl_list:
        pkl = open(pkl_fn)
        [n_vis2, n_baselines, n_ldd_fit, wavelengths] = pickle.load(pkl)
        pkl.close()
        
        for sci in n_ldd_fit.keys():
            if sci in n_ldd_fit_all.keys():
                n_ldd_fit_all[sci].extend(n_ldd_fit[sci])
            else:
                n_ldd_fit_all[sci] = n_ldd_fit[sci]
                
    return n_ldd_fit_all

def complete_obs_diagnostics(complete_sequences):
    """
    """
    for seq in complete_sequences.keys():
        print("-"*30)
        print(seq,len(complete_sequences[seq][2]))
        for i, yy in enumerate(complete_sequences[seq][2]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
            
def night_log_diagnostics(night_log):
    """
    """
    for night in night_log.keys():
        print("-"*30)
        print(night, len(night_log[night]))
        for i, yy in enumerate(night_log[night]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
    
            
def get_unique_key(tgt_info, id_list):
    """
    """
    unique_ids = []
    
    # Grab the primary IDs
    # Note that several stars are observed multiple times under different
    # primary IDs, so we need to check HD and Bayer IDs as well
    for star in id_list:
        # Remove non-alpha-numeric characters 
        star = star.replace("_", "").replace(" ", "").replace(".", "")
        
        prim_id = tgt_info[tgt_info["Primary"]==star].index
        
        if len(prim_id)==0:
            prim_id = tgt_info[tgt_info["Bayer_ID"]==star].index
            
        if len(prim_id)==0:
            prim_id = tgt_info[tgt_info.index==star].index
        
        try:
            assert len(prim_id) > 0
        except:
            print("...failed on %s, %s" % (star))
            failed = True
            break
        unique_ids.append(prim_id[0])
        
    return unique_ids
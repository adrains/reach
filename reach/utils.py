"""Miscellaneous function module for reach
"""
from __future__ import division, print_function

import csv
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

# -----------------------------------------------------------------------------
# Utilities Functions
# -----------------------------------------------------------------------------
def summarise_sequences():
    """Creates a dictionary summary of each sequence, specified with the unique
    key (period, science, bright/faint).
    
    Returns
    -------
    sequences: OrderedDict
        Dict mapping key (period, science, bright/faint) with list of target 
        IDs in the sequence.
    """
    # Read in each sequence
    bright_list_files = ["data/p99_bright.txt", "data/p101_bright.txt", 
                         "data/p102_bright.txt"]
    faint_list_files = ["data/p99_faint.txt", "data/p101_faint.txt",
                        "data/p102_faint.txt"]
    period = [99, 101, 102]

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
    """Function to combine results from independent bootstrapping runs for the
    purpose of plotting histograms/estimating uncertainties.
    
    Parameters
    ----------
    pkl_list: list
        List of pickle files to combine.
        
    Returns
    -------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    """
    # Initialise structure to store results in
    bs_result_list = []   
    
    # Open each pickle and join together into n_ldd_fit_all
    for pkl_fn in pkl_list:
        pkl = open(pkl_fn)
        bs_result_list.append(pickle.load(pkl))
        pkl.close()
        
    # Get a reference to the dataframe we want to join to    
    all_bs_results = bs_result_list[0]     
        
    # For every science target, combine all bootstrapping iterations
    for bs_result_n in bs_result_list[1:]:
        for sci in bs_result_n.keys():
            # Increment the index of the data to be joined
            orig_n_bs = len(bs_result_n[sci])
            base_n = len(all_bs_results[sci])
            bs_result_n[sci].set_index(np.arange(base_n, base_n + orig_n_bs),
                                       inplace=True)
            
            # Join
            all_bs_results[sci] = pd.concat([all_bs_results[sci], 
                                             bs_result_n[sci]])
                
    return all_bs_results
    

def complete_obs_diagnostics(complete_sequences):
    """Prints complete_complete sequences in a human readable format for 
    troubleshooting.
    """
    for seq in complete_sequences.keys():
        print("-"*30)
        print(seq,len(complete_sequences[seq][2]))
        for i, yy in enumerate(complete_sequences[seq][2]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
          
            
def night_log_diagnostics(night_log):
    """Prints the night log in a human readable format for troubleshooting.
    """
    for night in night_log.keys():
        print("-"*30)
        print(night, len(night_log[night]))
        for i, yy in enumerate(night_log[night]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
    
            
def get_unique_key(tgt_info, id_list):
    """Some stars were observed multiple times under different names (e.g. a
    Bayer designation once, and a HD number another time). This complicates
    uniquely IDing each star, so this method serves to take an ID that we may
    have referenced a star with, and take the value that allows us to easily
    reference tgt_info.
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    id_list: list
        List of string IDs.
    
    Returns
    -------
    unique_ids: list
        List of IDs.
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
    
    
def compute_dist(tgt_info):
    """Calculate distances and distance errors for both stars with Gaia and HIP
    parallaxes
    """
    # Compute distance
    tgt_info["Dist"] = 1000 / tgt_info["Plx"]

    tgt_info["Dist"].where(~np.isnan(tgt_info["Dist"]), 
                       1000/tgt_info["Plx_alt"][np.isnan(tgt_info["Dist"])],
                       inplace=True)
    
    # Compute distance error
    # e_dist = |D*-1*e_plx / plx|                   
    tgt_info["e_Dist"] = np.abs(tgt_info["Dist"] * tgt_info["e_Plx"] 
                                / tgt_info["Plx"])
    tgt_info["e_Dist"].where(~np.isnan(tgt_info["e_Dist"]),
                        np.abs(tgt_info["Dist"] * tgt_info["e_Plx"] 
                               / tgt_info["Plx"]))
    
    
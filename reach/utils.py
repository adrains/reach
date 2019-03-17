"""Miscellaneous function module for reach
"""
from __future__ import division, print_function

import csv
import pickle
import numpy as np
import pandas as pd
import reach.photometry as rphot
import reach.diameters as rdiam
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
    
def initialise_tgt_info():
    """
    """
    # Import the base target info sans calculations
    tgt_info = load_target_information()

    # Calculate distances and distance errors
    compute_dist(tgt_info)

    # -----------------------------------------------------------------------------
    # (2) Convert Tycho magnitudes to Johnson-Cousins magnitudes
    # -----------------------------------------------------------------------------
    # Convert Tycho V to Johnson system using Bessell 2000

    # For simplification during testing, remove any stars that fall outside the 
    # VT --> V conversion from Bessell 2000
    tgt_info = tgt_info.drop(["GJ551","HD133869"])

    # Convert VT and BT to V and B
    # TODO: proper treatment of magnitude errors
    Bmag, Vmag = rphot.convert_vtbt_to_vb(tgt_info["BTmag"], tgt_info["VTmag"])

    tgt_info["Bmag"] = Bmag   
    tgt_info["e_Bmag"] = tgt_info["e_BTmag"]

    tgt_info["Vmag"] = Vmag   
    tgt_info["e_Vmag"] = tgt_info["e_VTmag"]

    # -----------------------------------------------------------------------------
    # (3) Correct photometry for interstellar extinction
    # -----------------------------------------------------------------------------
    # These are the filter effective wavelengths *not* considering the effect of 
    # spectral type (Angstroms)
    filter_eff_lambda = {"B":4450, "V":5510, "J":12200, "H":16300, "K":21900, 
                         "W1":34000, "W2":46000, "W3":120000, "W4":220000}
                     
    filter_eff_lambda = np.array([4450., 5510., 12200., 16300., 21900., 34000., 
                                  46000., 120000., 220000.])

    # Import/create the SpT vs B-V grid
    grid = rphot.create_spt_uv_grid()
                     
    # Calculate selective extinction (i.e. (B-V) colour excess)
    tgt_info["eb_v"] = rphot.calculate_selective_extinction(tgt_info["Bmag"], 
                                                            tgt_info["Vmag"], 
                                                            tgt_info["SpT_simple"],
                                                            grid)

    # Calculate V band extinction
    tgt_info["A_V"] = rphot.calculate_v_band_extinction(tgt_info["eb_v"])

    # Calculate the filter effective wavelength *considering* spectral type
    #eff_lambda = rch.calculate_effective_wavelength(tgt_info["SpT"], filter_list)

    # Determine extinction
    a_mags = rphot.deredden_photometry(tgt_info[["Bmag", "Vmag", "Jmag", "Hmag", 
                                                "Kmag", "W1mag","W2mag", "W3mag", 
                                                "W4mag"]], 
                                      tgt_info[["e_Bmag", "e_Vmag", "e_Jmag",  
                                                "e_Hmag", "e_Kmag", "e_W1mag",  
                                                "e_W2mag", "e_W3mag", "e_W4mag"]], 
                                      filter_eff_lambda, tgt_info["A_V"])

    # Create a mask which has values of 1 for stars outside the local bubble, and
    # values of 0 for stars within it. This is multiplied by the calculated 
    # extinction in each band, treating it as zero for stars within the bubble and
    # as calculated for those stars outside it.
    lb_mask = (tgt_info["Dist"] > 150).astype(int)
                                 
    # Correct for extinction only for those stars outside the Local Bubble
    tgt_info["Bmag_dr"] = tgt_info["Bmag"] - a_mags[:,0] * lb_mask
    tgt_info["Vmag_dr"] = tgt_info["Vmag"] - a_mags[:,1] * lb_mask
    tgt_info["Jmag_dr"] = tgt_info["Jmag"] - a_mags[:,2] * lb_mask
    tgt_info["Hmag_dr"] = tgt_info["Hmag"] - a_mags[:,3] * lb_mask
    tgt_info["Kmag_dr"] = tgt_info["Kmag"] - a_mags[:,4] * lb_mask
    tgt_info["W1mag_dr"] = tgt_info["W1mag"] - a_mags[:,5] * lb_mask
    tgt_info["W2mag_dr"] = tgt_info["W2mag"] - a_mags[:,6] * lb_mask
    tgt_info["W3mag_dr"] = tgt_info["W3mag"] - a_mags[:,7] * lb_mask
    tgt_info["W4mag_dr"] = tgt_info["W4mag"] - a_mags[:,8] * lb_mask

    # Calculate predicted V-K colour
    tgt_info["V-K_calc"] = rphot.calc_vk_colour(tgt_info["VTmag"], tgt_info["RPmag"])

    # -----------------------------------------------------------------------------
    # (4) Estimate angular diameters
    # -----------------------------------------------------------------------------
    # Estimate angular diameters using colour relations. We want to do this using 
    # as many colour combos as is feasible, as this can be a useful diagnostic
    # TODO: Is not correcting reddening for W1-3 appropriate given the laws don't
    # extend that far?
    rdiam.predict_all_ldd(tgt_info)

    # Determine the linear LDD coefficents
    #tgt_info["u_lld"] = rdiam.get_linear_limb_darkening_coeff(tgt_info["logg"],
    #                                                          tgt_info["Teff"],
    #                                                          tgt_info["FeH_rel"], 
    #                                                          "H")

    # Don't have parameters for HD187289, assume u_lld=0.5 for now
    #tgt_info.loc["HD187289", "u_lld"] = 0.5
    
    return tgt_info
    
def load_sequence_logs():
    """
    """
    pkl_obslog = open("data/pionier_observing_log.pkl", "r")
    complete_sequences = pickle.load(pkl_obslog)
    pkl_obslog.close()

    pkl_sequences = open("data/sequences.pkl", "r")
    sequences = pickle.load(pkl_sequences)
    pkl_sequences.close()
    
    return complete_sequences, sequences
"""Script to take bootstrapped oifits files and combine for final results
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.diagnostics as rdiag
import reach.parameters as rparam
import reach.paper as rpaper
import reach.plotting as rplt
import reach.photometry as rphot
import reach.pndrs as rpndrs
import reach.utils as rutils
import pickle

# -----------------------------------------------------------------------------
# Setup & Loading
# -----------------------------------------------------------------------------
combined_fit = True                     # Fit for LDD for multiple seq at once
load_saved_results = False               # Load or do fitting fresh
assign_default_uncertainties = True     # Give default errors to stars without
force_claret_params = False             # Force use of Claret+11 limb d. params
n_bootstraps = 1000
e_wl_frac = 0.02                        # Fractional error on wl scale
#results_folder = "19-03-26_i1000"
#results_folder = "19-05-27_i2000"   # 1st with wavelength cal, only 480
#results_folder = "19-06-06_i2000"  # Attempted in parallel, but incomplete
results_folder = "19-06-10_i1000"  # Wavelength cal, serial
results_path = "results/%s/" % results_folder
e_wl_cal_percent = 1

bc_path =  "/Users/adamrains/code/bolometric-corrections"
band_mask = [1, 0, 0, 0, 0]

# Load in files
print("Loading in files...")
tgt_info = rutils.initialise_tgt_info()
complete_sequences, sequences = rutils.load_sequence_logs()

# Currently no proxima cen or gam pav data, so pop
sequences.pop((102, 'gamPav', 'faint'))
sequences.pop((102, 'gamPav', 'bright'))
sequences.pop((102, 'ProximaCen', 'bright'))

# Don't care about distant RGB
sequences.pop((99, "HD187289", "faint"))
sequences.pop((99, "HD187289", "bright"))
complete_sequences.pop((99, 'HD187289', 'faint'))
complete_sequences.pop((99, 'HD187289', 'bright'))

# -----------------------------------------------------------------------------
# Loading Existing Results
# -----------------------------------------------------------------------------
# Collate bootstrapped results
if load_saved_results:
    print("Loading saved results...")
    sampled_sci_params = rutils.load_sampled_params(results_folder, 
                                                    force_claret_params,
                                                    final_teff_sample=True)
    
    bs_results, results = rutils.load_results(results_folder)

# -----------------------------------------------------------------------------
# Calculating Results For First Time
# -----------------------------------------------------------------------------
# Do two iterations of the fitting, one with literature teffs, and one with
# interferometric teffs
else:
    # 1111111111111111111111111111111111111111111111111111111111111111111111111
    # Run through initially using **literature** teffs
    # 1111111111111111111111111111111111111111111111111111111111111111111111111
    print("-"*79, "\n", "\tInitial Analysis (Literature Teff)\n", "-"*79)
    sampled_sci_params = rutils.load_sampled_params(results_folder, 
                                                    force_claret_params)
    
    print("Getting results of bootstrapping for %s bootstraps..." 
          % n_bootstraps)
    bs_results = rdiam.fit_ldd_for_all_bootstraps(tgt_info, n_bootstraps, 
                                            results_path, sampled_sci_params, 
                                            e_wl_frac=e_wl_frac,
                                            combined_fit=combined_fit) 

    # Summarise results
    results = rdiam.summarise_results(bs_results, tgt_info)
    
    # Calculate **initial** fundamental parameters using literature values
    print("Determining **initial** fundamental parameters...")
    rparam.calc_sample_and_final_params(tgt_info, sampled_sci_params, 
                                        bs_results, results)
    
    # 2222222222222222222222222222222222222222222222222222222222222222222222222
    # Now resample, and run through again using **interferometric** teffs
    # 2222222222222222222222222222222222222222222222222222222222222222222222222
    print("-"*79, "\n", "\tFinal Analysis (Interferometric Teff)\n", "-"*79)
    sampled_sci_params = rparam.sample_all(tgt_info, n_bootstraps, bc_path,
                                           force_claret_params, band_mask,
                                           use_literature_teffs=False)
                                                                       
    rutils.save_sampled_params(sampled_sci_params, results_folder, 
                               final_teff_sample=True)
    
    bs_results = rdiam.fit_ldd_for_all_bootstraps(tgt_info, n_bootstraps, 
                                            results_path, sampled_sci_params, 
                                            e_wl_frac=e_wl_frac,
                                            combined_fit=combined_fit) 
    # Summarise results
    results = rdiam.summarise_results(bs_results, tgt_info)
    
    # Save results
    rutils.save_results(bs_results, results, results_folder)
    
    # Calculate **final** fundamental parameters using interferometric values
    print("Determining **final** fundamental parameters...")
    rparam.calc_sample_and_final_params(tgt_info, sampled_sci_params, 
                                        bs_results, results)
                                        
# -----------------------------------------------------------------------------
# Table generation and plotting
# -----------------------------------------------------------------------------
print("-"*79, "\n", "\tTables and Plots (Literature Teff)\n", "-"*79)
# Generate tables
print("Generating tables...")
rpaper.make_table_targets(tgt_info)
rpaper.make_table_calibrators(tgt_info, sequences)
rpaper.make_table_observation_log(tgt_info, complete_sequences, sequences)
rpaper.make_table_fbol(tgt_info)
rpaper.make_table_seq_results(results)
rpaper.make_table_final_results(tgt_info)
rpaper.make_table_limb_darkening(tgt_info)

# Generate plots
print("Generating plots...")
rplt.plot_hr_diagram(tgt_info, plot_isochrones_basti=True)
rplt.plot_casagrande_teff_comp(tgt_info)
rplt.plot_lit_diam_comp(tgt_info)
rplt.plot_sidelobe_vis2_fit(tgt_info, results)  
rplt.plot_joint_seq_paper_vis2_fits(tgt_info, results, n_rows=4, n_cols=2)
rplt.plot_colour_rel_diam_comp(tgt_info)
rplt.plot_bootstrapping_summary(results, bs_results, plot_cal_info=False, 
                                sequences=sequences, 
                                complete_sequences=complete_sequences, 
                                tgt_info=tgt_info)
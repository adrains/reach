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

load_saved_results = False
n_bootstraps = 1000
results_folder = "19-03-26_i1000"
#results_path = "/home/arains/code/reach/results/%s/" % results_folder
results_path = "results/%s/" % results_folder
e_wl_cal_percent = 1

# Load in files
print("Loading in files...")
tgt_info = rutils.initialise_tgt_info()
complete_sequences, sequences = rutils.load_sequence_logs()

# Load in distributions
n_pred_ldd = pd.read_csv(results_path + "n_pred_ldd.csv") 
e_pred_ldd = pd.read_csv(results_path + "e_pred_ldd.csv") 
n_logg = pd.read_csv(results_path + "n_logg.csv") 
n_teff = pd.read_csv(results_path + "n_teff.csv") 
n_feh = pd.read_csv(results_path + "n_feh.csv") 
n_u_lld = pd.read_csv(results_path + "n_u_lld.csv") 

# Currently broken, don't consider
complete_sequences.pop((102, 'delEri', 'bright'))

# Currently no proxima cen or gam pav data, so pop
sequences.pop((102, 'gamPav', 'faint'))
sequences.pop((102, 'gamPav', 'bright'))
sequences.pop((102, 'ProximaCen', 'bright'))

# Determine u_lld from its distribution
tgt_info.loc[n_u_lld.columns.values, "u_lld"] = n_u_lld.mean().values
tgt_info.loc[n_u_lld.columns.values, "e_u_lld"] = n_u_lld.std().values

# Collate bootstrapped results
if load_saved_results:
    print("Loading saved results...")
    bs_results, results = rutils.load_results(results_folder)
    
    # Pop HD187289
    results.drop(results[results["STAR"]=="HD187289"].index, inplace=True)
    bs_results.pop("HD187289 (faint, 99)")
    bs_results.pop("HD187289 (bright, 99)")

else:
    # Get results
    print("Getting results of bootstrapping...")
    bs_results = rdiam.collate_bootstrapping(tgt_info, n_bootstraps, results_path,
                                         n_u_lld) 

    # Summarise results
    results = rdiam.summarise_bootstrapping(bs_results, tgt_info)

print("Determining fundamental parameters...")

# Combine angular diameter measurements
rparam.combine_seq_ldd(tgt_info, results)

# Calculate the physical radii
rparam.calc_all_r_star(tgt_info)

# Compute fluxes
rparam.calc_all_f_bol(tgt_info, 10000)

# Compute temperatures
rparam.calc_all_teff(tgt_info, 10000)

# Compute luminosity
rparam.calc_all_L_bol(tgt_info, 10000)

# Generate tables
print("Generating tables...")
rpaper.make_table_targets(tgt_info)
rpaper.make_table_calibrators(tgt_info, sequences)
rpaper.make_table_observation_log(tgt_info, complete_sequences, sequences)
rpaper.make_table_fbol(tgt_info)
rpaper.make_table_seq_results(results)
rpaper.make_table_final_results(tgt_info)

# Generate plots
print("Generating plots...")
rplt.plot_lit_diam_comp(tgt_info)
rplt.plot_paper_vis2_fits(results, n_rows=8, n_cols=2)
rplt.plot_colour_rel_diam_comp(tgt_info, colour_rel="V-W3")
rplt.plot_colour_rel_diam_comp(tgt_info, colour_rel="V-W4")
rplt.plot_bootstrapping_summary(results, bs_results, plot_cal_info=True, 
                                sequences=sequences, 
                                complete_sequences=complete_sequences, 
                                tgt_info=tgt_info)
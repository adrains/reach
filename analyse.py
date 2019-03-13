"""
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

n_bootstraps = 200
results_path = "/home/arains/code/reach/results/19-02-22_i200/"

# Load in files
tgt_info = rutils.initialise_tgt_info()
complete_sequences, sequences = rutils.load_sequence_logs()

# Currently broken, don't consider
complete_sequences.pop((102, 'delEri', 'bright'))

# Currently no proxima cen or gam pav data, so pop
sequences.pop((102, 'gamPav', 'faint'))
sequences.pop((102, 'gamPav', 'bright'))
sequences.pop((102, 'ProximaCen', 'bright'))

# Collate bootstrapped results
bs_results = rdiam.collate_bootstrapping(tgt_info, n_bootstraps, results_path) 

# Summarise results
results = rdiam.summarise_bootstrapping(bs_results, tgt_info)

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
rpaper.make_table_targets(tgt_info)
rpaper.make_table_calibrators(tgt_info, sequences)
rpaper.make_table_observation_log(tgt_info, complete_sequences, sequences)
rpaper.make_table_fbol(tgt_info, results)
rpaper.make_table_seq_results(results)
rpaper.make_table_final_results(tgt_info)

# Generate plots
rplt.plot_lit_diam_comp(tgt_info)



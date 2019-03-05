"""
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.diagnostics as rdiag
import reach.parameters as rparam
import reach.plotting as rplt
import reach.photometry as rphot
import reach.pndrs as rpndrs
import reach.utils as rutils
import pickle

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
results_path = "/home/arains/code/reach/results/19-02-22_i200/"
bs_results = rdiam.collate_bootstrapping(tgt_info, n_bootstraps, results_path) 

# Summarise results
results = rdiam.summarise_bootstrapping(bs_results, tgt_info)

# Compute fluxes
rparam.calc_teff_from_bc(tgt_info, results, 10000)

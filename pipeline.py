"""
Script to run complete diameter computation pipeline. The broad steps are 
outlined below.

A) Target and calibrator background:
    1) Collate target information from literature
        i)   SpT, + spectroscopic T_eff, log(g), [Fe/H]
        ii)  Magnitudes: V (Tycho-2), K (2MASS), reddening (Green et al. 2015)
        iii) F_bol from spectrophotometrically calibrated spectra + atmospheres 
        iv)  Previous interferometric measurements
    2) Collate calibrator information from literature
        i)  SpT
        ii) Magnitudes: V (Tycho-2), K (2MASS), reddening (Green et al. 2015)
    
B) Observation reduction, organisation
    1) Download all observations.
    2) Organise observations by moving only complete CAL-SCI-CAL-SCI-CAL 
       sequences to a new directory structure.
    3) Reduce obervations.
    4) Check reduction (e.g. inspect visibility curves for signs of binarity).
    5) Calibrate only the complete/best quality bright/faint sequences. 
       This is done by creating/editing pndrsScript.i with the functions:
        i)   oiFitsFlagOiData - ignoring observsations within pndrs
        ii)  oiFitsSplitNight - splitting the "night" between different 
             sequences within pndrs
       Here there also needs to be some consideration for the calibration files
       sent by Xavier Habois from ESO (see 10/07/18 Ireland email).
       
       At the conclusion of this calibration, there should be a single oiFits
       file for each science target containing the (among other things) the 
       squared visibilities and closure phases.

C) Calculation of angular diameters
    1) Determine limb darkening coefficients by interpolating grids derived
       from model atmospheres for the previously obtained spectroscopic values
    2) Determine linear limb-darkened diameters as per Hanbury Brown et al 1974
    3) Perform Monte Carlo model-fitting parameter uncertainty estimation
    4) Compare with photometric predictions of LDD
    
Important reference publications:
 - Boyajian et al. 2014 - Predicting Stellar Angular Diameters
 - Green et al. 2015 - A Three-dimensional Map of Milky Way Dust
 - Bessell 2000 - The Hipparcos and Tycho Photometric System Passbands
 - Hog et al. 2000 - The Tycho-2 catalogue of the 2.5 million brightest stars
 - Skrutskie et al. 2006 - The Two Micron All Sky Survey (2MASS)
 - Wright et al. 2010 - The Wide-field Infrared Survey Explorer (WISE): Mission
   Description and Initial On-orbit Performance
 - Mann et al. 2015 - How to Constrain Your M Dwarf: Measuring Effective 
   Temperature, Bolometric Luminosity, Mass, and Radius
 - Hanbury Brown et al. 1974 - The effects of limb darkening on measurements of 
   angular size with an intensity interferometer
 - Claret & Bloemen 2011 - Gravity and limb-darkening coefficients for the 
   Kepler, CoRoT, Spitzer, uvby, UBVRIJHK, and Sloan photometric systems
 - Magic et al. 2015 - The Stagger-grid: A grid of 3D stellar atmosphere 
   models. IV. Limb darkening coefficients
 - Derekas et al. 2011 - HD 181068: A Red Giant in a Triply Eclipsing Compact 
   Hierarchical Triple System
 - Casagrande et al. 2010 - An absolutely calibrated Teff scale from the 
   Infrared Flux Method Dwarfs and subgiants
 - Casagrande et al. 2014 - Towards stellar effective temperatures and 
   diameters at 1 per cent accuracy for future surveys
   
Required Software
 - dustmaps - Python package to query Green at al. 2015/2017 dust maps.
   Located at: http://argonaut.skymaps.info/
 - pndrs - PIONIER data reduction software
 
Required Catalogues
 - Tycho
 - 2MASS
 - WISE
 - Gaia DR2
"""
from __future__ import division, print_function

import os
import time
import glob
import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.diagnostics as rdiag
import reach.plotting as rplt
import reach.photometry as rphot
import reach.pndrs as rpndrs
import reach.utils as rutils
import reach.parameters as rparam
import platform
from sys import exit

# -----------------------------------------------------------------------------
# (0) Parameters
# -----------------------------------------------------------------------------
# Time to append to results_path to prevent being overriden
str_date = time.strftime("%y-%m-%d")  

# TODO: move to parameter file
lb_pc = 150 # The size of the local bubble in parsecs
calibrate_calibrators = False
test_all_cals = False
run_local = False
already_calibrated = False
do_random_ifg_sampling = True
do_gaussian_diam_sampling = True
test_one_seq_only = False
assign_default_uncertainties = True
n_bootstraps = 2000
pred_ldd_col = "LDD_pred"
e_pred_ldd_col = "e_LDD_pred"
base_path = "/priv/mulga1/arains/pionier/complete_sequences/%s_v3.73_abcd/"
results_folder = "%s_i%i" % (str_date, n_bootstraps)
results_path = "/home/arains/code/reach/results/%s/" % results_folder

if not os.path.exists(results_path):
    os.mkdir(results_path)

print("\nBeginning calibration and fitting run. Parameters set as follow:")
print(" - n_bootstraps\t\t\t=\t%i" % n_bootstraps)
print(" - run_local\t\t\t=\t%s" % run_local)
print(" - already_calibrated\t\t=\t%s" % already_calibrated)
print(" - do_random_ifg_sampling\t=\t%s" % do_random_ifg_sampling)
print(" - do_gaussian_diam_sampling\t=\t%s" % do_gaussian_diam_sampling)

print("<i>Strap in</i> for bootstrapping.")

# -----------------------------------------------------------------------------
# (1) Import target details
# -----------------------------------------------------------------------------
# Targets information is loaded into a pandas dataframe, with column labels for
# each of the stored parameters (e.g. VTmag) and row indices of HD ID
tgt_info = rutils.initialise_tgt_info()

# Sample diameters for bootstrapping (if n_bootstraps < 1, actual predictions)
n_pred_ldd, e_pred_ldd = rdiam.sample_n_pred_ldd(tgt_info, n_bootstraps, 
                                                 pred_ldd_col, e_pred_ldd_col,
                                                 do_gaussian_diam_sampling)
                                                 
# Sample stellar parameters
sampled_sci_params = rparam.sample_parameters(tgt_info, n_bootstraps)
rutils.save_sampled_params(sampled_sci_params, results_folder)

# -----------------------------------------------------------------------------
# (5) Import observing logs
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py
# Format of this file is as follows
complete_sequences, sequences = rutils.load_sequence_logs()

# Currently broken, don't consider
complete_sequences.pop((102, 'delEri', 'bright'))

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
# (6) Inspect reduced data
# -----------------------------------------------------------------------------
# Check visibilities for anything unusual (?) or potential saturated data

# -----------------------------------------------------------------------------
# (7) Write YYYY-MM-DD_pndrsScript.i
# -----------------------------------------------------------------------------
# Do the following:
#  i)  Exclude bad calibrators (informed by 5)
#  ii) Split nights between sequences
if not run_local and not already_calibrated:
    rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info)
elif not already_calibrated:
    rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info, 
                                     run_local=run_local)

# -----------------------------------------------------------------------------
# (7.5) Calibrate calibrators against each other
# -----------------------------------------------------------------------------
if calibrate_calibrators:
    print("-"*79, "\nCalibrating Calibrators\n", "-"*79)
    rdiag.calibrate_calibrators(sequences, complete_sequences, base_path, 
                                tgt_info, n_pred_ldd, e_pred_ldd, 
                                test_all_cals)
    
    # Finished calibrating calibrators, exit
    print("Finished calibrating calibrators")
    exit(0)

# -----------------------------------------------------------------------------
# (8+) Bootstrap the calibration pipeline
# -----------------------------------------------------------------------------
# Run N bootstrapping iterations of the following:
# - Write YYYY-MM-DD_oiDiam.fits files for each night of observing
# - Run pndrsCalibrate for each night of observing
# - Collate vis^2 and fit angular diameters for all science targets

# For testing purposes, only consider one star
if test_one_seq_only:
    seq1 = (99, 'epsEri', 'faint')
    seq2 = (99, 'epsEri', 'bright')
    complete_sequences = {seq2:complete_sequences[seq2]}
                      
    sequences = {seq2:sequences[seq2]}

rpndrs.run_n_bootstraps(sequences, complete_sequences, base_path, tgt_info,
                        n_pred_ldd, e_pred_ldd, n_bootstraps, results_path,
                        run_local=run_local, 
                        already_calibrated=already_calibrated,
                        do_random_ifg_sampling=do_random_ifg_sampling)
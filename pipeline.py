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
import pickle
import platform
from sys import exit
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# -----------------------------------------------------------------------------
# (0) Parameters
# -----------------------------------------------------------------------------
# Time to append to results_path to prevent being overriden
str_date = time.strftime("%y-%m-%d")  

# TODO: move to parameter file
lb_pc = 150 # The size of the local bubble in parsecs
calibrate_calibrators = False
run_local = False
already_calibrated = False
do_random_ifg_sampling = True
do_gaussian_diam_sampling = True
test_one_seq_only = False
do_ldd_fitting = True
combine_previous_bootstraps = False
n_bootstraps = 120
pred_ldd_col = "LDD_VW3_dr"
e_pred_ldd_col = "e_LDD_VW3_dr"
base_path = "/priv/mulga1/arains/pionier/complete_sequences/%s_v3.73_abcd/"
results_path = "/home/arains/code/reach/results/" + str_date + "/"

print("\nBeginning calibration and fitting run. Parameters set as follow:")
print(" - n_bootstraps\t\t\t=\t%i" % n_bootstraps)
print(" - run_local\t\t\t=\t%s" % run_local)
print(" - already_calibrated\t\t=\t%s" % already_calibrated)
print(" - do_random_ifg_sampling\t=\t%s" % do_random_ifg_sampling)
print(" - do_gaussian_diam_sampling\t=\t%s" % do_gaussian_diam_sampling)
print(" - combine_previous_bootstraps\t=\t%s" % combine_previous_bootstraps)

print("<i>Strap in</i> for bootstrapping.")

# -----------------------------------------------------------------------------
# (1) Import target details
# -----------------------------------------------------------------------------
# Targets information is loaded into a pandas dataframe, with column labels for
# each of the stored parameters (e.g. VTmag) and row indices of HD ID
tgt_info = rutils.load_target_information()

# Calculate distances
tgt_info["Dist"] = 1000 / tgt_info["Plx"]

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

# -----------------------------------------------------------------------------
# (4) Estimate angular diameters
# -----------------------------------------------------------------------------
# Estimate angular diameters using colour relations. We want to do this using 
# as many colour combos as is feasible, as this can be a useful diagnostic
# TODO: Is not correcting reddening for W1-3 appropriate given the laws don't
# extend that far?
"""
ldd_bv_dr, e_ldd_vk_dr = rdiam.predict_ldd_boyajian(tgt_info["Bmag_dr"], 
                                                    tgt_info["e_BTmag"], 
                                                    tgt_info["Vmag_dr"], 
                                                    tgt_info["e_VTmag"], "B-V")
"""                                            
ldd_vk_dr, e_ldd_vk_dr = rdiam.predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                                    tgt_info["e_VTmag"], 
                                                    tgt_info["Kmag_dr"], 
                                                    tgt_info["e_Kmag"], "V-K")
                                                  
ldd_vw3_dr, e_ldd_vw3_dr = rdiam.predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                                    tgt_info["e_VTmag"], 
                                                    tgt_info["W3mag"], 
                                                    tgt_info["e_W3mag"],"V-W3")                                            
                                                     
#tgt_info["LDD_BV_dr"] = ldd_bv_dr
#tgt_info["e_LDD_BV_dr"] = e_ldd_bv_dr

tgt_info["LDD_VK_dr"] = ldd_vk_dr
tgt_info["e_LDD_VK_dr"] = e_ldd_vk_dr

tgt_info["LDD_VW3_dr"] = ldd_vw3_dr
tgt_info["e_LDD_VW3_dr"] = e_ldd_vw3_dr

#rplt.plot_diameter_comparison(ldd_vk, ldd_vw3, ldd_vk_dr, ldd_vw3_dr, "(V-K)", 
                               #"(V-W3)")

# Sample diameters for bootstrapping (if n_bootstraps < 1, actual predictions)
n_pred_ldd, e_pred_ldd = rdiam.sample_n_pred_ldd(tgt_info, n_bootstraps, 
                                                 pred_ldd_col, e_pred_ldd_col,
                                                 do_gaussian_diam_sampling)

# Determine the linear LDD coefficents
tgt_info["u_lld"] = rdiam.get_linear_limb_darkening_coeff(tgt_info["logg"],
                                                          tgt_info["Teff"],
                                                          tgt_info["FeH_rel"], 
                                                          "H")

# Don't have parameters for HD187289, assume u_lld=0.5 for now
tgt_info.loc["HD187289", "u_lld"] = 0.5

# -----------------------------------------------------------------------------
# (5) Import observing logs
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py
# Format of this file is as follows
pkl_obslog = open("data/pionier_observing_log.pkl", "r")
complete_sequences = pickle.load(pkl_obslog)
pkl_obslog.close()

pkl_sequences = open("data/sequences.pkl", "r")
sequences = pickle.load(pkl_sequences)
pkl_sequences.close()

# Currently broken, don't consider
complete_sequences.pop((102, 'delEri', 'bright'))

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
                                tgt_info, n_pred_ldd, e_pred_ldd)
    
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

# -----------------------------------------------------------------------------
# (N) Create summary pdf with vis^2 plots for all science targets
# -----------------------------------------------------------------------------
# Collate the bootstrapping run
bs_results = rdiam.collate_bootstrapping(tgt_info, n_bootstraps, results_path, 
                                         pred_ldd_col, 
                                         prune_errant_baselines=True)

# Save the results from this bootstrapping run                   
pkl_bs_results = open("results/bootstrapped_results_%s_n%i.pkl" 
                      % (str_date, n_bootstraps), "wb")
pickle.dump(bs_results, pkl_bs_results)
pkl_bs_results.close()
                             
if combine_previous_bootstraps:                                         
    # Combine these bootstrapping results with any previous ones
    pkl_list = glob.glob("results/*results*pkl")
    all_bs_results = rutils.combine_independent_boostrap_runs(pkl_list)

    # Summarise the bootstrapping run                                
    results = rdiam.summarise_bootstrapping(all_bs_results, tgt_info, 
                                            pred_ldd_col, e_pred_ldd_col)

    rplt.plot_bootstrapping_summary(results, all_bs_results)

else:
    results = rdiam.summarise_bootstrapping(bs_results, tgt_info, pred_ldd_col, 
                                            e_pred_ldd_col)

    rplt.plot_bootstrapping_summary(results, bs_results)
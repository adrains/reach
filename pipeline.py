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

import numpy as np
import pandas as pd
import reach.core as rch
import reach.plotting as rplt
import pickle
import platform
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# Parameter to run a subset of the pipeline
already_calibrated = False

# -----------------------------------------------------------------------------
# (1) Import target details
# -----------------------------------------------------------------------------
# Targets information is loaded into a pandas dataframe, with column labels for
# each of the stored parameters (e.g. VTmag) and row indices of HD ID
tgt_info = rch.load_target_information()

# -----------------------------------------------------------------------------
# (2) Convert Tycho magnitudes to Johnson-Cousins magnitudes
# -----------------------------------------------------------------------------
# Convert Tycho V to Johnson system using Bessell 2000

# For simplification during testing, remove any stars that fall outside the 
# VT --> V conversion from Bessell 2000
tgt_info = tgt_info.drop(["GJ551","HD133869"])

# Convert VT and BT to V and B
# TODO: proper treatment of magnitude errors
Bmag, Vmag = rch.convert_vtbt_to_vb(tgt_info["BTmag"], tgt_info["VTmag"])

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
grid = rch.create_spt_uv_grid()
                     
# Calculate selective extinction (i.e. (B-V) colour excess)
tgt_info["eb_v"] = rch.calculate_selective_extinction(tgt_info["Bmag"], 
                                                      tgt_info["Vmag"], 
                                                      tgt_info["SpT_simple"],
                                                      grid)

# Calculate V band extinction
tgt_info["A_V"] = rch.calculate_v_band_extinction(tgt_info["eb_v"])

# Calculate the filter effective wavelength *considering* spectral type
#eff_lambda = rch.calculate_effective_wavelength(tgt_info["SpT"], filter_list)

# Determine extinction
a_mags = rch.deredden_photometry(tgt_info[["Bmag", "Vmag", "Jmag", "Hmag", 
                                           "Kmag", "W1mag","W2mag", "W3mag", 
                                           "W4mag"]], 
                                 tgt_info[["e_Bmag", "e_Vmag", "e_Jmag",  
                                           "e_Hmag", "e_Kmag", "e_W1mag",  
                                           "e_W2mag", "e_W3mag", "e_W4mag"]], 
                                 filter_eff_lambda, tgt_info["A_V"])
                                 
# Correct magnitudes for extinction
# TODO: Only correct the magnitudes if the star is beyond the local bubble
tgt_info["Bmag_dr"] = tgt_info["Bmag"] - a_mags[:,0]
tgt_info["Vmag_dr"] = tgt_info["Vmag"] - a_mags[:,1]
tgt_info["Jmag_dr"] = tgt_info["Jmag"] - a_mags[:,2]
tgt_info["Hmag_dr"] = tgt_info["Hmag"] - a_mags[:,3]
tgt_info["Kmag_dr"] = tgt_info["Kmag"] - a_mags[:,4]
#tgt_info["W1mag_dr"] = a_mags[:,5]
#tgt_info["W2mag_dr"] = a_mags[:,6]
#tgt_info["W3mag_dr"] = a_mags[:,7]
#tgt_info["W4mag_dr"] = a_mags[:,8]

# -----------------------------------------------------------------------------
# (4) Estimate angular diameters
# -----------------------------------------------------------------------------
# Estimate angular diameters using colour relations. We want to do this using 
# as many colour combos as is feasible, as this can be a useful diagnostic
ldd_vk, e_ldd_vk = rch.predict_ldd_boyajian(tgt_info["Vmag"], 
                                            tgt_info["e_VTmag"], 
                                            tgt_info["Kmag"], 
                                            tgt_info["e_Kmag"], "V-K")
                                            
ldd_vw3, e_ldd_vw3 = rch.predict_ldd_boyajian(tgt_info["Vmag"], 
                                              tgt_info["e_VTmag"], 
                                              tgt_info["W3mag"], 
                                              tgt_info["e_W3mag"], "V-W3")
                                            
ldd_vk_dr, e_ldd_vk_dr = rch.predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                                  tgt_info["e_VTmag"], 
                                                  tgt_info["Kmag_dr"], 
                                                  tgt_info["e_Kmag"], "V-K")
                                                  
ldd_vw3_dr, e_ldd_vw3_dr = rch.predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                                    tgt_info["e_VTmag"], 
                                                    tgt_info["W3mag"], 
                                                    tgt_info["e_W3mag"],"V-W3")                                            
                                                     
#tgt_info["LDD_VK"] = ldd_vk
#tgt_info["e_LDD_VK"] = e_ldd_vk
#tgt_info["LDD_VW3"] = ldd_vw3
#tgt_info["e_LDD_VK"] = e_ldd_vk
tgt_info["LDD_VK_dr"] = ldd_vk_dr
tgt_info["e_LDD_VK_dr"] = e_ldd_vk_dr

tgt_info["LDD_VW3_dr"] = ldd_vw3_dr
tgt_info["e_LDD_VW3_dr"] = e_ldd_vw3_dr

#rplt.plot_diameter_comparison(ldd_vk, ldd_vw3, ldd_vk_dr, ldd_vw3_dr, "(V-K)", 
                               #"(V-W3)")

# -----------------------------------------------------------------------------
# (5) Import observing logs
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py
# Format of this file is as follows
pkl_obslog = open("data/pionier_observing_log.pkl", "r")
complete_sequences = pickle.load(pkl_obslog)
pkl_obslog.close()

# -----------------------------------------------------------------------------
# (6) Inspect reduced data
# -----------------------------------------------------------------------------
# Check visibilities for anything unusual (?) or potential saturated data
pass

# -----------------------------------------------------------------------------
# (7) Write YYYY-MM-DD_oiDiam.fits files for each night of observing
# -----------------------------------------------------------------------------
# Fits file with two HDUs: [0] is (empty) primary image, [1] is table of diams
pkl_sequences = open("data/sequences.pkl", "r")
sequences = pickle.load(pkl_sequences)
pkl_sequences.close()

if "wintermute" not in platform.node() and not already_calibrated:
    nights = rch.save_nightly_ldd(sequences, complete_sequences, tgt_info)
    
elif not already_calibrated:
    nights = rch.save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                                  run_local=True, ldd_col="LDD_VK_dr", 
                                  e_ldd_col="e_LDD_VK_dr")

# -----------------------------------------------------------------------------
# (8) Write YYYY-MM-DD_pndrsScript.i
# -----------------------------------------------------------------------------
# Do the following:
#  i)  Exclude bad calibrators (informed by 5)
#  ii) Split nights between sequences
if "wintermute" not in platform.node() and not already_calibrated:
    rch.save_nightly_pndrs_script(complete_sequences, tgt_info)
elif not already_calibrated:
    rch.save_nightly_pndrs_script(complete_sequences, tgt_info, run_local=True)

# -----------------------------------------------------------------------------
# (9) Run pndrsCalibrate for each night of observing
# -----------------------------------------------------------------------------
base_path = "/priv/mulga1/arains/pionier/complete_sequences/@_v3.73_abcd/"

if "wintermute" not in platform.node() and not already_calibrated:
    obs_folders = [base_path.replace("@", night) for night in nights.keys()]
    rch.calibrate_all_observations(obs_folders)

    # Move oifits files back to central location (reach/results/ by default)
    rch.move_sci_oifits()

# Collate calibrated vis2 data
if "wintermute" not in platform.node():
    vis2, e_vis2, baselines, wavelengths = rch.collate_vis2_from_file()
else:
    path = "/Users/adamrains/code/reach/results/"
    vis2, e_vis2, baselines, wavelengths = rch.collate_vis2_from_file(path)
    
# -----------------------------------------------------------------------------
# (10) Fit angular diameters to vis^2 of all science targets
# -----------------------------------------------------------------------------
# Determine the linear LDD coefficents
tgt_info["u_lld"] = rch.get_linear_limb_darkening_coeff(tgt_info["logg"],
                                                        tgt_info["Teff"],
                                                        tgt_info["FeH_rel"], 
                                                        "H")

rch.fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info)
"""
feh = -0.29
teff = 7014
logg = 4.04

u_lld = rch.get_linear_limb_darkening_coeff(logg, teff, feh, "H")

# Get the visibilities
faint_seq = "results/2018-04-18_SCI_bet_TrA_oidataCalibrated.fits"
bright_seq = "results/2018-08-07_SCI_bet_TrA_oidataCalibrated.fits"

vis2_f, e_vis2_f, baselines_f, wavelengths = rch.extract_vis2(faint_seq)
vis2_b, e_vis2_b, baselines_b, wavelengths = rch.extract_vis2(bright_seq)

vis2 = np.vstack((vis2_f, vis2_b))
e_vis2 = np.vstack((e_vis2_f, e_vis2_b))
baselines = np.hstack((baselines_f, baselines_b))
#wavelengths = np.hstack((wavelengths_f, wavelengths_b))

popt, pcov = rch.fit_for_ldd(vis2, e_vis2, baselines, wavelengths, u_lld, 
                             tgt_info.loc["HD141891"]["LDD_VK_dr"])
                             
# Tau Cet
feh = -0.55
teff = 5227
logg = 4.2

u_lld = rch.get_linear_limb_darkening_coeff(logg, teff, feh, "H")

faint_seq = "results/2018-08-07_SCI_Tau_Cet_oidataCalibrated.fits"
bright_seq = "results/2018-08-06_SCI_Tau_Cet_oidataCalibrated.fits"

vis2_f, e_vis2_f, baselines_f, wavelengths = rch.extract_vis2(faint_seq)
vis2_b, e_vis2_b, baselines_b, wavelengths = rch.extract_vis2(bright_seq)

vis2 = np.vstack((vis2_f, vis2_b))
e_vis2 = np.vstack((e_vis2_f, e_vis2_b))
baselines = np.hstack((baselines_f, baselines_b))

popt, pcov = rch.fit_for_ldd(vis2, e_vis2, baselines, wavelengths, u_lld, 
                             tgt_info.loc["HD10700"]["LDD_VK_dr"])
"""
# -----------------------------------------------------------------------------
# (N) Create summary pdf with vis^2 plots for all science targets
# -----------------------------------------------------------------------------
pass
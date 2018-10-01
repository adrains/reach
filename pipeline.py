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
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# -----------------------------------------------------------------------------
# (1) Import target details
# -----------------------------------------------------------------------------
# Targets information is loaded into a pandas dataframe, with column labels for
# each of the stored parameters (e.g. VTmag) and row indices of HD ID
tgt_info = rch.load_target_information()

# -----------------------------------------------------------------------------
# (2) Estimate angular diameters for all targets
# -----------------------------------------------------------------------------
# Do the following:
#  i)   Convert Tycho V to Johnson system using Bessell 2000
#  ii)  Correct for reddening 
#  iii) Estimate angular diameters using colour relation

# For simplification during testing, remove any stars that fall outside the 
# VT --> V conversion from Bessell 2000
tgt_info = tgt_info.drop(["GJ551","HD133869"])

# Convert VT to V
# TODO: proper treatment of magnitude errors
tgt_info["Vmag"] = rch.convert_vt_to_v(tgt_info["BTmag"], tgt_info["VTmag"])   
tgt_info["e_Vmag"] = tgt_info["e_VTmag"]

# Correct for reddening 
pass

# Estimate angular diameters
ldd, e_ldd = rch.predict_ldd_boyajian(tgt_info.Vmag, tgt_info.e_VTmag, 
                                    tgt_info.W3mag, tgt_info.e_W3mag, "V-W3")
                                    
tgt_info["LDD_V_W3"] = ldd
tgt_info["e_LDD_V_W3"] = e_ldd

# -----------------------------------------------------------------------------
# (3) Import observing logs
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py
# Format of this file is as follows
pkl_obslog = open("pionier_observing_log.pkl", "r")
complete_sequences = pickle.load(pkl_obslog)
pkl_obslog.close()

# -----------------------------------------------------------------------------
# (4) Inspect reduced data
# -----------------------------------------------------------------------------
# Check visibilities for anything unusual (?) or potential saturated data

# -----------------------------------------------------------------------------
# (5) Write YYYY-MM-DD_oiDiam.fits files for each night of observing
# -----------------------------------------------------------------------------
# Fits file with two HDUs: [0] is (empty) primary image, [1] is table of diams
pkl_sequences = open("sequences.pkl", "r")
sequences = pickle.load(pkl_sequences)
pkl_sequences.close()

nights = rch.save_nightly_ldd(sequences, complete_sequences, tgt_info)

# -----------------------------------------------------------------------------
# (6) Write YYYY-MM-DD_pndrsScript.i
# -----------------------------------------------------------------------------
# Do the following:
#  i)  Exclude bad calibrators (informed by 5)
#  ii) Split nights between sequences

# -----------------------------------------------------------------------------
# (7) Run pndrsCalibrate for each night of observing
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (8) Create summary pdf with vis^2 plots for all science targets
# -----------------------------------------------------------------------------
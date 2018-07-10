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
    1) Download and reduce all observations
    2) Organise reduced obervations
    3) Inpect visibility curves for signs of binarity.
    4) Using generate_obs_summary.py, determine which observations are complete
       and calibrate only the complete/best quality bright/faint sequences. 
       This is done by creating/editing pndrsScript.i with the functions:
        i)   oiFitsFlagOiData - ignoring observsations within pndrs
        ii)  oiFitsSplitNight - splitting the "night" between different 
             sequences within pndrs
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
import pickle
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# -----------------------------------------------------------------------------
# Import list of stars, query for 2MASS and Tycho Details
# -----------------------------------------------------------------------------
# This is currently done in query_for_targets.py, which at the moment is a 
# standalone script. Later it might be incorporated as a function.
#
# Initialise data structures
# e.g. --> sci_targets["eps eri"] = [Gaia_ID, Tycho_ID, V, V_err, J, J_err, 
#                                    H, H_err, K, K_err, SpT, Parallax, T_eff,
#                                    T_eff_err, logg, logg_err, [Fe/H], 
#                                    [Fe,H]_err, reddening, reddening_err]
#sci_targets = {}
#cal_targets = {}

pkl_ids = open("target_data_ids.pkl", "r")
pkl_2mass = open("target_data_2mass.pkl", "r")
pkl_tycho = open("target_data_tycho2.pkl", "r")

all_target_ids = pickle.load(pkl_ids)
[dict_2mass, table_2mass] = pickle.load(pkl_2mass)
[dict_tycho, table_tycho] = pickle.load(pkl_tycho)

pkl_ids.close()
pkl_2mass.close()
pkl_tycho.close()

# -----------------------------------------------------------------------------
# Retrieve Literature Angular Diameter Estimations
# -----------------------------------------------------------------------------
# Get angular diameter predictions from JMMC Stellar Diameters Catalogue (JSDC)
# from Chelli et al. 2016

# Vizier catalogue ID for JSDC
cat_id_JSDC = "II/346/jsdc_v2"

# Setup custom Vizier object to query all columns
vizier_all = Vizier(columns=["**"])
#radius_2 = 450 * arcsec

# Construct new tables to store all of the target data
table_jsdc = vizier_all.query_object("eps eri", cat_id_JSDC)[0]
table_jsdc.remove_rows(slice(0, len(table_jsdc)))

dict_jsdc = {}

print("Getting JSDC entries...")

# For every target, query Vizier for the details. Select the targets based on
# their HD numbers
for star in all_target_ids.keys():
    if star not in dict_jsdc.keys() and all_target_ids[star][2]:

        results = []
        results = vizier_all.query_object(star, cat_id_JSDC)

        if len(results) > 0:
            results = results[0]
        else:
            print(star, "not in catalogue")
            continue
            
        # JSDC catalogue does not use a standard set of IDs (e.g. HD), so
        # we will also try the more common name (e.g. Bayer designation, or 
        # HR #). 
        for star_i, jsdc_id in enumerate(results[:]["Name"]):
            id_found = False
            
            # Get HD number if available (and remove spaces)
            if all_target_ids[star][2]:
                temp_star_hd = all_target_ids[star][2].replace(" ", "")
                
            # Get Bayer Designation if available (and remove spaces)
            if all_target_ids[star][3]:
                temp_star_bayer = all_target_ids[star][3].replace(" ", "")
            
            # Get primary (i.e. key) ID (and remove spaces)
            temp_star_id = star.replace(" ", "")
            
            # Get ID used for JSDC, removing spaces and dots
            temp_jsdc_id = jsdc_id.replace(" ", "").replace(".", "")
            
            # If one of Primary/HD/Bayer IDs match, retrieve and store the data
            if ((temp_star_hd in temp_jsdc_id) 
                or (temp_star_id in temp_jsdc_id)
                or (temp_star_bayer in temp_jsdc_id)):
                table_jsdc.add_row(results[star_i])
                dict_jsdc[star] = table_jsdc[-1]
                
                print(star)
                id_found = True
                break
                
        if not id_found:
            print(star, "not matched with", jsdc_id)
    
    # Account for the fact that several of the stars do not have JSDC data
    elif not all_target_ids[star][0]:
        dict_jsdc[star] = None

# Store the results
pkl_jsdc = open("target_data_jsdc.pkl", "wb")
pickle.dump([dict_jsdc, table_jsdc], pkl_jsdc)
pkl_jsdc.close()

# -----------------------------------------------------------------------------
# Estimate Angular Diameters
# -----------------------------------------------------------------------------
# Using V-K relation from Boyajian et al. 2014, possible for all stars with 
# Tycho and 2MASS data
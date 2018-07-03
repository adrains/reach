"""
Script to query information about the science and calibrator stars.

Usage for astroquery can be found here:
    https://github.com/astropy/astroquery/blob/master/docs/vizier/vizier.rst
"""
import numpy as np
import pickle
from astroquery.vizier import Vizier

# Import list of science and calibrator stars
targets_csv = "all_targets.csv"
all_targets = np.loadtxt(targets_csv, "string", delimiter=",")

# Catalogue IDs on Vizier
cat_id_2mass = "II/246"
cat_id_tycho2 = "I/259"

# Setup custom Vizier object to query all columns
vizier_all = Vizier(columns=["**"])

# Store all star data in dictionaries
dict_2mass = {}
dict_tycho2 = {}

# For every target, query Vizier for the details. In the event more than one
# result is returned, select only the brightest star as all of the PIONIER
# targets are relatively bright
for star in all_targets[:,3]:
    # Get 2MASS results
    if star not in dict_2mass.keys():
        results = vizier_all.query_object(star, cat_id_2mass, None)
        star_i = np.argmin(results[0][:]["Hmag"])
        dict_2mass[star] = results[0][star_i]
    
    # Get Tycho-2 results
    if star not in dict_tycho2.keys():
        results = vizier_all.query_object(star, cat_id_tycho2, None)
        star_i = np.argmin(results[0][:]["VTmag"])
        dict_tycho2[star] = results[0][star_i]
        

# Store the results
pickle.dump(dict_2mass, open("2mass_retrieved_target_data.pkl", "wb"))
pickle.dump(dict_tycho2, open("tycho2_retrieved_target_data.pkl", "wb"))




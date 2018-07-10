"""
Script to query information about the science and calibrator stars.

Usage for astroquery can be found here:
    https://github.com/astropy/astroquery/blob/master/docs/vizier/vizier.rst
"""
from __future__ import division, print_function
import numpy as np
import pickle
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.table import Table
from collections import OrderedDict
from astropy.units import arcsec
from collections import Counter


# Import list of science and calibrator stars
targets_csv = "all_targets.csv"
all_targets = np.loadtxt(targets_csv, "string", delimiter=",")
all_target_ids = OrderedDict()

# -----------------------------------------------------------------------------
# Query for 2MASS, Tycho, HD IDs, plus Bayer Designation
# -----------------------------------------------------------------------------
# We want to get the 2MASS and Tycho2 IDs to ensure we are getting magnitudes
# for the right targets. The following queries Simbad for 2MASS and Tycho IDs,
# storing them in the dictionary all_target_ids, indexed by common name. If no
# key is found, the value "None" will be used.
missing_2mass = set()
missing_tycho = set()
missing_hd = set()
missing_bayer = set()

print("Getting 2MASS and Tycho IDs...")

for star in all_targets[:,3]:
    ids = Simbad.query_objectids(star)
       
    # Get IDs [2MASS, Tycho, HD, Bayer]
    all_target_ids[star] = [None, None, None, None]
    
    for id in ids:
        if "2MASS" in id[0]:
            all_target_ids[star][0] = id[0]
            
        elif "TYC" in id[0]:
            all_target_ids[star][1] = id[0]
            
        elif "HD" in id[0]:
            all_target_ids[star][2] = id[0].replace(" ", "")
        
        # Query for Bayer Designation (beginning with a single *), and 
        # preferring names that are alpha numeric. Remove leading * and any . 
        # before saving the ID. This level of specificity is to meet the ID
        # specifics of the JMMC Stellar Diameter Catalogue (JSDC) 
        elif ("* " == id[0][:2] and (not all_target_ids[star][3] 
            or not all_target_ids[star][3].split(" ")[1].isalpha())):
            all_target_ids[star][3] = id[0].replace("* ", "").replace(".", "")
            
    if not all_target_ids[star][0]:
        missing_2mass.add(star)
        
    if not all_target_ids[star][1]:
        missing_tycho.add(star)
        
    if not all_target_ids[star][2]:
        missing_hd.add(star)
        
    if not all_target_ids[star][3]:
        missing_bayer.add(star)

print("Missing 2MASS IDs: ", str(missing_2mass))
print("Missing Tycho IDs: ", str(missing_tycho))
print("Missing HD IDs: ", str(missing_hd))
print("Missing Bayer IDs: ", str(missing_bayer))

# -----------------------------------------------------------------------------
# Remove duplicate stars
# ----------------------------------------------------------------------------- 
# Some of our calibrators might be listed under multiple different names. Since
# all stars have HD numbers, use this to find and remove the duplicates
all_hd_ids = np.asarray(all_target_ids.values())[:,2]

duplicates = [id for id, num in Counter(all_hd_ids).items() if num > 1]

for dup_star in duplicates:
    all_target_ids.pop(dup_star.replace("D", "D "))

# -----------------------------------------------------------------------------
# Query Vizier for 2MASS and Tycho catalogue information
# -----------------------------------------------------------------------------            
# Catalogue IDs on Vizier
cat_id_2mass = "II/246"
cat_id_tycho2 = "I/259"

# Setup custom Vizier object to query all columns
vizier_all = Vizier(columns=["**"])

# Store all star data in dicts for easy querying with non-2MASS/Tycho IDs
dict_2mass = {}
dict_tycho = {}

# Search radius for Vizier query - without this it defaults to an arcmin and
# resolves more objects than it returns (i.e. it truncates to 50 objects) for 
# the 2MASS query. Due to a quirk in the way the radius query works, the Tycho 
# query requires a larger radius to resolve the primary target in the main 
# table (36 columns), rather than a supplementary table with fewer (22) columns
radius_1 = 20 * arcsec
radius_2 = 450 * arcsec

# Construct new tables to store all of the target data
table_2mass = vizier_all.query_object(all_targets[:,3][0], cat_id_2mass)[0]
table_2mass.remove_rows(slice(0, len(table_2mass)))

table_tycho = vizier_all.query_object(all_targets[:,3][0], cat_id_tycho2)[0]
table_tycho.remove_rows(slice(0, len(table_tycho)))

print("\nGetting 2MASS and Tycho catalogue entries...")

# For every target, query Vizier for the details. Select the targets based on
# their respective 2MASS and Tycho IDs
for star in all_target_ids.keys():
    print(star)
    # -------------------------------------------------------------------------
    # Get 2MASS results
    # -------------------------------------------------------------------------
    if star not in dict_2mass.keys() and all_target_ids[star][0]:
        results = vizier_all.query_object(star, cat_id_2mass, radius_1)[0]
        star_i = int(np.argwhere(results[:]["_2MASS"]
                                 ==all_target_ids[star][0].split("J")[1]))
        
        table_2mass.add_row(results[star_i])
        dict_2mass[star] = table_2mass[0]
    
    # Account for the fact that several of the stars do not have 2MASS data
    elif not all_target_ids[star][0]:
        dict_2mass[star] = None

    # -------------------------------------------------------------------------
    # Get Tycho-2 results
    # -------------------------------------------------------------------------
    if star not in dict_tycho.keys() and all_target_ids[star][1]:
        try:
            results = []
            results = vizier_all.query_object(star, cat_id_tycho2, radius_2)[0]
        
            for row_i, row in enumerate(results):
                # Tycho IDs are stored across 3 columns, have to string parse
                id_1 = str(row["TYC1"]) + str(row["TYC2"]) + str(row["TYC3"])
                id_2 = all_target_ids[star][1][3:].strip().replace("-", "")
            
                if id_1 == id_2:
                    table_tycho.add_row(results[row_i])
                    dict_tycho[star] = table_tycho[0]
                    
            #print(str(len(results[0])) + " -- " + star)
            
        except:
            if len(results) > 0:
                print(str(len(results[0])) + " -- " + star + " [Unresolved]")
            else:
                print(str(len(results)) + " -- " + star + " [Unresolved]")
    
    # Account for the fact that several of the stars do not have Tycho data   
    elif not all_target_ids[star][1]:
        dict_tycho[star] = None 
    
# Store the results
pkl_ids = open("target_data_ids.pkl", "wb")
pkl_2mass = open("target_data_2mass.pkl", "wb")
pkl_tycho = open("target_data_tycho2.pkl", "wb")

pickle.dump(all_target_ids, pkl_ids)
pickle.dump([dict_2mass, table_2mass], pkl_2mass)
pickle.dump([dict_tycho, table_tycho], pkl_tycho)

pkl_ids.close()
pkl_2mass.close()
pkl_tycho.close()
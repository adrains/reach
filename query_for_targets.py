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
missing_gaia_dr2 = set()

print("Getting 2MASS, Tycho, HD, Bayer, and Gaia DR2 IDs...")

for star in all_targets[:,3]:
    ids = Simbad.query_objectids(star)
       
    # Get IDs [2MASS, Tycho, HD, Bayer, Gaia]
    all_target_ids[star] = [None, None, None, None, None]
    
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
         
        elif "Gaia DR2" in id[0]:
            all_target_ids[star][4] = int(id[0].replace("Gaia DR2 ", ""))
            
    if not all_target_ids[star][0]:
        missing_2mass.add(star)
        
    if not all_target_ids[star][1]:
        missing_tycho.add(star)
        
    if not all_target_ids[star][2]:
        missing_hd.add(star)
        
    if not all_target_ids[star][3]:
        missing_bayer.add(star)
        
    if not all_target_ids[star][4]:
        missing_gaia_dr2.add(star)

print("Missing 2MASS IDs: ", str(missing_2mass))
print("Missing Tycho IDs: ", str(missing_tycho))
print("Missing HD IDs: ", str(missing_hd))
print("Missing Bayer IDs: ", str(missing_bayer))
print("Missing Gaia DR2 IDs: ", str(missing_gaia_dr2))

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
cat_id_gaia_dr2 = "I/345/gaia2"

# Setup custom Vizier object to query all columns
vizier_all = Vizier(columns=["**"], row_limit=2000)

# Store all star data in dicts for easy querying with non-2MASS/Tycho IDs
dict_2mass = {}
dict_tycho = {}
dict_gaia_dr2 = {}

# Search radius for Vizier query - without this it defaults to an arcmin and
# resolves more objects than it returns (i.e. it truncates to 50 objects) for 
# the 2MASS query. Due to a quirk in the way the radius query works, the Tycho 
# query requires a larger radius to resolve the primary target in the main 
# table (36 columns), rather than a supplementary table with fewer (22) columns
radius_1 = 20 * arcsec
radius_2 = 450 * arcsec
radius_3 = 105 * arcsec

# Construct new tables to store all of the target data
table_2mass = vizier_all.query_object(all_targets[:,3][0], cat_id_2mass)[0]
table_2mass.remove_rows(slice(0, len(table_2mass)))

table_tycho = vizier_all.query_object(all_targets[:,3][0], cat_id_tycho2)[0]
table_tycho.remove_rows(slice(0, len(table_tycho)))

table_gaia_dr2 = vizier_all.query_object(all_targets[:,3][0], cat_id_gaia_dr2)[0]
table_gaia_dr2.remove_rows(slice(0, len(table_gaia_dr2)))

print("\nGetting 2MASS, Tycho, and Gaia DR2 catalogue entries...")

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
        
    # -------------------------------------------------------------------------
    # Get Gaia DR2 Results
    # -------------------------------------------------------------------------
    if star not in dict_gaia_dr2.keys() and all_target_ids[star][4]:
        results = vizier_all.query_object(star, cat_id_gaia_dr2, radius_3)[0]
        star_i = int(np.argwhere(results[:]["Source"]
                                 ==all_target_ids[star][4]))
        
        table_gaia_dr2.add_row(results[star_i])
        dict_gaia_dr2[star] = table_gaia_dr2[0]
    
    # Account for the fact that several of the stars do not have 2MASS data
    elif not all_target_ids[star][4]:
        dict_gaia_dr2[star] = None
    
# Store the results
pkl_ids = open("target_data_ids.pkl", "wb")
pkl_2mass = open("target_data_2mass.pkl", "wb")
pkl_tycho = open("target_data_tycho2.pkl", "wb")
pkl_gaia_dr2 = open("target_data_gaia_dr2.pkl", "wb")

pickle.dump(all_target_ids, pkl_ids)
pickle.dump([dict_2mass, table_2mass], pkl_2mass)
pickle.dump([dict_tycho, table_tycho], pkl_tycho)
pickle.dump([dict_gaia_dr2, table_gaia_dr2], pkl_gaia_dr2)

pkl_ids.close()
pkl_2mass.close()
pkl_tycho.close()
pkl_gaia_dr2.close()

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
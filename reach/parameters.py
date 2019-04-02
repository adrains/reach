"""
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd

def sample_stellar_params(tgt_info, n_samples):
    """Sample stellar parameters for use with the bolometric correction code
    from Casagrande & VandenBerg (2014, 2018a, 2018b):
    
    https://github.com/casaluca/bolometric-corrections
    """
    loggs = []
    fehs = []
    teffs = []
    
    # Assign default errors to params
    tgt_info["e_logg"][np.logical_and(np.isnan(tgt_info["e_logg"]), 
                                      tgt_info["Science"])] = 0.1
    tgt_info["e_FeH_rel"][np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                                      tgt_info["Science"])] = 0.1
    tgt_info["e_teff"][np.logical_and(np.isnan(tgt_info["e_teff"]), 
                                      tgt_info["Science"])] = 100
    
    for star, row in tgt_info[tgt_info["Science"]].iterrows():
        loggs.append(np.random.normal(row["logg"], row["e_logg"], n_samples))
        fehs.append(np.random.normal(row["FeH_rel"], row["e_FeH_rel"], 
                                     n_samples))
        teffs.append(np.random.normal(row["Teff"], row["e_teff"], n_samples))
        
    params = np.vstack((np.array(loggs).flatten(), np.array(fehs).flatten(), 
                        np.array(teffs).flatten())).T
    
    np.savetxt("data/input.sample", params, delimiter=" ", 
               fmt=["%0.2f","%0.2f","%i"])
    
    return params
    
    
def sample_stellar_params_pd(tgt_info, n_bootstraps, 
                             assign_default_uncertainties=True):
    """Sample stellar parameters for use when calculating the limb-darkening
    coefficient.
    """
    # Make new dataframes for each stellar parameter
    ids = tgt_info[tgt_info["Science"]].index.values
    
    n_logg = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    n_teff = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    n_feh = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    
    if assign_default_uncertainties:
        # logg
        logg_mask = np.logical_and(np.isnan(tgt_info["e_logg"]), 
                                   tgt_info["Science"])
        tgt_info["e_logg"].where(~logg_mask, 0.2, inplace=True)
        
        # [Fe/H]
        feh_mask = np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                                  tgt_info["Science"])
        tgt_info["e_FeH_rel"].where(~feh_mask, 0.1, inplace=True)
        
        # Teff
        teff_mask = np.logical_and(np.isnan(tgt_info["e_teff"]), 
                                   tgt_info["Science"])
        tgt_info["e_teff"].where(~teff_mask, 100, inplace=True)      
    
    
    for id in ids:
        n_logg[id] = np.random.normal(tgt_info.loc[id, "logg"],
                                      tgt_info.loc[id, "e_logg"],
                                      n_bootstraps)   
        n_teff[id] = np.random.normal(tgt_info.loc[id, "Teff"],
                                      tgt_info.loc[id, "e_teff"],
                                      n_bootstraps) 
        n_feh[id] = np.random.normal(tgt_info.loc[id, "FeH_rel"],
                                      tgt_info.loc[id, "e_FeH_rel"],
                                      n_bootstraps)                                             
    return n_logg, n_teff, n_feh 


def save_params(tgt_info):
    """Save parameters to a text file with uncertainties
    """
    # logg
    logg_mask = np.logical_and(np.isnan(tgt_info["e_logg"]), 
                               tgt_info["Science"])
    tgt_info["e_logg"].where(~logg_mask, 0.2, inplace=True)
    
    # [Fe/H]
    feh_mask = np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                              tgt_info["Science"])
    tgt_info["e_FeH_rel"].where(~feh_mask, 0.1, inplace=True)
    
    # Teff
    teff_mask = np.logical_and(np.isnan(tgt_info["e_teff"]), 
                               tgt_info["Science"])
    tgt_info["e_teff"].where(~teff_mask, 100, inplace=True)    
    
    # Save
    path = "white_ld/pionier_targets_new.txt"
    cols = ["Primary", "Teff", "e_teff", "logg", "e_logg", "FeH_rel", 
            "e_FeH_rel"]
    tgt_info[cols][tgt_info["Science"]].to_csv(path, sep="\t",index=False)  
        


def combine_seq_ldd(tgt_info, results):
    """Combine independent measures of LDD from multiple different sequences to
    a single measurement of LDD +/- e_LDD
    """
    stars = set(results["HD"])
    
    tgt_info["ldd_final"] = np.zeros(len(tgt_info))
    tgt_info["e_ldd_final"] = np.zeros(len(tgt_info))
    
    # For every star, do a weighted average of the angular diameters, with 
    # weights equal to the inverse variance. 
    for star_i, star in enumerate(stars):
        weights = results[results["HD"]==star]["e_LDD_FIT"].values**(-2)
        ldd_avg = np.average(results[results["HD"]==star]["LDD_FIT"], 
                             weights=weights)
        e_ldd_avg = (np.sum(weights)**-1)**0.5
        
        # Save the final values
        tgt_info.loc[star, "ldd_final"] = ldd_avg
        tgt_info.loc[star, "e_ldd_final"] = e_ldd_avg
        

def calc_all_f_bol(tgt_info, n_samples):
    """f_bol in ergs s^-1 cm^-2
    """
    # Import in the sampled bolometric corrections
    bcs = np.loadtxt("data/bc_science.data", skiprows=1)
    
    # Reshape the n_samples bolometric corrections per star
    n_science = len(tgt_info[tgt_info["Science"]])
    n_filt = bcs.shape[-1]
    
    bcs = np.reshape(bcs, (n_science, n_samples, n_filt))
    
    # Define bands to reference, construct new headers
    bands = ["Hpmag", "BTmag", "VTmag", "BPmag", "RPmag"]
    e_bands = ["e_%s" % band for band in bands] 
    f_bol_bands = ["f_bol_%s" % band for band in bands] 
    e_f_bol_bands = ["e_f_bol_%s" % band for band in bands] 
    
    for band in bands:
        tgt_info["f_bol_%s" % band] = np.zeros(len(tgt_info))
        tgt_info["e_f_bol_%s" % band] = np.zeros(len(tgt_info))
        
    # And the averaged fbol value
    tgt_info["f_bol_final"] = np.zeros(len(tgt_info))
    tgt_info["e_f_bol_final"] = np.zeros(len(tgt_info))
    
    # Calculate bolometric fluxes for each band for every star
    for star_i, (star, row) in enumerate(tgt_info[tgt_info["Science"]].iterrows()):
        print(star)
        for band_i, (band, e_band) in enumerate(zip(bands, e_bands)):
            # Sample the magnitudes
            mags = np.random.normal(row[band], row[e_band], n_samples) 
            
            # Calculate the bolometric flux
            f_bols = calc_f_bol(bcs[star_i, :, band_i], mags)
            
            f_bol = np.mean(f_bols)
            e_f_bol = np.std(f_bols)
            
            tgt_info.loc[star, f_bol_bands[band_i]] = f_bol
            tgt_info.loc[star, e_f_bol_bands[band_i]] = e_f_bol
            
    # Now use a weighted average to work out fbol, using the reciprocal of
    # the variance as weights
    for star_i, (star, row) in enumerate(tgt_info[tgt_info["Science"]].iterrows()):
        weights = row[e_f_bol_bands][row[e_f_bol_bands] > 0].values**(-2)
        f_bol_avg = np.average(row[f_bol_bands][row[f_bol_bands] > 0].values, 
                               weights=weights)
        e_f_bol_avg = (np.sum(weights)**-1)**0.5
        
        tgt_info.loc[star, "f_bol_final"] = f_bol_avg
        tgt_info.loc[star, "e_f_bol_final"] = e_f_bol_avg
        

def calc_all_r_star(tgt_info):
    """
    """
    # Constants
    pc = 3.0857*10**13 # km / pc
    r_sun = 6.957 *10**5 # km
    
    # Compute the physical radii
    for star, row in tgt_info[np.logical_and(tgt_info["Science"], 
                              tgt_info["ldd_final"] > 0)].iterrows():
        # Convert to km and radians
        dist_km = row["Dist"] * pc
        e_dist_km = row["e_Dist"] * pc
        ldd_rad = row["ldd_final"] * np.pi/180/3600/1000
        e_ldd_rad = row["e_ldd_final"] * np.pi/180/3600/1000
        
        # Calculate the stellar radii
        r_star = 0.5 * ldd_rad * dist_km / r_sun
        e_r_star = r_star * ((e_ldd_rad/ldd_rad)**2
                             + (e_dist_km/dist_km)**2)**0.5
    
        tgt_info.loc[star, "r_star_final"] = r_star
        tgt_info.loc[star, "e_r_star_final"] = e_r_star        
    
    
def calc_all_teff(tgt_info, n_samples):
    """Calculate the effective temperature for all stars
    """ 
    # Stefan-Boltzmann constant
    sigma = 5.6704 * 10**-5 #erg cm^-2 s^-1 K^-4
    
    # Define bands to reference, construct new headers
    bands = ["Hpmag", "BTmag", "VTmag", "BPmag", "RPmag"]
    
    for band in bands:
        tgt_info["teff_%s" % band] = np.zeros(len(tgt_info))
        tgt_info["e_teff_%s" % band] = np.zeros(len(tgt_info))
    
    # And the averaged fbol value
    tgt_info["teff_final"] = np.zeros(len(tgt_info))
    tgt_info["e_teff_final"] = np.zeros(len(tgt_info))
    
    # Calculate the Teff for every star using an MC sampling approach         
    for star, row in tgt_info[np.logical_and(tgt_info["Science"], 
                              tgt_info["ldd_final"] > 0)].iterrows():
        # Sample the diameters
        ldds = np.random.normal(row["ldd_final"], row["e_ldd_final"], n_samples)
        ldds = ldds * np.pi/180/3600/1000
        
        # Sample fbol
        f_bols = np.random.normal(row["f_bol_final"], row["e_f_bol_final"], n_samples)
        
        # Calculate Teff
        teffs = (4*f_bols / (sigma * ldds**2))**0.25 
        
        # Calculate final teff and error
        teff = np.mean(teffs)
        e_teff = np.std(teffs)
        
        # Store final value
        tgt_info.loc[star, "teff_final"] = teff
        tgt_info.loc[star, "e_teff_final"] = e_teff


def calc_all_L_bol(tgt_info, n_samples):
    """
    """
    # Constants
    L_sun = 3.839 * 10**33 # erg s^-1
    pc = 3.0857*10**18 # cm / pc
    
    # Initialise L_star column
    tgt_info["L_star_final"] = np.zeros(len(tgt_info))
    tgt_info["e_L_star_final"] = np.zeros(len(tgt_info))
    
    # Calculate the Teff for every star using an MC sampling approach         
    for star, row in tgt_info[np.logical_and(tgt_info["Science"], 
                              tgt_info["ldd_final"] > 0)].iterrows():
        # Sample fbol
        f_bols = np.random.normal(row["f_bol_final"], row["e_f_bol_final"], n_samples)
        
        # Sample distances
        dists = np.random.normal(row["Dist"], row["e_Dist"], n_samples) * pc
        
        # Calculate luminosities
        L_stars = 4 * np.pi * f_bols * dists**2
        
        # Calculate final L_star (in solar units) and error
        L_star = np.mean(L_stars) / L_sun
        e_L_star = np.std(L_stars) / L_sun
        
        # Store final value
        tgt_info.loc[star, "L_star_final"] = L_star
        tgt_info.loc[star, "e_L_star_final"] = e_L_star
        
  
def print_mean_flux_errors(tgt_info):
    """
    """
    bands = ["Hp", "BT", "VT", "BP", "RP"]
    f_bols = ["f_bol_Hpmag", "f_bol_BTmag", "f_bol_VTmag", "f_bol_BPmag", 
             "f_bol_RPmag"]
    e_f_bols = ["e_f_bol_Hpmag", "e_f_bol_BTmag", "e_f_bol_VTmag", 
               "e_f_bol_BPmag", "e_f_bol_RPmag"]       
     
    print("Band | % err")           
    for f_i in np.arange(0, len(f_bols)):
        med_e_f_bol = (tgt_info[e_f_bols[f_i]][tgt_info["Science"]]
                       / tgt_info[f_bols[f_i]][tgt_info["Science"]]).median()
        print("%s --- %0.2f" % (bands[f_i], med_e_f_bol*100))         
    
    # For the averaged Fbol
    med_e_f_bol = (tgt_info["e_f_bol_avg"][tgt_info["Science"]]
                       / tgt_info["f_bol_avg"][tgt_info["Science"]]).median()
    print("\nAVG --- %0.2f" % (med_e_f_bol*100))

def calc_f_bol(bc, mag):
    """
    """
    L_sun = 3.839 * 10**33 # erg s^-1
    au = 1.495978707*10**13 # cm
    M_bol_sun = 4.75
    
    exp = -0.4 * (bc - M_bol_sun + mag - 10)
    
    f_bol = (np.pi * L_sun / (1.296 * 10**9 * au)**2) * 10**exp
    
    return f_bol      
    

def calc_L_star(tgt_info):
    """
    """
    L_sun = 3.839 * 10**33 # erg s^-1
    au = 1.495978707*10**13 # cm
    M_bol_sun = 4.75
    
    tgt_info["L_star"] = 10**(-0.4 * (tgt_info["M_bol"] - M_bol_sun))
    
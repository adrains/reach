"""
"""
from __future__ import division, print_function
import numpy as np

def sample_stellar_params(tgt_info, n_samples):
    """
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
        fehs.append(np.random.normal(row["FeH_rel"], row["e_FeH_rel"], n_samples))
        teffs.append(np.random.normal(row["Teff"], row["e_teff"], n_samples))
        
    params = np.vstack((np.array(loggs).flatten(), np.array(fehs).flatten(), 
                        np.array(teffs).flatten())).T
    
    np.savetxt("data/input.sample", params, delimiter=" ", fmt=["%0.2f","%0.2f","%i"])


def calc_teff_from_bc(tgt_info, results, n_samples):
    """
    """
    # Stefan-Boltzmann constant
    sigma = 5.6704 * 10**-5 #erg cm^-2 s^-1 K^-4
    
    # Import in the sampled bolometric corrections
    #bcs = pd.read_csv("data/bc_science.data", header=0, delim_whitespace=True)
    bcs = np.loadtxt("data/bc_science.data", skiprows=1)[:, 1:]
    
    n_science = len(tgt_info[tgt_info["Science"]])
    n_filt = bcs.shape[-1]
    
    bcs = np.reshape(bcs, (n_science, n_samples, n_filt))
    
    bcs_mean = bcs.mean(axis=1)
    bcs_std = bcs.std(axis=1)
    
    #bands = ["Hpmag", "BTmag", "VTmag", "BPmag", "RP_mag"]
    #e_bands = ["e_Hpmag", "e_BTmag", "e_VTmag", "e_BPmag", "e_RP_mag"]
    bands = ["BTmag", "VTmag", "BPmag", "RPmag"]
    e_bands = ["e_BTmag", "e_VTmag", "e_BPmag", "e_RPmag"]
    
    for band in bands:
        results["Teff_%s" % band] = np.zeros(len(results))
        results["e_Teff_%s" % band] = np.zeros(len(results))
        tgt_info["f_bol_%s" % band] = np.zeros(len(tgt_info))
        tgt_info["e_f_bol_%s" % band] = np.zeros(len(tgt_info))
    
    # Go through every star
    for star_i, (star, row) in enumerate(tgt_info[tgt_info["Science"]].iterrows()):
        print(star)
        for band_i, (band, e_band) in enumerate(zip(bands, e_bands)):
            # Sample the magnitudes
            mags = np.random.normal(row[band], row[e_band], n_samples) 
            
            # Calculate the bolometric flux
            f_bols = calc_f_bol(bcs[star_i, :, band_i], mags)
            
            f_bol = np.mean(f_bols)
            e_f_bol = np.std(f_bols)
            
            tgt_info.loc[star, "f_bol_%s" % band] = f_bol
            tgt_info.loc[star, "e_f_bol_%s" % band] = e_f_bol
            
            # Calculate Teff for each result for this star
            for res, res_row in results[results["HD"]==star].iterrows():
                # Sample LDD
                ldds = np.random.normal(res_row["LDD_FIT"], res_row["e_LDD_FIT"], n_samples)
                ldds = ldds * np.pi/180/3600/1000
                
                # Calculate Teff
                teffs = (4*f_bols / (sigma * ldds**2))**0.25 
                
                # Calculate final teff and error
                teff = np.mean(teffs)
                e_teff = np.std(teffs)
                
                results.loc[res, "Teff_%s" % band] = teff
                results.loc[res, "e_Teff_%s" % band] = e_teff
                
    

def calc_f_bol(bc, mag):
    """
    """
    L_sun = 3.839 * 10**33 # erg s^-1
    au = 1.495978707*10**13 # cm
    M_bol_sun = 4.75
    
    exp = -0.4 * (bc - M_bol_sun + mag - 10)
    
    f_bol = (np.pi * L_sun / (1.296 * 10**9 * au)**2) * 10**exp
    
    return f_bol
    
    
        
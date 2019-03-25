"""Module for angular diameter prediction, calculation, and vis^2 fitting
"""
from __future__ import division, print_function
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import Counter
from astropy.io import fits
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from scipy.odr import ODR, Model, Data, RealData


class UnknownOIFitsFileFormat(Exception):
    pass

# -----------------------------------------------------------------------------
# Predicting LDD
# -----------------------------------------------------------------------------
def predict_ldd_boyajian(F1_mag, F1_mag_err, F2_mag, F2_mag_err, colour=None,
                         colour_rel="V-W3"):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Boyajian et al. 2014:
     - http://adsabs.harvard.edu/abs/2014AJ....147...47B
        
    Parameters
    ----------
    F1_mag: float
        Magnitude of the star in the first filter.
    F1_mag_err: float
        Error in magnitude of the star in the first filter.
    F2_mag: float
        Magnitude of the star in the second filter.
    F2_mag_err: float
        Error in magnitude of the star in the second filter.
    colour_rel: string
        The colour relation to use for predicting the LDD (i.e. V-K)
        
    Returns
    -------
    ldd: float
        The predicted limb darkened angular diameter
    """
    # Convert to numpy arrays
    F1_mag = np.array(F1_mag)
    F1_mag_err = np.array(F1_mag_err)
    F2_mag = np.array(F2_mag)
    F2_mag_err = np.array(F2_mag_err)
    
    # Determine whether we have been provided with a pre-existing colour, and
    # if not compute it. If given a colour, we will use that instead.
    if colour is None:
        colour = F1_mag - F2_mag
    
    num_exponents = 4
    
    # Import the Boyajian relations, columns are:
    # [Colour Index, Num Points, Range (mag), a0, a0_err, a1, a1_err,
    #  a2, a2_err, a3, a3_err, Reduced chi^2, err (%)]
    dd = os.path.dirname(__file__)[:-5]
    boyajian_2014_rel_file = os.path.join(dd, 
                            "data/boyajian_2014_colour_diameter_relations.csv")
    
    diam_rel = np.loadtxt(boyajian_2014_rel_file, delimiter=",", 
                          skiprows=1, dtype="str")
    # Create dictionary for coefficients and for error on relation
    diam_rel_coeff = {}
    diam_rel_err = {}
    
    for rel in diam_rel:
        diam_rel_coeff[rel[0]] = rel[3:11].astype(float)
        diam_rel_err[rel[0]] = rel[-1].astype(float)
    
    # Calculate diameters
    # Relationship is log_diam = Sigma(i=0) a_i * (F1_mag-F2_mag) ** i
    exponents = np.arange(0, num_exponents)
    
    # Repeat along new direction so operation is vectorised
    colour = np.repeat(np.array(colour)[:,None], num_exponents, 1)
    
    log_diam = np.sum(diam_rel_coeff[colour_rel][::2]*colour**exponents, 1)
    
    ldd = 10**(-0.2*F1_mag) * 10**log_diam
    
    # Calculate the error. Currently this is done solely using the percentage
    # error given in the paper, and does not treat errors in either magnitude
    e_ldd = ldd * diam_rel_err[colour_rel]/100
    
    return ldd, e_ldd
    
    
def predict_ldd_kervella(V_mag, V_mag_err, K_mag, K_mag_err):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Kervella et al. 2004:
     - http://adsabs.harvard.edu/abs/2004A%26A...426..297K
    """
    # Calculate the LDD
    log_ldd = 0.0755 * (V_mag - K_mag) + 0.5170 - 0.2 * K_mag

    ldd = 10**log_ldd

    # Calculate the error on this value (assuming no covariance)
    log_ldd_err = np.sqrt((0.0755*V_mag_err)**2 + (0.2755*K_mag_err)**2)

    ldd_err = 10**(np.log(10)*log_ldd_err)
                                       
    return log_ldd, log_ldd_err, ldd, ldd_err   


def predict_all_ldd(tgt_info):
    """
    
    Given 2MASS is saturated, there are three colour relations available:
     1 - V-W3 (where V is converted from Vt through Bessell 2000 relations)
     2 - V-W4 (where V is converted from Vt through Bessell 2000 relations)
     3 - V-K (where V-K is computed using Vt-Rp/V-K relation)
     
    This is also the order of preference for the relations, with most stars
    having V-W3. V-W4 is only used for stars with saturated W3, and V-K for
    those stars without ALLWISE data.
    """
    # V-K from Vt-Rp relation
    ldd_vk, e_ldd_vk = predict_ldd_boyajian(tgt_info["Vmag_dr"],
                                            tgt_info["e_VTmag"], None, None,
                                            tgt_info["V-K_calc"], "V-K")
    # V-W3                                              
    ldd_vw3, e_ldd_vw3 = predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                              tgt_info["e_VTmag"], 
                                              tgt_info["W3mag"], 
                                              tgt_info["e_W3mag"], None, 
                                              "V-W3") 
    # V-W4                                                
    ldd_vw4, e_ldd_vw4 = predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                              tgt_info["e_VTmag"], 
                                              tgt_info["W4mag"], 
                                              tgt_info["e_W4mag"], None, 
                                              "V-W4") 
    
    # Save these values
    tgt_info["LDD_VK"] = ldd_vk
    tgt_info["e_LDD_VK"] = e_ldd_vk

    tgt_info["LDD_VW3"] = ldd_vw3
    tgt_info["e_LDD_VW3"] = e_ldd_vw3
    
    tgt_info["LDD_VW4"] = ldd_vw4
    tgt_info["e_LDD_VW4"] = e_ldd_vw4
    
    ldd_pred = []
    e_ldd_pred = []
    
    # Save the 
    for star, star_data in tgt_info.iterrows():
        ldd_rel = star_data["LDD_rel"]
        
        if type(ldd_rel) is str:
            ldd_pred.append(star_data[ldd_rel])
            e_ldd_pred.append(star_data["e_%s" % ldd_rel])
        
        # For those stars without a relation (entirely bad calibrators that 
        # will be excluded), assign placeholder diameters so the PIONIER
        # pipeline is happy    
        else:
            ldd_pred.append(1.0)
            e_ldd_pred.append(0.1)
            
    tgt_info["LDD_pred"] = ldd_pred
    tgt_info["e_LDD_pred"] = e_ldd_pred
        
     
# -----------------------------------------------------------------------------
# Fitting LDD
# -----------------------------------------------------------------------------
def calc_vis2_ls(b_on_lambda, ldd, c_scale, u_lld):
    """Calculates squared fringe visibility assuming a linearly limb-darkened 
    disk. As outlined in Hanbury Brown et al. 1974: 
     - http://adsabs.harvard.edu/abs/1974MNRAS.167..475H
        
    This function is called from fit_for_ldd using scipy.optimize.curve_fit.
        
    Parameters
    ----------
    b_on_lambda: float or float array
        The projected baseline B (m), divided by the wavelength lambda (m)
    
    ldd: float or float array
        The limb-darkened angular diameter (mas)
        
    c_scale: float
        Scaling parameter to not force the fit to be anchored at 1.
    
    u_lld: float or float array
        The wavelength dependent linear limb-darkening coefficient
        
    Returns
    -------
    vis2: float or float array
        Calibrated squared fringe visibility
    """
    # Calculate x and convert ldd to radians (this serves two purposes: making
    # the input/output more human readable, and the fitting function performs
    # better
    x = np.pi * b_on_lambda * (ldd / 1000 / 3600 / 180 * np.pi)
    
    vis = (((1 - u_lld)/2 + u_lld/3)**-1 * 
          ((1 - u_lld)*jv(1,x)/x + u_lld*(np.pi/2)**0.5 * jv(3/2,x)/x**(3/2)))
    
    # Square visibility and return       
    return c_scale * vis**2


def calc_vis2_odr(beta, b_on_lambda):
    """Calculates squared fringe visibility assuming a linearly limb-darkened 
    disk. As outlined in Hanbury Brown et al. 1974: 
     - http://adsabs.harvard.edu/abs/1974MNRAS.167..475H
        
    This function is called from fit_for_ldd using scipy.odr.ODR.
        
    Parameters
    ----------
    beta: tuple
        Tuple containing ldd, c_scale, and u_lld
        
    b_on_lambda: float or float array
        The projected baseline B (m), divided by the wavelength lambda (m)
        
    Returns
    -------
    vis2: float or float array
        Calibrated squared fringe visibility
    """
    # Unpack
    ldd, c_scale, u_lld = beta
    
    # Calculate x and convert ldd to radians (this serves two purposes: making
    # the input/output more human readable, and the fitting function performs
    # better
    x = np.pi * b_on_lambda * (ldd / 1000 / 3600 / 180 * np.pi)
    
    vis = (((1 - u_lld)/2 + u_lld/3)**-1 * 
          ((1 - u_lld)*jv(1,x)/x + u_lld*(np.pi/2)**0.5 * jv(3/2,x)/x**(3/2)))
    
    # Square visibility and return       
    return c_scale * vis**2
          
          
def fit_for_ldd(vis2, e_vis2, baselines, wavelengths, u_lld, ldd_pred,
                method="odr", e_wl_frac=0.02):
    """Fit to calibrated squared visibilities to obtain the measured limb-
    darkened stellar diameter in mas.
    
    Lambda function per:
     - https://stackoverflow.com/questions/12208634/fitting-only-one-
        parameter-of-a-function-with-many-parameters-in-python
        
    Parameters
    ----------
    vis2: float array
        Calibrated squared visibiity measurements
        
    e_vis2: float array
        Error on the calibrated squared visibility measurements
        
    baseline: float array
        Projected interferometric baselines (m)
        
    wavelengths: float array
        Wavelengths the observations were taken at (m)
        
    u_lld: float
         Wavelength dependent linear limb-darkening coefficient 
         
    ldd_pred: float
        Predicted limb-darkened stellar angular diameter (mas)
        
    Returns
    -------
    popt: float
        Optimal values for limb-darkened diameter (mas) and scaling param C
        
    pcov: float
        Errors (one standard deviation) oon LDD and C
    """
    # Baseline/lambda should have dimensions [B,W], where B is the number of 
    # baselines, and W is the number of wavelengths
    n_bl = len(baselines)
    n_wl = len(wavelengths)
    bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
    wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
    b_on_lambda = (bl_grid / wl_grid).flatten()
    
    # Initial C param
    c_scale = 1
    
    # Don't consider bad data during fitting process
    valid_i = (vis2.flatten() >= 0) & (e_vis2.flatten() > 0) & (~np.isnan(vis2.flatten()))
    
    # Fit for LDD. The lambda function means that we can fix u_lld and not have
    # to optimise for it too. Loose, but physically realistic bounds on LDD for
    # science targets (LDD cannot be zero else the fitting/formula will fail) 
    if method == "ls":
        popt, pcov = curve_fit((lambda b_on_lambda, ldd_pred, c_scale: 
                                calc_vis2_ls(b_on_lambda, ldd_pred, c_scale, 
                                           u_lld)), 
                                b_on_lambda[valid_i], vis2.flatten()[valid_i], 
                                sigma=e_vis2.flatten()[valid_i], 
                                bounds=(0.1, (10, 2)))

        return popt, [np.sqrt(np.diag(pcov_i)) for pcov_i in pcov] 
    
    # Run instead using Orthogonal Distance Regression so we can have 
    # uncertainties on the wavelength calibration    
    elif method == "odr":
        data = RealData(b_on_lambda[valid_i], vis2.flatten()[valid_i], 
                        e_vis2.flatten()[valid_i], 
                        e_wl_frac*b_on_lambda[valid_i])
        model = Model(calc_vis2_odr)
        odr = ODR(data, model, [ldd_pred, c_scale, u_lld], ifixb=[1,1,0])
        odr.set_job(fit_type=2)
        output = odr.run()   
        
        return output.beta, output.sd_beta   


def fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, pred_ldd_col,
                u_lld, method="odr"):
    """Fits limb-darkened diameters to all science targets using all available
    vis^2, e_vis^2, and projected baseline data.
    
    Parameters
    ----------
    vis2: dict
        Dictionary mapping science target ID to all vis^2 values
    
    e_vis2: dict
        Dictionary mapping science target ID to all e_vis^2 values
        
    baselines: dict
        Dictionary mapping science target ID to all projected baselines (m)
    
    wavelengths: list
        List recording the wavelengths observed at (m)
    
    Returns
    -------
    successful_fits: list
        List containing ldd_opt, e_ldd_opt, c_scale, e_c_scale.
    """
    successful_fits = {}
    #print("\n", "-"*79, "\n", "\tFitting for LDD\n", "-"*79)
    for sci in vis2.keys():
        # Only take the ID part of sci - could have " (Sequence)" after it
        sci_data = tgt_info[tgt_info["Primary"]==sci.split(" ")[0]]
        id = sci_data.index.values[0]
        
        if not sci_data["Science"].values:
            print("%s is not science target, aborting fit" % sci)
            continue
        else:
            print("\tFitting linear LDD to %s" % sci, end="")
        
        popt, pstd = fit_for_ldd(vis2[sci], e_vis2[sci], 
                                 baselines[sci], wavelengths[sci], 
                                 u_lld[id], 
                                 sci_data[pred_ldd_col].values[0], 
                                 method=method)
        print("...fit successful")
        
        # Extract parameters from fit
        ldd_opt = popt[0]
        c_scale = popt[1]
    
        e_ldd_opt = pstd[0]
        e_c_scale = pstd[1]
        
        successful_fits[sci] = [ldd_opt, e_ldd_opt, c_scale, e_c_scale]                          
            
    return successful_fits


def extract_vis2(oi_fits_file):
    """Read the calibrated squared visibility + errors, baseline, and 
    wavelength information from a given oifits file.
    
    Parameters
    ----------
    oi_fits_file: string
        Filepath to the oifits file
        
    Returns
    -------
    mjds: float array
        MJDs of the observations.
    
    pairs: string array
        Telescope pairs for each each baseline.
    
    vis2: float array
        Calibrated squared visibiity measurements
        
    e_vis2: float array
        Error on the calibrated squared visibility measurements
    
    flags: float array
        Quality flags for each observation.
        
    baseline: float array
        Projected interferometric baselines (m)
        
    wavelengths: float array
        Wavelengths the observations were taken at (m)
    """
    # The format of the oiFits file varies depending on how many CAL-SCI
    # CAL-SCI-CAL-SCI-CAL sequences were observed in the same night. For a
    # single sequence, the fits extensions are as follows:
    # [imageHDU, target info, wavelengths, telescopes, vis^2, t3phi]
    # When multiple sequences are observed, there are extra extensions for
    # each wavelength, vis^2, and t3phi set (e.g. two observed sequences 
    # would have two of each of these in a row)
    with fits.open(oi_fits_file, memmap=False) as oifits:
        #oifits = fits.open(oi_fits_file)
        n_extra_seq = (len(oifits) - 6) // 3
        
        mjds = []
        pairs = []
        vis2 = []
        e_vis2 = []
        flags = []
        baselines = []
        wavelengths = []
    
        # Retrieve visibility and baseline information for an arbitrary (>=1) 
        # number of sequences within a given night
        for seq_i in xrange(0, n_extra_seq+1):
            oidata = oifits[4 + n_extra_seq + seq_i].data
            
            # Figure out how large the chunk is. For the majority of cases
            # there will be two different MJDs (barring any weird sequences or
            # dropped baselines). Want to figure out how many of the maximum
            # 6 baselines per observation are available. If we don't have six
            # baselines, we need to insert an empty placeholder set to keep the
            # ordering and ensure we can compute means/standard deviations 
            # later.
            
            mjd_counts = Counter(oidata["MJD"])
            unique_mjds = list(set(oidata["MJD"]))
            unique_mjds.sort()
            n_1st_mjd = mjd_counts[unique_mjds[0]]
            
            # Sometimes the MJDs are split trivially in time (< 1 minute),
            # which splits the baselines into chunks smaller than 6. Science
            # observations actually split in time are actually 15 mins+ apart.
            # Thus we want to count any close in time as occurring at the same
            # time.
            for mjd in unique_mjds[1:]:
                if ((mjd - unique_mjds[0]) * 24 * 60) < 5: # < 5 mins in time
                    n_1st_mjd += mjd_counts[mjd]
            
            expected_pairs = set(["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"])
            
            observed_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"][:n_1st_mjd]])
            
            # Grab the relevant info prior to modification
            mjds_obs = oidata["MJD"]
            pairs_obs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"]])
            vis2_obs = oidata["VIS2DATA"]
            e_vis2_obs = oidata["VIS2ERR"]
            flags_obs = oidata["FLAG"]
            baselines_obs = np.sqrt(oidata["UCOORD"]**2 + oidata["VCOORD"]**2)
            
            # For every missing baseline, insert dummy NaN data to keep array
            # dimensions the same
            for missing_bl in list(expected_pairs - set(observed_pairs)):
                print("\tAdding missing info for first science block on %s" 
                      % oi_fits_file)
                mjds_obs = np.insert(mjds_obs, n_1st_mjd, np.nan)
                pairs_obs = np.insert(pairs_obs, n_1st_mjd, missing_bl)
                vis2_obs = np.insert(vis2_obs, n_1st_mjd, [np.nan]*6, axis=0)
                e_vis2_obs = np.insert(e_vis2_obs, n_1st_mjd, [np.nan]*6, 
                                       axis=0)
                flags_obs = np.insert(flags_obs, n_1st_mjd, [np.nan]*6, 
                                      axis=0)
                baselines_obs = np.insert(baselines_obs, n_1st_mjd, np.nan, 
                                          axis=0)
                
            # Now do this again for the other expected observation
            observed_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"][6:]])
            
            for missing_bl in list(expected_pairs - set(observed_pairs)):
                print("\tAdding missing info for first science block on %s" 
                      % oi_fits_file)
                mjds_obs = np.insert(mjds_obs, 6, np.nan)
                pairs_obs = np.insert(pairs_obs, 6, missing_bl)
                vis2_obs = np.insert(vis2_obs, 6, [np.nan]*6, axis=0)
                e_vis2_obs = np.insert(e_vis2_obs, 6, [np.nan]*6, axis=0)
                flags_obs = np.insert(flags_obs, 6, [np.nan]*6, axis=0)
                baselines_obs = np.insert(baselines_obs, 6, np.nan, axis=0)                      
            
            # Sort baselines within each observation (chunk of 6) to ensure 
            # ordering is the same for bootstrapping. Given there are two 
            # observations of each science target within the 
            # CAL1-SCI1-CAL2-SCI2-CAL3 sequence, there will be 2 sets of six
            # per sequence. To simplify the sorting procedure, convert the 
            # tuple pairs of telescope IDs to a string.
            #tel_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  #for tel in oidata["STA_INDEX"]])                  
                                  
            order = np.concatenate((pairs_obs[:6].argsort(), 
                                    pairs_obs[6:].argsort() + 6))
            
            # New solution to keep sequences separate is to simply append them
            # to the list as they come in, which per the oifits standard are
            # already sorted in time
            mjds.append(mjds_obs[order])
            pairs.append(pairs_obs[order])
            vis2.append(vis2_obs[order])
            e_vis2.append(e_vis2_obs[order])
            flags.append(flags_obs[order])
            baselines.append(baselines_obs[order])
            
            """
            if (len(mjds)==0 and len(pairs)==0 and len(vis2)==0 
                and len(e_vis2)==0 and len(flags)==0 and len(baselines)==0):
                # Arrays are empty
                mjds = mjds_obs[order]
                pairs = pairs_obs[order]
                vis2 = vis2_obs[order]
                e_vis2 = e_vis2_obs[order]
                flags = flags_obs[order]
                baselines = baselines_obs[order]
                #wavelengths = oifits[2].data["EFF_WAVE"]
                
            else:
                # Not empty, stack
                mjds = np.hstack((mjds, mjds_obs[order]))
                pairs = np.hstack((pairs, pairs_obs[order]))
                vis2 = np.vstack((vis2, vis2_obs[order]))
                e_vis2 = np.vstack((e_vis2, e_vis2_obs[order]))
                flags = np.vstack((flags, flags_obs[order]))
                baselines = np.hstack((baselines, baselines_obs[order]))
                #wavelengths = np.vstack((wavelengths, 
                                         #oifits[2].data["EFF_WAVE"])
            """
        # Assume that we'll always be using same wavelength mode within a night      
        wavelengths = oifits[2].data["EFF_WAVE"]
    
    return mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths


def collate_vis2_from_file(results_path, bs_i=None, separate_sequences=False):
    """Collates calibrated squared visibilities, errors, baselines, and 
    wavelengths for each science target in the specified results folder.
    
    Parameters
    ----------
    results_path: string
        Directory where the calibrated oifits results files are stored.
        
    Returns
    -------
    all_mjds: dict
        Dictionary mapping science target ID to all observation MJDs (times).
    
    all_tel_pairs: dict
         Dictionary mapping science target ID to all telescope pairs.
    
    all_vis2: dict
        Dictionary mapping science target ID to all vis^2 values
    
    all_e_vis2: dict
        Dictionary mapping science target ID to all e_vis^2 values
        
    all_flags: dict    
         Dictionary mapping science target ID to all quality flags.
        
    all_baselines: dict
        Dictionary mapping science target ID to all projected baselines (m)
    
    wavelengths: list
        List recording the wavelengths observed at (m)
    """
    # Initialise data structures to store calibrated results, where dict keys
    # are the science target IDs. Note that the wavelengths are common to all.
    all_mjds = {}
    all_tel_pairs = {}
    all_vis2 = {}
    all_e_vis2 = {}
    all_flags = {}
    all_baselines = {}
    all_wavelengths = {}
    
    ith_bs_oifits = glob.glob(results_path 
                              + "*SCI*oidataCalibrated_%02i.fits" % bs_i)
    ith_bs_oifits.sort()
    
    if separate_sequences:
        # We want to keep the bright and faint sequences separate for 
        # diagnostic purposes, but still need to collate in the instance
        # that a star as duplicate sequences
        dates_obs = pd.read_csv("data/dates_observed.tsv", sep="\t")                     
    
    print("\nFound %i oifits file/s for bootstrap %i" % (len(ith_bs_oifits), 
                                                       bs_i+1))
    
    for oifits in ith_bs_oifits:
        # Get the target name from the file name - this is clunky, but more
        # robust than the former method of slicing using static indices which
        # inherently assumes a constant file length (which changes when we
        # begin bootstrapping)
        sci = oifits.split("SCI")[1].split("oidata")[0].replace("_", "")
        
        # Extract data from oifits file. If multiple sequences were observed
        # on the same night, each of the retuned lists will contain more than
        # one list of results
        mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            extract_vis2(oifits)
        
        for seq_i in np.arange(0, len(mjds)):
            # Initialise 
            seq_id = sci
        
            # If keeping separate sequences, record bright/faint and period in key
            if separate_sequences:
            
                night = oifits.split("/")[-1].split("_SCI")[0]
            
                faint_entry = dates_obs[np.logical_and(dates_obs["star"]==sci, 
                                                       dates_obs["f_night"]==night)]
            
                bright_entry = dates_obs[np.logical_and(dates_obs["star"]==sci, 
                                                       dates_obs["b_night"]==night)]
                
                # If returning both a faint and bright entry, need to define 
                # which is which                                   
                if len(bright_entry) > 0 and len(faint_entry) > 0:
                    # Bright
                    
                    if seq_i == bright_entry["b_order"].values[0]:
                        seq_id += " (bright, %s)" % bright_entry["period"].values[0]
                    
                    elif seq_i == faint_entry["f_order"].values[0]:
                        seq_id += " (faint, %s)" % faint_entry["period"].values[0]
                                    
                elif len(bright_entry) > 0 and len(faint_entry) == 0:
                    seq_id += " (bright, %s)" % bright_entry["period"].values[0]
                
                elif len(bright_entry) == 0 and len(faint_entry) > 0:
                    seq_id += " (faint, %s)" % faint_entry["period"].values[0]
        
        # Extract data from oifits file and stack as appropriate
        #mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            #extract_vis2(oifits)

            if seq_id not in all_vis2.keys():
                all_mjds[seq_id] = mjds[seq_i]
                all_tel_pairs[seq_id] = pairs[seq_i]
                all_vis2[seq_id] = vis2[seq_i]
                all_e_vis2[seq_id] = e_vis2[seq_i]
                all_flags[seq_id] = flags[seq_i]
                all_baselines[seq_id] = baselines[seq_i]
                all_wavelengths[seq_id] = wavelengths
            
            else:
                all_mjds[seq_id] = np.hstack((all_mjds[seq_id], mjds[seq_i]))
                all_tel_pairs[seq_id] = np.hstack((all_tel_pairs[seq_id], pairs[seq_i]))
                all_vis2[seq_id] = np.vstack((all_vis2[seq_id], vis2[seq_i]))
                all_e_vis2[seq_id] = np.vstack((all_e_vis2[seq_id], e_vis2[seq_i]))
                all_flags[seq_id] = np.vstack((all_flags[seq_id], flags[seq_i]))
                all_baselines[seq_id] = np.hstack((all_baselines[seq_id], baselines[seq_i]))
                all_wavelengths[seq_id] = wavelengths # Fix if bootstrapping over this
                                                   
    return all_mjds, all_tel_pairs, all_vis2, all_e_vis2, all_flags, \
           all_baselines, all_wavelengths
    
    
def get_linear_limb_darkening_coeff(n_logg, n_teff, n_feh, filt="H", xi=2.0):
    """Function to interpolate the linear-limb darkening coefficients given 
    values of stellar logg, Teff, [Fe/H], microturbulent velocity, and a 
    photometric filter. The interpolated grid is from Claret and Bloemen 2011:
     - http://adsabs.harvard.edu/abs/2011A%26A...529A..75C
    
    Paremeters
    ----------
    logg: float or float array
        Stellar surface gravity
        
    teff: float or float array
        Stellar effective temperature in Kelvin
        
    feh: float or float array
        Stellar metallicity, [Fe/H] (relative to Solar)
        
    filt: string or string array
        The filter bandpass to use (H for PIONIER observations)
        
    xi: float or float array
        Microturbulent velocity (km/s)
        
    Returns
    -------
    u_lld: float or floar array
        The wavelength dependent linear limb-darkening coefficient
        
    u_ld_err: float or float array
        The error on u_lld
    """
    # Read the 
    filepath = "data/claret_bloemen_ld_grid.tsv" 
    ldd_grid = pd.read_csv(filepath, delim_whitespace=True, comment="#", 
                           header=0, dtype={"logg":np.float, "Teff":np.float, 
                                            "Z":np.float, "xi":np.float, 
                                            "u":np.float, "Filt":np.str, 
                                            "Met":np.str, "Mod":np.str})
    
    # Interpolate only over the portion of the grid with the relevant filter
    # and assumed microturbulent velocity
    subset = ldd_grid[(ldd_grid["Filt"]==filt) & (ldd_grid["xi"]==xi)]
    
    # Interpolate along logg and Teff for all entries for filter
    calc_u = LinearNDInterpolator(subset[["logg", "Teff", "Z"]], subset["u"])
    
    # Compute u_lld for every star
    ids = n_teff.columns.values
    
    n_params = np.zeros([len(n_teff), len(ids)])
    
    n_u_lld = pd.DataFrame(n_params, columns=ids)
    
    # Determine value for u given logg and teff
    for id in ids:
        n_u_lld[id] = calc_u(n_logg[id], n_teff[id], n_feh[id])
    
    # Return the results    
    return n_u_lld
    

def sample_n_pred_ldd(tgt_info, n_bootstraps, pred_ldd_col="LDD_pred", 
                      e_pred_ldd_col="e_LDD_pred",
                      do_gaussian_diam_sampling=True):
    """Prepares a pandas dataframe of predicted target diameters for 
    bootstrapping over. Each row will either be sampled from a Gaussian 
    distribution if doing calibrator bootstrapping, otherwise will simply be N
    repeats of the actual predicted diameters.
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
        
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    e_pred_ldd_col: string
        The column to use from tgt_info for the predicted diameter 
        uncertainties.
        
    do_gaussian_diam_sampling: bool
        Boolean indicating whether to sample the n_bootstraps LDD from a 
        Gaussian distribution constructed from pred_ldd_col and e_pred_ldd_col,
        or simply make n_bootstraps repeats of the predicted diameters without
        sampling.
        
    Returns
    -------
    n_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and each row being a set of
        LDD for a given bootstrapping iteration. If not doing calibrator 
        bootstrapping (do_gaussian_diam_sampling=False), each row will be the 
        same, but otherwise the calibrator angular diameters are drawn from a 
        Gaussian distribution as part of the bootstrapping.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to n_pred_ldd. Only one row.    
    """
    # Get the IDs
    ids = tgt_info.index.values
    
    e_pred_ldd = pd.DataFrame([tgt_info[e_pred_ldd_col].values], columns=ids)
    
    # If not running second stage of bootstrapping on calibrator predicted 
    # diameters create a dataframe with duplicate rows simply as the predicted
    # calibrator diameters, rather than drawing from a Gaussian distribution
    if not do_gaussian_diam_sampling:
        print("No calibrator bootstrapping --> using actual predicted LDD")
        n_pred_ldd = pd.DataFrame([tgt_info[pred_ldd_col].values], 
                                      columns=ids)
        n_pred_ldd = pd.concat([n_pred_ldd]*n_bootstraps, ignore_index=True)
        return n_pred_ldd, e_pred_ldd
        
    # Otherwise we are running cal bootstrapping, draw LDD from a Gaussian dist
    # Make a new pandas dataframe with columns representing an individual star,
    # and each row being the predicted LDD (pulled from a Gaussian 
    # distribution) for the ith bootstrapping iteration
    print("Calibrator bootstrapping --> drawing LDD from Gaussian dist")
    ldds = np.zeros([n_bootstraps, len(ids)])
    
    n_pred_ldd = pd.DataFrame(ldds, columns=ids)
    
    for id in ids:
        n_pred_ldd[id] = np.random.normal(tgt_info.loc[id, pred_ldd_col],
                                              tgt_info.loc[id, e_pred_ldd_col],
                                              n_bootstraps)                                           
    return n_pred_ldd, e_pred_ldd
    
    
def collate_bootstrapping(tgt_info, n_bootstraps, results_path, n_u_lld,
                          pred_ldd_col="LDD_pred", 
                          prune_errant_baselines=True, 
                          separate_sequences=True):
    """Collates all bootstrapped oifits files within results_path into
    sumarising pandas dataframes. 
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
    
    results_path: string
        Path to store the bootstrapped oifits files.
            
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    prune_errant_baselines: boolean
        Whether to replace non-matching vis2 and baseline data with NaNs to
        facillitate computation of vis2 errors. This does not affect the fitted
        LDD or its error computed from the distribution of fitted LDDs, and is
        purely to allow plotting of vis2 curves with errors.
            
    Returns
    -------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    """
    # Determine the stars that we have results on
    #oifits_files = glob.glob(results_path + "*SCI*.fits")
    #oifits_files.sort()
    mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            collate_vis2_from_file(results_path, 0, separate_sequences)
    
    #stars = set([file.split("SCI")[-1].split("oidata")[0].replace("_","")
                 #for file in oifits_files])
                 
    stars = mjds.keys()
    stars.sort()
                
    # Initialise a pandas dataframe for each star. At present it's hard to
    # entirely preallocate memory, but we'll try to at least preallocate the
    # rows
    cols1 = ["MJD", "TEL_PAIR", "VIS2", "FLAG", "BASELINE", 
            "WAVELENGTH", "LDD_FIT",  "LDD_PRED", "e_LDD_PRED", "u_LLD",
            "C_SCALE"]
            
    # Store the results for each star in a pandas dataframe, accessed by key 
    # from a dictionary
    bs_results = {}
        
    for star in stars:
        bs_results[star] = pd.DataFrame(index=np.arange(0, n_bootstraps), 
                                     columns=cols1)
        
        # TEL_PAIR --> array of tuples, MJD --> array of floats, VIS2 -->
        # array of 6 length arrays, BASELINE --> array of floats, WAVELENGTH 
        # --> array of 6 length arrays, FLAG --> array of 6 length arrays,
        # LDD --> array of floats
        bs_results[star]["MJD"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["TEL_PAIR"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["VIS2"] = np.zeros((n_bootstraps, 0)).tolist()
        #bs_results[star]["e_VIS2"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["FLAG"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["BASELINE"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["WAVELENGTH"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["LDD_FIT"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["C_SCALE"] = np.zeros((n_bootstraps, 0)).tolist()
    
    # Fit a LDD for every bootstrap iteration, and save the vis2, time, 
    # baseline, and wavelength information from each iteration
    for bs_i in np.arange(0, n_bootstraps):
        # Collate the information
        mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            collate_vis2_from_file(results_path, bs_i, separate_sequences)
          
        # Fit LDD, ldd_fits = [ldd_opt, e_ldd_opt, c_scale, e_c_scale]
        print("\nFitting diameters for bootstrap %i" % (bs_i+1))
        ldd_fits = fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, 
                               pred_ldd_col, n_u_lld.iloc[bs_i])  
                          
        # Populate
        for star in mjds.keys():
            bs_results[star]["MJD"][bs_i] = mjds[star]
            bs_results[star]["TEL_PAIR"][bs_i] = pairs[star]
            bs_results[star]["VIS2"][bs_i] = vis2[star]
            #bs_results[star]["e_VIS2"][bs_i] = e_vis2[star]
            bs_results[star]["FLAG"][bs_i] = flags[star]
            bs_results[star]["BASELINE"][bs_i] = baselines[star]
            bs_results[star]["WAVELENGTH"][bs_i] = wavelengths[star]
            bs_results[star]["LDD_FIT"][bs_i] = ldd_fits[star][0]
            bs_results[star]["C_SCALE"][bs_i] = ldd_fits[star][2]
            
            #bs_results[star]["LDD_PRED"][bs_i] = ldd_fits[star][2]
            #bs_results[star]["e_LDD_PRED"][bs_i] = ldd_fits[star][3]
            #bs_results[star]["u_LLD"][bs_i] = ldd_fits[star][4]
    
    # A minority of bootstraps result in a different number of observed 
    # baseline/vis2 measurements, which cannot be stacked to produce vis2 
    # errors. This step prunes them to enable plotting.
    if prune_errant_baselines:
        # Get the most common baseline count, and remove any not adhering
        shape_dict = {}
        for star in bs_results.keys():                   
            shape_dict[star] = []              
            for vis2 in bs_results[star]["TEL_PAIR"]:
                shape_dict[star].append(vis2.shape)                                                     
            shape_dict[star] = Counter(shape_dict[star])    
                
            # Get a list of the indices to drop
            num_most_common = shape_dict[star].most_common(1)[0][0][0]
            i_to_drop = [i_ob for i_ob, ob 
                         in enumerate(bs_results[star]["VIS2"].values)
                         if len(ob) != num_most_common]
                         
            # Now replace the errant vis2 and baseline data with nans
            for ob_i in i_to_drop:
                bs_results[star].iloc[ob_i]["VIS2"] = \
                    np.ones([num_most_common, 6])*np.nan
                bs_results[star].iloc[ob_i]["BASELINE"] = \
                    np.ones(num_most_common)*np.nan

    return bs_results


def summarise_bootstrapping(bs_results, tgt_info, pred_ldd_col="LDD_pred",
                           e_pred_ldd_col="e_LDD_pred"):
    """Summarise N boostrapping results by computing mean and standard 
    deviations for each distribution.
    
    Parameters
    ----------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    e_pred_ldd_col: string
        The column to use from tgt_info for the predicted diameter 
        uncertainties.
            
    Returns
    -------
    results: pandas dataframe
        Summarised results of the bootstrapping with mean and std values
        computed from respective parameter distributions.
    """    
    # Initialise
    cols = ["STAR", "HD", "PERIOD", "SEQUENCE", "VIS2", "e_VIS2", "BASELINE", 
            "WAVELENGTH", "LDD_FIT", "e_LDD_FIT", "LDD_PRED", "e_LDD_PRED", 
            "u_LLD", "C_SCALE"]
    results = pd.DataFrame(index=np.arange(0, len(bs_results.keys())), 
                           columns=cols)  
    
    stars = bs_results.keys()
    stars.sort()
    
    # All done collating, combine bootstrapped values into mean and std
    for star_i, star in enumerate(stars):
        # Set the common ID, and get the primary ID
        results.iloc[star_i]["STAR"] = star.split(" ")[0]
        
        pid = tgt_info[tgt_info["Primary"]==star.split(" ")[0]].index.values[0]
        
        if "(" in star:
            sequence = star.split(" ")[1][1:-1]
            period = int(star.split(" ")[-1][:-1])
        else:
            sequence = "combined"
            period = ""
        
        results.iloc[star_i]["HD"] = pid
        results.iloc[star_i]["PERIOD"] = period
        results.iloc[star_i]["SEQUENCE"] = sequence
        
        # Stack and compute mean and standard deviations 
        results.iloc[star_i]["LDD_FIT"] = \
            np.nanmean(np.hstack(bs_results[star]["LDD_FIT"]), axis=0)
            
        results.iloc[star_i]["e_LDD_FIT"] = \
            np.nanstd(np.hstack(bs_results[star]["LDD_FIT"]), axis=0)

        results.iloc[star_i]["VIS2"] = \
            np.nanmean(np.dstack(bs_results[star]["VIS2"]), axis=2)
            
        results.iloc[star_i]["e_VIS2"] = \
            np.nanstd(np.dstack(bs_results[star]["VIS2"]), axis=2)

        results.iloc[star_i]["BASELINE"] = \
            np.nanmean(np.vstack(bs_results[star]["BASELINE"]), axis=0)
    
        results.iloc[star_i]["WAVELENGTH"] = \
            np.nanmedian(np.vstack(bs_results[star]["WAVELENGTH"]), axis=0)
            
        results.iloc[star_i]["LDD_PRED"] = tgt_info.loc[pid, pred_ldd_col]    
        results.iloc[star_i]["e_LDD_PRED"] = tgt_info.loc[pid, e_pred_ldd_col]   
        results.iloc[star_i]["u_LLD"] = tgt_info.loc[pid, "u_lld"]    
        
        results.iloc[star_i]["C_SCALE"] = \
            np.nanmean(np.hstack(bs_results[star]["C_SCALE"]), axis=0)
        
        # Print some simple diagnostics                
        sci_percent_fit = (results.iloc[star_i]["e_LDD_FIT"]
                           / results.iloc[star_i]["LDD_FIT"]) * 100
           
        print("%-12s\tLDD = %f +/- %f (%0.2f%%), C=%0.2f" 
              % (star, results.iloc[star_i]["LDD_FIT"], 
                 results.iloc[star_i]["e_LDD_FIT"], sci_percent_fit, 
                 results.iloc[star_i]["C_SCALE"]))
    
    return results
    

def inspect_dr_photometry(tgt_info):
    """Diagnostic function to inspect for issues with reddening/diameters. WIP. 
    """
    print("%7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s" %
          ("ID", "B_a_mag", "V_a_mag", "J_a_mag", "H_a_mag", "K_a_mag", "Flag",
           "Dist", "LDD (V-K)", "LDD (V-W3)", "ID"))
    
    num_flagged = 0
    
    for star, row in tgt_info.iterrows():
        b_a_mag = row["Bmag"] - row["Bmag_dr"]
        v_a_mag = row["Vmag"] - row["Vmag_dr"]
        j_a_mag = row["Jmag"] - row["Jmag_dr"]
        h_a_mag = row["Hmag"] - row["Hmag_dr"]
        k_a_mag = row["Kmag"] - row["Kmag_dr"]
        primary = row["Primary"]
        vk_ldd = row["LDD_VK_dr"]
        vw3_ldd = row["LDD_VW3_dr"]
        flag = ""
        dist = row["Dist"]
        
        if np.max(np.abs([v_a_mag, j_a_mag, h_a_mag, k_a_mag])) > 0.1:
            flag = "***"
            num_flagged += 1
        
        print(("%8s \t %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f \t %7s \t"
               "%4.2f \t %5.3f \t %5.3f \t %s") 
                % (star, b_a_mag, v_a_mag, j_a_mag, h_a_mag, k_a_mag, flag, 
                   dist, vk_ldd, vw3_ldd, primary))
                       
    print("\nFlagged Stars: %i/%i" % (num_flagged, len(tgt_info)))
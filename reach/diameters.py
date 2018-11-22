"""Module for angular diameter prediction, calculation, and vis^2 fitting
"""
from __future__ import division, print_function
import os
import glob
import numpy as np
import pandas as pd
import reach.plotting as rplt
import matplotlib.pylab as plt
from astropy.io import fits
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator


class UnknownOIFitsFileFormat(Exception):
    pass


def predict_ldd_boyajian(F1_mag, F1_mag_err, F2_mag, F2_mag_err, 
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
    num_exponents = 4
    
    # Convert to numpy arrays
    F1_mag = np.array(F1_mag)
    F1_mag_err = np.array(F1_mag_err)
    F2_mag = np.array(F2_mag)
    F2_mag_err = np.array(F2_mag_err)
    
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
    colour = F1_mag - F2_mag
    
    # Repeat along new direction so operation is vectorised
    colour = np.repeat(np.array(colour)[:,None], num_exponents, 1)
    
    log_diam = np.sum(diam_rel_coeff[colour_rel][::2]*colour**exponents, 1)
    
    ldd = 10**(-0.2*F1_mag) * 10**log_diam
    
    # Calculate the error. Currently this is done solely using the percentage
    # error given in the paper, and does not treat errors in either magnitude
    e_ldd = ldd * diam_rel_err[colour_rel]/100
    
    # Return zeroes rather than nans
    # NOTE: This should be considered placeholder code only
    ldd[np.isnan(ldd)] = 1    
    e_ldd[np.isnan(e_ldd)] = 0.1 
    
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
     

def calculate_vis2(b_on_lambda, ldd, u_lld):
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
    return vis**2
          
          
def fit_for_ldd(vis2, e_vis2, baselines, wavelengths, u_lld, ldd_pred):
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
    ldd_opt: float
        Optimal value for the limb-darkened diameter (mas)
        
    e_ldd_opt: float
        Error (one standard deviation) of ldd_opt (mas)
    """
    # Baseline/lambda should have dimensions [B,W], where B is the number of 
    # baselines, and W is the number of wavelengths
    n_bl = len(baselines)
    n_wl = len(wavelengths)
    bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
    wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
    b_on_lambda = (bl_grid / wl_grid).flatten()
    
    # Don't consider bad data during fitting process
    valid_i = (vis2.flatten() >= 0) & (e_vis2.flatten() > 0)
    
    # Fit for LDD. The lambda function means that we can fix u_lld and not have
    # to optimise for it too. Loose, but physically realistic bounds on LDD for
    # science targets (LDD cannot be zero else the fitting/formula will fail) 
    ldd_opt, ldd_cov = curve_fit((lambda b_on_lambda, ldd_pred: 
                                 calculate_vis2(b_on_lambda, ldd_pred, u_lld)), 
                                 b_on_lambda[valid_i], vis2.flatten()[valid_i], 
                                 sigma=e_vis2.flatten()[valid_i], 
                                 bounds=(0.1, 10))
    
    # Compute standard deviation of ldd 
    e_ldd_opt = np.sqrt(np.diag(ldd_cov))
    
    # Diagnostic checks on fitting perfomance. 
    # TODO: move out of this function later
    #print("Predicted: %f, Actual: %f" % (ldd_pred, ldd_opt[0]))
    #rplt.plot_vis2_fit(b_on_lambda, vis2.flatten(), e_vis2.flatten(), 
    #                  ldd_opt[0], ldd_pred, u_lld)
     
    # Only estimating one parameter, so no need to send back N=1 array                       
    return ldd_opt[0], e_ldd_opt[0]


def fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, do_plot=False):
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
    """
    successful_fits = {}
    print("\n", "-"*79, "\n", "\tFitting for LDD\n", "-"*79)
    for sci in vis2.keys():
        
        sci_data = tgt_info[tgt_info["Primary"]==sci]
        
        if not sci_data["Science"].values:
            print("%s is not science target, aborting fit" % sci)
            continue
        else:
            print("Fitting linear LDD to %s" % sci, end="")
            
        #print(vis2[sci].shape, e_vis2[sci].shape, baselines[sci].shape, 
              #len(wavelengths), sci_data["u_lld"].values[0], 
              #sci_data["LDD_VW3_dr"].values[0])
        try:
            ldd_opt, e_ldd_opt = fit_for_ldd(vis2[sci], e_vis2[sci], 
                                             baselines[sci], wavelengths, 
                                             sci_data["u_lld"].values[0], 
                                             sci_data["LDD_VW3_dr"].values[0])
            print("...fit successful")
            successful_fits[sci] = [ldd_opt, e_ldd_opt, 
                                    sci_data["LDD_VW3_dr"].values[0],
                                    sci_data["e_LDD_VW3_dr"].values[0],
                                    sci_data["u_lld"].values[0]]
            
        except Exception, err:
            print("...exception, aborting fit - %s" % err)                              
                                         
    # All Done, create diagnostic plots
    if do_plot:
        rplt.plot_all_vis2_fits(successful_fits, baselines, wavelengths, vis2, 
                                e_vis2)
            
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
    vis2: float array
        Calibrated squared visibiity measurements
        
    e_vis2: float array
        Error on the calibrated squared visibility measurements
        
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
    oifits = fits.open(oi_fits_file)
    n_extra_seq = (len(oifits) - 6) // 3
    
    vis2 = []
    e_vis2 = []
    baselines = []
    
    # Retrieve visibility and baseline information for an arbitrary (>=1) 
    # number of sequences within a given night
    for seq_i in xrange(0, n_extra_seq+1):
        oidata = oifits[4 + n_extra_seq + seq_i].data
        
        if len(vis2)==0 and len(e_vis2)==0 and len(baselines)==0:
            vis2 = oidata["VIS2DATA"]
            e_vis2 = oidata["VIS2ERR"]
            baselines = np.sqrt(oidata["UCOORD"]**2 + oidata["VCOORD"]**2)
        else:
            vis2 = np.vstack((vis2, oidata["VIS2DATA"]))
            e_vis2 = np.vstack((e_vis2, oidata["VIS2ERR"]))
            baselines = np.hstack((baselines, np.sqrt(oidata["UCOORD"]**2 
                                                   + oidata["VCOORD"]**2)))
    
    # Assume that we'll always be using the same wavelength mode within a night      
    wavelengths = oifits[2].data["EFF_WAVE"]
    
    return vis2, e_vis2, baselines, wavelengths


def collate_vis2_from_file(results_path="/home/arains/code/reach/results/"):
    """Collates calibrated squared visibilities, errors, baselines, and 
    wavelengths for each science target in the specified results folder.
    
    Parameters
    ----------
    results_path: string
        Directory where the calibrated oifits results files are stored.
        
    Returns
    -------
    all_vis2: dict
        Dictionary mapping science target ID to all vis^2 values
    
    all_e_vis2: dict
        Dictionary mapping science target ID to all e_vis^2 values
        
    all_baselines: dict
        Dictionary mapping science target ID to all projected baselines (m)
    
    wavelengths: list
        List recording the wavelengths observed at (m)
    """
    # Initialise data structures to store calibrated results, where dict keys
    # are the science target IDs. Note that the wavelengths are common to all.
    all_vis2 = {}
    all_e_vis2 = {}
    all_baselines = {}
    wavelengths = []
    
    all_results = glob.glob(results_path + "*SCI*oidataCalibrated.fits")
    all_results.sort()
    
    print("\n", "-"*79, "\n", "\tCollating Calibrated vis2\n", "-"*79)
    
    for oifits in all_results:
        # Get the target name from the file name
        sci = oifits.split("/")[-1][15:-22].replace("_", "")
        
        print("Collating %s" % oifits, end="")
        
        # Open the file
        try:
            vis2, e_vis2, baselines, wavelengths = extract_vis2(oifits)
            print("...success")
        except:
            print("...failure, unknown oifits format")
            continue
        
        if sci not in all_vis2.keys():
            all_vis2[sci] = vis2
            all_e_vis2[sci] = e_vis2
            all_baselines[sci] = baselines
            
        else:
            all_vis2[sci] = np.vstack((all_vis2[sci], vis2))
            all_e_vis2[sci] = np.vstack((all_e_vis2[sci], e_vis2))
            all_baselines[sci] = np.hstack((all_baselines[sci], baselines))
                                                   
    return all_vis2, all_e_vis2, all_baselines, wavelengths
    
    
def get_linear_limb_darkening_coeff(logg, teff, feh, filt="H", xi=2.0):
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
    
    # Determine value for u given logg and teff
    u_lld = calc_u(logg, teff, feh)
    
    # Return the results    
    return u_lld
    

def sample_n_gaussian_ldd(tgt_info, n_bootstraps, pred_ldd_col, 
                          e_pred_ldd_col):
    """
    """
    # Get the IDs
    ids = tgt_info.index.values
    
    e_pred_ldd = pd.DataFrame([tgt_info[e_pred_ldd_col].values], columns=ids)
    
    # If n_bootstraps = 0, return the actual predicted LDD
    if n_bootstraps < 1:
        print("No bootstrapping, using actual predicted LDD")
        n_gaussian_ldd = pd.DataFrame([tgt_info[pred_ldd_col].values], 
                                      columns=ids)
        return n_gaussian_ldd, e_pred_ldd
        
    # n_bootstraps is >= 1, draw LDD from a Gaussian distribution.
    # Make a new pandas dataframe with columns representing an individual star,
    # and each row being the predicted LDD (pulled from a Gaussian 
    # distribution) for the ith bootstrapping iteration
    print("%s bootstrap iterations, drawing LDD from Gaussian distributions" 
          % n_bootstraps)
    ldds = np.zeros([n_bootstraps, len(ids)])
    
    n_guassian_ldd = pd.DataFrame(ldds, columns=ids)
    
    for id in ids:
        n_guassian_ldd[id] = np.random.normal(tgt_info.loc[id, pred_ldd_col],
                                              tgt_info.loc[id, e_pred_ldd_col],
                                              n_bootstraps)                                           
    return n_guassian_ldd, e_pred_ldd
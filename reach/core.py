"""
Main location for reach functions
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

class ColourOutOfBoundsException(Exception):
    pass

def predict_ldd_boyajian(F1_mag, F1_mag_err, F2_mag, F2_mag_err, 
                         colour_rel="V-W3"):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Boyajian et al. 2014:
        http://adsabs.harvard.edu/abs/2014AJ....147...47B
        
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
    boyajian_2014_rel_file = "boyajian_2014_colour_diameter_relations.csv"
    
    diam_rel = np.loadtxt(boyajian_2014_rel_file, delimiter=",", 
                          skiprows=1, dtype="str")
    # Create dictionary for coefficients
    diam_rel_coeff = {}
    
    for rel in diam_rel:
        diam_rel_coeff[rel[0]] = rel[3:11].astype(float)
    
    # Calculate diameters
    # Relationship is log_diam = Sigma(i=0) a_i * (F1_mag-F2_mag) ** i
    exponents = np.arange(0, num_exponents)
    colour = F1_mag - F2_mag
    
    # Repeat along new direction so operation is vectorised
    colour = np.repeat(np.array(colour)[:,None], num_exponents, 1)
    
    log_diam = np.sum(diam_rel_coeff[colour_rel][::2]*colour**exponents, 1)
    
    ldd = 10**(-0.2*F1_mag) * 10**log_diam
    
    # Calculate the error (TODO)
    ldd_err = 0
    
    return ldd, ldd_err
    
    
def predict_ldd_kervella(V_mag, V_mag_err, K_mag, K_mag_err):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Kervella et al. 2004:
        http://adsabs.harvard.edu/abs/2004A%26A...426..297K
    """
    # Calculate the LDD
    log_ldd = 0.0755 * (V_mag - K_mag) + 0.5170 - 0.2 * K_mag

    ldd = 10**log_ldd

    # Calculate the error on this value (assuming no covariance)
    log_ldd_err = np.sqrt((0.0755*V_mag_err)**2 + (0.2755*K_mag_err)**2)

    ldd_err = 10**(np.log(10)*log_ldd_err)
                                       
    return log_ldd, log_ldd_err, ldd, ldd_err   
     

def calculate_ldd():
    """Calculates limb-darkened disk model to fit to observations as outlined 
    in Hanbury Brown et al. 1974: 
        http://adsabs.harvard.edu/abs/1974MNRAS.167..475H
    """
    pass
    
    
def convert_vt_to_v(B_T, V_T):
    """Convert Tycho V_T band magnitudes to Cousins-Johnson V band using 
    relations from Bessell 2000:
        http://adsabs.harvard.edu/abs/2000PASP..112..961B)
    
    Import the relation points from file, fit a cubic spline to them, and use
    that to predict the Cousins-Johnson V band magnitude.
        
    Parameters
    ----------
    B_T: float or float array
        Tycho B_T band magnitude.
    V_T: float or float array
        Tycho V_T band magnitude.
    
    Returns
    -------
    V_predicted: float or float array
        The predicted V band magnitude in the Cousins-Johnson system.
    """
    # Load the stored colour relations table in from Bessell 2000 (Table 2)
    # Columns are: [B_T-V_T    V-V_T    (B-V)-(B_T-V_T)    V-H_P]
    bessell_2000_rel_file = "bessell_2000_bt_colour_relations.csv"
    colour_rel = np.loadtxt(bessell_2000_rel_file, delimiter=",", skiprows=1)
    
    # Create a cubic spline using the first two columns of the imported table
    colour_func = interp1d(colour_rel[:,0], colour_rel[:,1], kind="cubic") 
    
    # Calculate (B_T-V_T)
    B_T_minus_V_T = B_T - V_T
    
    # Interpolation only works for colour range in Bessell 2000 - reject values
    # that fall outside of this
    if not np.min(B_T_minus_V_T) > np.min(colour_rel[:,0]):
       raise ColourOutOfBoundsException("Minimum (B_T-V_T) colour must be >"
                                        " %f20" % np.min(colour_rel[:,0]))
    elif not np.max(B_T_minus_V_T) < np.max(colour_rel[:,0]):
       raise ColourOutOfBoundsException("Maximum (B_T-V_T) colour must be <"
                                        " %f0" % np.max(colour_rel[:,0]))
    
    # Predict (V-V_T) from (B_T-V_T)
    V_minus_V_T_predicted = colour_func(B_T_minus_V_T)
    
    # Determine V from the (V-V_T) prediction
    V_predicted = V_minus_V_T_predicted + V_T
    
    return V_predicted
    
    
def load_target_information(filepath="target_info.tsv"):
    """Loads in the target information tsv (tab separated) as a pandas 
    dataframne with appropriate column labels and rows labelled as each star.
    
    https://pandas.pydata.org/pandas-docs/version/0.21/generated/
    pandas.read_csv.html#pandas.read_csv
    """
    # Import (TODO: specify dtypes)
    target_info = pd.read_csv(filepath, sep="\t", header=1, index_col=5, 
                              skiprows=0)
    
    # Organise dataframe by removing duplicates
    # Note that the tilde is a bitwise not operation on the mask
    target_info = target_info[~target_info.index.duplicated(keep="first")]
    
    # Return result
    return target_info
    
    
    
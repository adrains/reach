"""
Main location for reach functions
"""
from __future__ import division, print_function
import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from astropy.io import fits
from collections import OrderedDict
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Science Functions
# -----------------------------------------------------------------------------
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
    dd = os.path.dirname(__file__)[:-5]
    boyajian_2014_rel_file = os.path.join(dd, 
                            "boyajian_2014_colour_diameter_relations.csv")
    
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
    
    # Calculate the error. Currently this is done solely using the percentage
    # error given in the paper, and does not treat errors in either magnitude
    e_ldd = ldd * diam_rel_coeff[colour_rel][-1]/100
    
    return ldd, e_ldd
    
    
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


def deredden():
    """
    """
    pass
    
# -----------------------------------------------------------------------------
# Utilities Functions
# -----------------------------------------------------------------------------
def summarise_sequences():
    """
    """
    # Read in each sequence
    bright_list_files = ["p99_bright.txt", "p101_bright.txt"]
    faint_list_files = ["p99_faint.txt", "p101_faint.txt"]
    period = [99, 101]

    target_list = []

    for p_i, bright_list_file in enumerate(bright_list_files):
        with open(bright_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "bright"))

    for p_i, faint_list_file in enumerate(faint_list_files):
        with open(faint_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "faint"))
        
    # Order each sequence
    sequences = OrderedDict()
    
    for tgt_i in xrange(0, len(target_list), 4):
        # All targets must share a sequence and period
        assert (target_list[tgt_i][::2] == target_list[tgt_i+1][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+2][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+3][::2])
        
        sequences[target_list[tgt_i]] = [target_list[tgt_i+1][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+2][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+3][1]]
    
    pkl_sequences = open("sequences.pkl", "wb")
    pickle.dump(sequences, pkl_sequences)
    pkl_sequences.close()
    
    return sequences
    
def load_target_information(filepath="target_info.tsv"):
    """Loads in the target information tsv (tab separated) as a pandas 
    dataframne with appropriate column labels and rows labelled as each star.
    
    https://pandas.pydata.org/pandas-docs/version/0.21/generated/
    pandas.read_csv.html#pandas.read_csv
    """
    # Import (TODO: specify dtypes)
    tgt_info = pd.read_csv(filepath, sep="\t", header=1, index_col=8, 
                              skiprows=0)
    
    # Organise dataframe by removing duplicates
    # Note that the tilde is a bitwise not operation on the mask
    tgt_info = tgt_info[~tgt_info.index.duplicated(keep="first")]
    
    # Force primary IDs to standard no-space format
    tgt_info.Primary = [pid.replace(" ", "").replace(".", "").replace("_","")
                           for pid in tgt_info.Primary]
    
    # Return result
    return tgt_info
    
    
# -----------------------------------------------------------------------------
# pndrs Affiliated Functions
# -----------------------------------------------------------------------------
def summarise_observations():
    """
    """
    pass
    
def save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                     base_path="/priv/mulga1/arains/pionier/complete_sequences/",
                     dir_suffix="_v3.73_abcd"):
    """This is a function to create and save the oiDiam.fits files referenced
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_oiDiam.fits containing an
    empty primary HDU, and a fits table with LDD and e_LDD for each star listed
    alphabetically.
    """
    
    nights = OrderedDict()
    
    # Get nightly sets of what targets have been observed
    for seq in complete_sequences:
        night = complete_sequences[seq][0]
        
        sequence = [star.replace("_", "").replace(".","").replace(" ", "") 
                    for star in sequences[seq]]
        
        if night not in nights:
            nights[night] = set(sequence)
        else:
            nights[night].update(sequence)
    
    print("Writing oiDiam.fits for %i nights" % len(nights))
    
    # For every night, construct a record array/fits file of target diameters
    # This record array takes the form:
    # TARGET_ID, DIAM, DIAMERR, HMAG, KMAG, VMAG, ISCAL, TARGET, INFO
    #   >i2      >f8     >f8    >f8   >f8   >f8    >i4    s8     s18
    # Where TARGET_ID is simply an integer index (one indexed), ISCAL is a 
    # boolean value of either 0 or 1, and TARGET is the name of the target. 
    # The targets are ordered by name, but sorted in ascii order (i.e. all 
    # numbers, then all capital letters, then all lower case letters). Unclear 
    # how significant this is. Only Hmags have been populated for some stars, 
    # though it is unclear what impact this has on the calibration.
    for night in nights:
        
        failed = False
        
        ids = []
        # Grab the primary IDs
        for star in nights[night]:
            prim_id = tgt_info[tgt_info.Primary==star].index
            try:
                assert len(prim_id) > 0
            except:
                print("...failed on %s, %s" % (night, star))
                failed = True
                break
            ids.append(prim_id[0])    
            
        if failed:
            continue
            
        # Sort the IDs
        ids.sort()   
        
        # Construct the record
        rec = tgt_info.loc[ids][["LDD_V_W3", "e_LDD_V_W3", "Hmag", "Kmag", 
                                 "Vmag", "Science", "Ref_ID_1"]]
        
        # Invert, as column is for calibrator status
        rec.Science =  np.abs(rec.Science - 1)
        rec["INFO"] = np.repeat("(V-W3) diameter from Boyajian et al. 2014",
                                len(rec))
        
        rec.insert(0,"TARGET_ID", np.arange(1,len(nights[night])+1))
        
        max_id = np.max([len(id) for id in rec["Ref_ID_1"]])
        max_info = np.max([len(info) for info in rec["INFO"]])
        
        formats = "int16,float64,float64,float64,float64,float64,int32,a%s,a%s"
        formats = formats % (max_id, max_info)
        
        names = "TARGET_ID,DIAM,DIAMERR,HMAG,KMAG,VMAG,ISCAL,TARGET,INFO"
        rec = np.rec.array(rec.values.tolist(), names=names, formats=formats)
        
        # Construct a fits/astopy table in this form
        hdu = fits.BinTableHDU.from_columns(rec)
        
        hdu.header["EXTNAME"] = ("OIU_DIAM", 
                                 "name of this binary table extension")
    
        # Save the fits file to the night directory
        dir = base_path + night + dir_suffix 
        
        if os.path.exists(dir):
            #fname = base_path + night + "_oiDiam.fits"
            fname = dir + "/" + night + "_oiDiam.fits" 
            hdu.writeto(fname, output_verify="warn", overwrite=True)
            
            # Done, move to the next night
            print("...wrote %s, %s" % (night, nights[night]))
        else:
            # The directory does not exist, flag
            print("...directory '%s' does not exist" % dir)
        
    return nights


def save_nightly_pndrs_script():
    """This is a function to create and save the pndrs script files referenced
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_pndrsScript.i containing a list
    of pndrs commands to run in order to customise the calibration procedure.
    
    Important here are the following commands:
        - Ignore some observations: oiFitsFlagOiData
        - Splot the night: oiFitsSplitNight
    """
    for night in all_nights:
        # Initialise empty string for YYYY-MM-DD_pndrsScript.i
        
        # Split the night if more than one sequence has been observed
        
        # Disqualify any bad calibrators
        pass
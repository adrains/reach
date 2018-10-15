"""
Main location for reach functions
"""
from __future__ import division, print_function
import os
import csv
import pickle
import extinction
import numpy as np
import pandas as pd
import reach.plotting as rplt
import matplotlib.pylab as plt
from astropy.io import fits
from collections import OrderedDict
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator

# -----------------------------------------------------------------------------
# Angular Diameters
# -----------------------------------------------------------------------------
class ColourOutOfBoundsException(Exception):
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
    
    # Fit for LDD. The lambda function means that we can fix u_lld and not have
    # to optimise for it too. Loose, but physically realistic bounds on LDD for
    # science targets (LDD cannot be zero else the fitting/formula will fail) 
    ldd_opt, ldd_cov = curve_fit((lambda b_on_lambda, ldd_pred: 
                                 calculate_vis2(b_on_lambda, ldd_pred, u_lld)), 
                                 b_on_lambda, vis2.flatten(), 
                                 sigma=e_vis2.flatten(), bounds=(0.1, 10))
    
    # Compute standard deviation of ldd 
    e_ldd_opt = np.sqrt(np.diag(ldd_cov))
    
    # Diagnostic checks on fitting perfomance. 
    # TODO: move out of this function later
    print("Predicted: %f, Actual: %f" % (ldd_pred, ldd_opt[0]))
    rplt.plot_vis2_fit(b_on_lambda, vis2.flatten(), e_vis2.flatten(), 
                      ldd_opt[0], ldd_pred, u_lld)
     
    # Only estimating one parameter, so no need to send back N=1 array                       
    return ldd_opt[0], e_ldd_opt[0]



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
    oidata = fits.open(oi_fits_file)[4].data
    
    vis2 = oidata["VIS2DATA"]
    e_vis2 = oidata["VIS2ERR"]
    baselines = np.sqrt(oidata["UCOORD"]**2 + oidata["VCOORD"]**2)
    wavelengths = fits.open(oi_fits_file)[2].data["EFF_WAVE"]
    
    return vis2, e_vis2, baselines, wavelengths


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
    
    
# -----------------------------------------------------------------------------
# Photometry/Reddening
# -----------------------------------------------------------------------------    
def convert_vtbt_to_vb(BTmag, VTmag):
    """Convert Tycho BT and VT band magnitudes to Cousins-Johnson B and V band  
    using relations from Bessell 2000:
     - http://adsabs.harvard.edu/abs/2000PASP..112..961B)
    
    Import the relation points from file, fit a cubic spline to them, and use
    that to predict the Cousins-Johnson B and V band magnitudes.
        
    Parameters
    ----------
    BTmag: float or float array
        Tycho B_T band magnitude.
    VTmag: float or float array
        Tycho V_T band magnitude.
    
    Returns
    -------
    Bmag: float or float array
        The predicted B band magnitude in the Cousins-Johnson system.
    Vmag: float or float array
        The predicted V band magnitude in the Cousins-Johnson system.
    """
    # Load the stored colour relations table in from Bessell 2000 (Table 2)
    # Columns are: [BT-VT    V-VT    (B-V)-(BT-VT)    V-H_P]
    bessell_2000_rel_file = "data/bessell_2000_bt_colour_relations.csv"
    colour_rel = np.loadtxt(bessell_2000_rel_file, delimiter=",", skiprows=1)
    
    # Create cubic splines interpolators to predict (V-VT) and delta(V-T)
    # from (BT-VT) - second, third, and first column of the csv respectively
    predict_V_minus_VT = interp1d(colour_rel[:,0], colour_rel[:,1], 
                                  kind="cubic") 
    predict_delta_B_minus_V = interp1d(colour_rel[:,0], colour_rel[:,2], 
                                       kind="cubic") 
    
    # Calculate (BT-VT)
    BT_minus_VT = BTmag - VTmag
    
    # Interpolation only works for colour range in Bessell 2000 - reject values
    # that fall outside of this
    if not np.min(BT_minus_VT) > np.min(colour_rel[:,0]):
       raise ColourOutOfBoundsException("Minimum (B_T-V_T) colour must be >"
                                        " %f20" % np.min(colour_rel[:,0]))
    elif not np.max(BT_minus_VT) < np.max(colour_rel[:,0]):
       raise ColourOutOfBoundsException("Maximum (B_T-V_T) colour must be <"
                                        " %f0" % np.max(colour_rel[:,0]))
    
    # Predict (V-VT) from (BT-VT)
    V_minus_VT= predict_V_minus_VT(BT_minus_VT)
    
    # Determine V from the (V-VT) prediction
    Vmag = V_minus_VT + VTmag
    
    # Predict delta(B-V)
    delta_B_minus_V = predict_delta_B_minus_V(BT_minus_VT)
    
    # Determine B from delta(B-V), BT, VT, and V
    Bmag = delta_B_minus_V + BT_minus_VT + Vmag
    
    return Bmag, Vmag


def create_spt_uv_grid(do_interpolate=True):
    """Create a grid of stellar instrinic (B-V) colours across spectral types.
    This is currently done for dwarfs using the following table, originally 
    from Pecaut & Mamajek 2013: 
     - http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt 
    
    Colours for stars not on the main sequence come from Schmidt-Kaler 1982:
     - http://adsabs.harvard.edu/abs/1982lbg6.conf.....A
    Which does not list Teff (currently we assume dwarf SpT for non-MS stars,
    which is not physically realistic), nor the subgiant branch (which is 
    interpolated as simply being halway between dwarfs and giants). As such, 
    this function is still a work in progress.
    
    Parameters
    ----------
    do_interpolate: boolean
        Escape parameter to construct the grid with solely the information
        provided in the input tables without interpolation.
        
    Returns
    -------
    grid: pandas dataframe
        Pandas dataframe of instrinsic (B-V) colours of form:
        [SpT, Teff, V, IV, III, Ib, Iab, Ia]
    """
    # Import the relations to be used
    m_colour_relations = "data/EEM_dwarf_UBVIJHK_colors_Teff.txt"
    sk_colour_relations = "data/schmidt-kaler_bv_colours.csv"
    
    mcr = pd.read_csv(m_colour_relations, comment="#", nrows=123, 
                      delim_whitespace=True, engine="python", index_col=0, 
                      na_values="...")
    
    skcr = pd.read_csv(sk_colour_relations, sep=",", index_col=0)
    
    # Initialise new pandas dataframe to store the entire grid. This should
    # be of the form:
    # | SpT | Teff |              (B-V)_0              |
    # |     |      | V | IV | III | II | Ib | Iab | Ia |
    # The values for dwarfs should come from the Mamajek table, as should the
    # labels and temperatures for the spectral types themselves. The (numeric)
    # temperatures will then be interpolated over to cover the Mamajek rang of
    # spectral types for the older (and less complete) Schmidt-Kaler dataset.
    
    # Remove the V from the Mamajek spectral types
    mcr.index = [spt[:-1] for spt in mcr.index]
    
    # Initialise the grid, with nans for empty spaces (take care to consider 
    # the difference between pandas views vs copy
    grid = mcr[["Teff", "B-V"]].copy()
    grid.rename(index=str, columns={"B-V":"V"}, inplace=True)
    grid["IV"] = np.nan
    grid["III"] = np.nan
    grid["II"] = np.nan
    grid["Ib"] = np.nan
    grid["Iab"] = np.nan
    grid["Ia"] = np.nan
    
    # Step through the Schmidt-Kaler relations and fill in the appropriate SpT
    for row_i, row in skcr.iterrows():
        if row.name in grid.index:
            # Add values for each spectral type
            grid.loc[row.name, "skV"] = row["V"]
            grid.loc[row.name, "III"] = row["III"]
            grid.loc[row.name, "II"] = row["II"]
            grid.loc[row.name, "Ib"] = row["Ia"]
            grid.loc[row.name, "Iab"] = row["Iab"]
            grid.loc[row.name, "Ia"] = row["Ia"]
    
    # Option to abort in case we only want the raw data sans interpolation
    if not do_interpolate:
        return grid
            
    # Only interpolate for spectral types without values *within* the 
    # interpolation range
    for col in ["III", "II", "Ib", "Iab", "Ia"]:
        # Using the temperatures, interpolate each along each spectral type and 
        # fill in the missing values
        teff = grid["Teff"][~np.isnan(grid[col])]
        b_minus_v = grid[col][~np.isnan(grid[col])]
        calc_b_minus_v = interp1d(teff, b_minus_v, kind="linear") 
        
        unknown_i = (np.isnan(grid[col]) & (grid["Teff"] > np.min(teff)) 
                          & (grid["Teff"] < np.max(teff)))

        grid.loc[unknown_i, col] = calc_b_minus_v(grid["Teff"][unknown_i])
    
    # Interpolate across the spectral types to fill in the values for subgiants
    # TODO - this is *bad*
    grid["IV"] = (grid["V"] + grid["III"]) / 2
    
    # Save and return the grid
    return grid
    
    
    
def calculate_selective_extinction(B_mag, V_mag, sptypes, grid):
    """Calculate the selective extinction (i.e. colour excess). This takes the
    form:
    
    E(B-V) = A(B) - A(V)
           = (B-V) - (B-V)_0
           
    Where E(B-V) is the selective extinction (i.e. the additional colour excess
    caused by extinction), A(B) and A(V) are the extinctions in the B and V 
    bands respectively, (B-V) is the observed B minus V colour, and (B-V)_0 is
    the unextincted B minus V colour.
    
    See http://w.astro.berkeley.edu/~ay216/08/NOTES/Lecture05-08.pdf
    
    Parameters
    ----------
    B_mag: float array
        Apparent B band magnitude.
    
    V_mag: float array
        Apparent V band magnitude
        
    sptypes: string array
        Spectral type/s
    
    grid: pandas dataframe
        Pandas dataframe of instrinsic (B-V) colours of form:
        [SpT, Teff, V, IV, III, Ib, Iab, Ia]
        
    Returns
    -------
    e_bv: float array
        Selective extinction, E(B-V) = (B-V) - (B-V)_0
    """
    # In the order listed, look for the luminosity class of the star in its 
    # full SpT. When a match is found, break from the look to avoid partial 
    # matches (e.g. both IV and V are in G4IV). This is required as SpT is 
    # typically written as a single string, whereas it is split over two
    # dimensions in the (B-V) grid.
    classes = ["IV", "V", "III", "II", "Ib", "Iab", "Ia"]
    
    bv_0_all = np.zeros(len(B_mag))
    
    lum_class_matched = False
    
    for spt_i, spt_full in enumerate(sptypes):
        # Determine class
        for lum_class in classes:
            if lum_class in spt_full:
                lum_class_matched = True
                break
               
        assert lum_class_matched
        
        spt = spt_full.replace(lum_class, "")
        
        # SpT has been identified and split, get (B-V) colour
        bv_0_all[spt_i] = grid.loc[spt, lum_class]
        
    ebv = (B_mag - V_mag) - bv_0_all
    
    return ebv
        
    
def calculate_v_band_extinction(e_bv, r_v=3.1):
    """Calculate A(V) from: 
    
    R_V = A(V) / [A(B) - A(V)]
        = A(V) / E(B-V)
        
    Where 1/R_V is the normalised extinction, and measures the steepness of the 
    extinction curve. R_V = 3.1 +/- 0.2 is for the diffuse ISM, R ~= 5 is for 
    dark interstellar clouds.
    
    A(V) is the V band extinction and serves as a scaling parameter. Thus:
    
    A(V) = R_V * E(B-V) 
    
    Parameters
    ----------
    e_bv: float
        Selective extinction, E(B-V).
    r_v: float
        Ratio of total to selective extinction, A_V / E(B-V).
    
    Returns
    -------
    a_v: float
        V band extinction.
    """
    a_v = r_v * e_bv
    
    return a_v
    
    
def calculate_effective_wavelength(spt, filter):
    """Given a stellar spectral type, and a particular photometric filter, 
    return the effective wavelength.
    
    This is per discussion in Bessell et al. 1998. As an example: 
     - "In broad-band photometry the nominal wavelength associated with a 
        passband (the effective wavelength) shifts with the color of the star.
        For Vega the effective wavelength of the V band is 5448 A and for the 
        sun it is 5502 A"
        
    Ideally this function should just reference a grid of effective wavelengths
    from a table comparing filters and spectral types.
    
    Parameters
    ----------
    spt: string
        Spectral type of the star/s.
        
    filter: string
        Name of the photometric band
    
    Returns
    -------
    filter_eff_lambda: float
    
    """
    pass
    
    
def deredden_photometry(ext_mag, ext_mag_err, filter_eff_lambda, a_v, r_v=3.1):
    """Use an extinction law to deredden photometry from a given band.
    
    Relies on:
        https://github.com/kbarbary/extinction
    With documentation at:
        https://extinction.readthedocs.io/en/latest/
    
    Parameters
    ----------
    ext_mag: np.array of type float
        The extincted magnitude.
    
    ext_mag_err: np.array of type float
        Error in the extincted magnitude
    
    filter_eff_lamda: np.array of type float
        Effective wavelength of the broad-band photometric filter specific to
        stellar spectral type.
    
    a_v : np.array of type float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : np.array of type float
        Ratio of total to selective extinction, A_V / E(B-V).
        
    Returns
    -------
    a_mags: float array
        Array of photometric extinction of form [W, S], where W is the number
        of wavelengths, and S is the number of stars.
    
    de_ext_mag_err: np.array of type float
        Error in the de-extincted magnitude.
    """
    # Create grid of extinction
    a_mags = np.zeros(ext_mag.shape)
    
    # Use the Cardelli, Clayton, & Mathis 1989 extinction model. The extinction
    # module is not vectorised, so we have to work with one star at a time
    for star_i, star in enumerate(ext_mag.itertuples(index=False)):
        a_mags[star_i,:] = extinction.ccm89(filter_eff_lambda, a_v[star_i], 
                                            r_v)
    
    return a_mags
    
    
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
    
    pkl_sequences = open("data/sequences.pkl", "wb")
    pickle.dump(sequences, pkl_sequences)
    pkl_sequences.close()
    
    return sequences
    
def load_target_information(filepath="data/target_info.tsv"):
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
                dir_suffix="_v3.73_abcd", run_local=False, 
                ldd_col="LDD_VW3_dr", e_ldd_col="e_LDD_VW3_dr"):
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
        rec = tgt_info.loc[ids][[ldd_col, e_ldd_col, "Hmag", "Kmag", 
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
        if not run_local:
            dir = base_path + night + dir_suffix
        else:
            dir = "test/"
        
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
        - Split the night: oiFitsSplitNight
    """
    for night in all_nights:
        # Initialise empty string for YYYY-MM-DD_pndrsScript.i
        
        # Split the night if more than one sequence has been observed
        
        # Disqualify any bad calibrators
        pass
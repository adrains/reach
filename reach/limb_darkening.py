"""
Functions for the calculation or sampling of limb darkening coefficients. Can
calculate u_lambda two different ways:
 1) Using the 1D atmosphere grid from Claret and Bloemen 2011 
 2) Equivalent linear coefficients using the STAGGER 3D atmosphere grid (code
    written by Tim White)
"""
from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy import stats
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------   
def sample_lld_coeff(n_logg, n_teff, n_feh, force_claret_params=False):
    """
    Get limb darkening coefficents. By default opt for Stagger grid if stars
    fall within it, opting for Claret 2000 grid.

    For every science target, sample linear u_lambda N times, where u_lambda is
    a vector of length 6, corresponding to each of the wavelength channels of
    PIONIER. u_lambda in this case is the equivalent linear term for the 4
    parameter law giving the same side-lobe height, and comes with a scaling
    parameter to scale the resulting LDD.
    
    Uses multi-dimensional pandas dataframe per:
        https://stackoverflow.com/questions/46884648/storing-3-dimensional-
        data-in-pandas-dataframe
        
    This 3D pandas dataframe will have the structure:
        [star, [bs_i, teff, logg, feh, u_lld_1, u_lld_2, u_lld_3, u_lld_4, 
                u_lld_5, u_lld_6, u_scale]]

    Parameters
    ----------
    n_logg: float array
        List of surface gravities (cgs units).

    n_teff: float array
        List of stellar effective temperatures (K).

    n_feh: float array
        List of stellar metallicities (relative to solar).

    force_claret_params: boolean
        Whether to only use the Claret grid. Defaults to true.
    
    Returns
    -------
    n_u_lld: float array
        Array of limb darkening coefficients and scaling parameters for each 
        star, of shape [n_star, 12]. All coefficients and scaling parameters
        will be the same if using Claret params.
    """
    # Check if any set of logg/teff points lie outside the stagger grid. If any
    # do, we can't use the stagger grid as we won't be able to properly sample
    # the uncertainties.

    # First try to get the Stagger coefficients, but if this doesn't work, just
    # get the Claret params
    if not force_claret_params:
        try:
            # Sample the grid
            elcs, scls, ftcs = elc_stagger(n_logg, n_teff, n_feh)
            
            # Combine
            n_u_lld = np.concatenate((elcs, scls)).T
            
            # Succeeded
            print("using Stagger grid.")
            out_of_stagger_bounds = False
            
        except:
            print("using Claret grid.")
            out_of_stagger_bounds = True

    else:
        print("using Claret grid.")

    # If we were out of grid bounds entirely, or some of the time, use Claret
    if (force_claret_params 
        or (out_of_stagger_bounds or elcs.shape[1] != len(n_logg))):
        # Get the H band linear coefficient
        n_u_lld = get_linear_limb_darkening_coeff(n_logg, n_teff, n_feh)
        
        # Stack x6 for each wavelength dimension
        n_u_lld = np.tile(n_u_lld, 6).reshape(6, len(n_logg))
        
        # Add a set of scaling coefficients (just ones)
        n_u_lld = np.concatenate((n_u_lld, np.ones((6, len(n_logg))))).T
    
    return n_u_lld

# -----------------------------------------------------------------------------
# Claret grid related functions
# -----------------------------------------------------------------------------    
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
    
    # Calculate and return
    n_u_lld = calc_u(n_logg, n_teff, n_feh)
    
    return n_u_lld

# -----------------------------------------------------------------------------
# Stagger grid related functions (Written by Tim White)
# -----------------------------------------------------------------------------   
def read_magic(file):
    """ Read Magic et al. (2015) file containing 4-term limb-darkening 
    coefficients for each Pionier wavelength channel    
    """
    
    data = []

    with open(file,'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            columns = line.split()
            model = {}
            model['00'] = {}
            model['01'] = {}
            model['02'] = {}
            model['03'] = {}
            model['04'] = {}
            model['05'] = {}
            model['teff'] = float(columns[0])
            model['sigteff'] = float(columns[1])
            model['logg'] = float(columns[2])
            model['feh'] = float(columns[3])
            model['00']['a1'] = float(columns[4])
            model['00']['a2'] = float(columns[5])
            model['00']['a3'] = float(columns[6])
            model['00']['a4'] = float(columns[7])
            model['01']['a1'] = float(columns[8])
            model['01']['a2'] = float(columns[9])
            model['01']['a3'] = float(columns[10])
            model['01']['a4'] = float(columns[11])
            model['02']['a1'] = float(columns[12])
            model['02']['a2'] = float(columns[13])
            model['02']['a3'] = float(columns[14])
            model['02']['a4'] = float(columns[15])
            model['03']['a1'] = float(columns[16])
            model['03']['a2'] = float(columns[17])
            model['03']['a3'] = float(columns[18])
            model['03']['a4'] = float(columns[19])
            model['04']['a1'] = float(columns[20])
            model['04']['a2'] = float(columns[21])
            model['04']['a3'] = float(columns[22])
            model['04']['a4'] = float(columns[23])
            model['05']['a1'] = float(columns[24])
            model['05']['a2'] = float(columns[25])
            model['05']['a3'] = float(columns[26])
            model['05']['a4'] = float(columns[27])
            model['05']['wl'] = 1.773
            model['04']['wl'] = 1.725
            model['03']['wl'] = 1.677
            model['02']['wl'] = 1.629
            model['01']['wl'] = 1.581
            model['00']['wl'] = 1.533
            data.append(model)
            
    return data

def elc_stagger(loggs, teffs, fehs):
    """ Find the equivalent limb-darkening coefficients for a given teff and 
    logg and their uncertainties. Code by Tim White.
    
    Parameters
    ----------
    loggs: float
        Surface gravities, log(g), in cgs units.
    teff: float
        Effective temperatures in K.    
    fehs: float
        Metallicities [Fe/H] relative to Solar.
    
    Returns
    ----------
    elcs: float array
        Equivalent linear coefficients, one per wavelength.
        
    scls: float array
        LDD scaling parameters to be used with the equivalent linear
        coefficents, one per wavelength channel.
        
    ftcs: float array
        Four term limb darkening coefficients, of shape 4 x N wavelengths.
    """
    grid1 = read_magic('data/stagger_pionier_m00.tab')
    grid2 = read_magic('data/stagger_pionier_m10.tab')
    ks = [0.5,1.0,1.5,2.0]
    
    # Triangulate the grids
    loggs1 = []
    teffs1 = []
    for d in grid1:
        logg = d['logg']
        teff = d['teff']
        loggs1.append(logg)
        teffs1.append(teff)
        
    loggs2 = []
    teffs2 = []
    for d in grid2:
        logg = d['logg']
        teff = d['teff']#
        loggs2.append(logg)
        teffs2.append(teff)

    points2D = np.vstack([np.log10(teffs1),np.log10(loggs1)]).T
    tri1 = Delaunay(points2D)
    
    points2D = np.vstack([np.log10(teffs2),np.log10(loggs2)]).T
    tri2 = Delaunay(points2D)
    
    #Values to be interpolated to
    t1 = np.log10(teffs)
    l1 = np.log10(loggs)
    f1 = fehs
    
    # Define wavelength channels
    channels = list(d)#.keys()
    channels.remove('teff')
    channels.remove('sigteff')
    channels.remove('logg')
    channels.remove('feh')
    channels.sort()

    wls = []
    elcs = []
    scls = []
    ftcs = []

    # Loop over wavelength channels and perform interpolation
    for c in channels:
        a1s1 = []
        a2s1 = []
        a3s1 = []
        a4s1 = []
        for d in grid1:
            ch = d[c]
            a1s1.append(ch['a1'])
            a2s1.append(ch['a2'])
            a3s1.append(ch['a3'])
            a4s1.append(ch['a4'])

        resCTa11 = CloughTocher2DInterpolator(tri1,a1s1)
        resCTa21 = CloughTocher2DInterpolator(tri1,a2s1)
        resCTa31 = CloughTocher2DInterpolator(tri1,a3s1)
        resCTa41 = CloughTocher2DInterpolator(tri1,a4s1)

        a1s2 = []
        a2s2 = []
        a3s2 = []
        a4s2 = []
        for d in grid2:
            ch = d[c]
            a1s2.append(ch['a1'])
            a2s2.append(ch['a2'])
            a3s2.append(ch['a3'])
            a4s2.append(ch['a4'])

        resCTa12 = CloughTocher2DInterpolator(tri2,a1s2)
        resCTa22 = CloughTocher2DInterpolator(tri2,a2s2)
        resCTa32 = CloughTocher2DInterpolator(tri2,a3s2)
        resCTa42 = CloughTocher2DInterpolator(tri2,a4s2)
        
        csCT1 = [resCTa11(t1,l1),resCTa21(t1,l1),resCTa31(t1,l1),resCTa41(t1,l1)]
        csCT2 = [resCTa12(t1,l1),resCTa22(t1,l1),resCTa32(t1,l1),resCTa42(t1,l1)]
        csCT = np.asarray(csCT1) - f1 * (np.asarray(csCT2) - np.asarray(csCT1))
        
        clean = [x for x in np.asarray(csCT).T if (np.isnan(x[0]) == False)]
        wls.append(ch['wl'])
        csCT = np.asarray(clean).T
        elc, scl = get_elc(ks,csCT)

        ftcs.append(np.asarray(csCT))
        elcs.append(elc)
        scls.append(scl)
    
    #melcs = np.nanmean(elcs,axis=1)
    #sigelcs = np.nanstd(elcs,axis=1)
    #mscls = np.nanmean(scls,axis=1)
    #sigscls = np.nanstd(scls,axis=1)
    #mftcs = np.nanmean(ftcs,axis=2)
    #sigftcs = np.nanstd(ftcs,axis=2)
    #wl = np.asarray(wls)
    
    return np.asarray(elcs), np.array(scls), np.array(ftcs)
    #return wls,melcs,sigelcs,mscls,sigscls,mftcs,sigftcs



def get_elc(ks,cs):
    """Find the equivalent linear coefficient for a given set of coefficients
    
    Returns:
        elc - equivalent linear coefficient
        scl - scale factor for angular diameter
    
    """
    
    if len(ks) != len(cs):
        raise UserWarning("Need one coefficient per k")
    
    # Determine how the maximum visbility in the sidelobe varies with the 
    # linear coefficient. Currently by fitting a polynomial to numerically 
    # calculated max visibilities. This won't change, so could hardcode the 
    # polynomical coefficients instead of recalculating.
    xs = np.arange(5.1,5.8,0.0001)
    #us = np.arange(0,1.001,0.001)
    #
    #mv0s = []
    #
    #for u in zip(us):
    #    vis0 = V_from_claret(xs,[1.0],u)
    #    mv0s.append(vis0.min())
    #mv0s = np.array(mv0s)
    
    #z = np.polyfit(us,mv0s**2,11)
    z = [2.61861994e-06, 1.87429278e-05, -6.07351449e-05, 9.25521846e-05,
         -6.72508875e-05, 8.32951276e-05, 1.48187938e-04, 4.58261148e-04,
         8.69003259e-04, -8.78512128e-05, -1.15292637e-02, 1.74978628e-02]
    p = np.poly1d(z)
    
    # Find maximum visibility for the higher-order term law
    cs = np.vstack(cs).T
    
    elcs = np.empty(cs.shape[0])
    #slcs = np.empty(cs.shape[0])
        
    vis = V_from_claret(xs,ks,cs)
    mv = vis.min(axis=0)
    mx = xs[vis.argmin(axis=0)]
    
    for idx, mvs in enumerate(mv):
        elcs[idx] = optimize.brentq(p-mvs**2,0,1)
#        elcs.append(elc)
    
    vis0 = V_from_claret(xs,[1.0],np.array([elcs]).T)
    mx0 = xs[vis0.argmin(axis=0)]
    scls = mx0/mx
      
    return elcs, scls


def V_from_claret(xs, ks, cs):
    """Find visibility as a function of x, using the formula from
    Quirrenbach (1996)
    
    Parameters
    ----------
    x: array-like
        Input x ordinate
    """
    if len(ks) != len(cs.T):
        raise UserWarning("Need one coefficient per k")
    try:
        lc = cs.T.shape[1]
    except IndexError:
        lc = 1
    Vs = np.zeros((len(xs),lc))
    norm = np.zeros(lc)
    cs0 = np.append(cs.T,1-np.sum(cs.T,axis=0)).reshape(len(cs.T)+1,lc).T
    ks0 = np.append(ks, 0)
    for k, c in zip(ks0, cs0.T):
        Vs += np.outer(sp.jv(k/2+1,xs)/xs**(k/2+1),c*2**(k/2)*sp.gamma(k/2+1))
        #Vs += c*2**(k/2)*sp.gamma(k/2+1)*sp.jv(k/2+1,xs)/xs**(k/2+1)
        norm += c/(k+2)
    return Vs/norm


def in_grid_bounds(teff, logg, e_teff=150, e_logg=0.01):
    """Tests whether a given [teff, logg] point is within the Stagger grid.
    Given we're interpolating [Fe/H], we don't take this into account.
    """
    # [teff, logg]
    grid_bounds = [[6915, 4.5],
                   [4912, 2.0],
                   [4023, 1.5],
                   [3941, 2.5],
                   [4500, 3.0],
                   [4515, 5.0],
                   [5488, 5.0]]
    
    grid = Polygon(grid_bounds)
    
    # Check the point within the default errors
    for x_i in np.arange(-1,2):
        for y_i in np.arange(-1,2):
            if not grid.contains(Point(teff + e_teff*x_i, logg + e_logg*y_i)): 
                return False
    
    return True
"""
Functions to allow calculation of limb-darkening coefficients from the STAGGER 
grid for Pionier observations. Written by Tim White. 
"""
from __future__ import division, print_function
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

def elc_stagger(inteff, insigteff, inlogg, insiglogg, infeh, insigfeh, nit):
    """ Find the equivalent limb-darkening coefficients for a given teff and 
    logg and their uncertainties. Code by Tim White.
    
    Parameters
    ----------
    inteff: float
        Effective temperature in K.
    
    insigteff: float
        Uncertainty in effective temperature.
    
    inlogg: float
        Surface gravity, log(g), in cgs units.
    
    insiglogg: float
        Uncertainty in logg
    
    infeh: float
        Metallicity [Fe/H] relative to Solar.
    
    insigfeh: float
        Uncertainty in [Fe/H]
    
    nit: float
        Number of Monte-Carlo iterations.
    
    Returns
    ----------
    wls: float array
        Observational wavelengths in um.
        
    melcs: float array
        Mean equivalent linear coefficients, one per wavelength.
        
    sigelcs: float array
        Uncertainty on equivalent linear coefficients.
        
    mscls: float array
        Mean LDD scaling parameters to be used with the equivalent linear
        coefficents, one per wavelength channel.
    
    sigscls: float array
        Uncertainty on LDD scaling parameter.
        
    mftcs: float array
        Mean four term limb darkening coefficients, of shape 4 x N wavelengths.
    
    sigftcs: float array
        Uncertainties on four term limb darkening coefficients.
    
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
    t1 = np.log10(insigteff * np.random.randn(nit) + inteff)
    l1 = np.log10(insiglogg * np.random.randn(nit) + inlogg)
    f1 = insigfeh * np.random.randn(nit) + infeh
    
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
    
    melcs = np.nanmean(elcs,axis=1)
    sigelcs = np.nanstd(elcs,axis=1)
    mscls = np.nanmean(scls,axis=1)
    sigscls = np.nanstd(scls,axis=1)
    mftcs = np.nanmean(ftcs,axis=2)
    sigftcs = np.nanstd(ftcs,axis=2)
        
    wl = np.asarray(wls)

    return wls,melcs,sigelcs,mscls,sigscls,mftcs,sigftcs



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


def in_grid_bounds(teff, logg):
    """Tests whether a given [teff, logg] point is within the Stagger grid.
    Given we're interpolating [Fe/H], we don't take this into account.
    """
    # [teff, logg]
    grid_bounds = [[6915, 4.5],
                   [4912, 2.0],
                   [4023, 1.5],
                   [3941, 2.5],
                   [4515, 5.0],
                   [5488, 5.0]]
    
    grid = Polygon(grid_bounds)
    
    in_grid = grid.contains(Point(teff, logg)) 
    
    return in_grid
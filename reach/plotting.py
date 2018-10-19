"""File to contain various plotting functions of reach.
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_diameter_comparison(diam_rel_1, diam_rel_2, diam_rel_1_dr, 
                            diam_rel_2_dr, diam_rel_1_label, diam_rel_2_label):
    """Function to compare two different measures of angular diameter (e.g. two
    different colour relations) before and after extinction correction.
    
    Parameters
    ----------
    diam_rel_1: float array
        Diameters from the first relation *before* extinction correction (mas)
    
    diam_rel_2: float array
        Diameters from the second relation *before* extinction correction (mas)
        
    diam_rel_1_dr: float array
        Diameters from the first relation *after* extinction correction (mas)
    
    diam_rel_2_dr: float array
        Diameters from the second relation *after* extinction correction (mas)
        
    diam_rel_1_label: string
        Name/label of the first relation (for legend)
        
    diam_rel_2_label: string
        Name/label of the second relation (for legend)
    """
    plt.close("all")
    plt.figure()
    plt.plot(diam_rel_1, diam_rel_2, "*", label="Reddened", alpha=0.5)
    plt.plot(diam_rel_1_dr, diam_rel_2_dr, "+", label="Corrected", alpha=0.5)
    plt.title("Angular diameter comparison for reddened/corrected photometry")
    plt.xlabel(diam_rel_1_label + "(mas)")
    plt.ylabel(diam_rel_2_label + "(mas)")
    plt.legend(loc="best")
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/angular_diameter_comp.pdf")
    

def plot_bv_intrinsic(grid):
    """Function to plot the grid of (B-V) colours for visualisation/comparison
    purposes.
    
    Parameters
    ----------
    grid: pandas dataframe
        The grid mapping Teff to SpT and (B-V)_0
    """
    plt.close("all")
    plt.figure()
    plt.plot(grid["Teff"], grid["V"], "*-", label="V (Mamajek)")
    plt.plot(grid["Teff"], grid["skV"], "o", label="V (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["IV"], "1-", label="IV (Mean V-III)")
    plt.plot(grid["Teff"], grid["III"], "x-", label="III (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["II"], "+-", label="II (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Ib"], "v-", label="Ib (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Iab"], "s-", label="Iab (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Ia"], "d-", label="Ia (Schmidt-Kaler)")
    
    flip = True
    for row_i, row in grid.iterrows():
        if flip and row["Teff"] > 2400:
            plt.text(row["Teff"], 0, row.name, fontsize=7, rotation="vertical",
                     horizontalalignment="center")
            plt.axvline(row["Teff"], alpha=0.5, color="grey", linestyle="--")
            
        flip = not flip
    plt.xlabel(r"T$_{\rm eff}$")
    plt.ylabel(r"(B-V)$_0$")
    plt.legend(loc="best")
    plt.xlim([46000,2400])
    plt.xscale("log")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/intrinsic_colours.pdf")
    
    
def plot_extinction_hists(a_mags, tgt_info):
    """Function for plotting diagnostic extinction related plots.
    """
    plt.close("all")
    plt.figure()
    
    mag_labels = ["B", "V", "J", "H", "K", "W1", "W2", "W3", "W4"]
    
    for mag_i, mag in enumerate(a_mags.T):
        plt.hist(mag[~np.isnan(mag)], bins=25, label=mag_labels[mag_i], 
                 alpha=0.25)
    
    plt.title("Distribution of stellar extinction, B through W4 filters")
    plt.xlabel("Extinction (mags)")
    plt.ylabel("# Stars")
    plt.legend(loc="best")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/extinction_hists.pdf")
    
    plt.figure()
    dists = 1000/tgt_info["Plx"]
    
    for mag_i, mag in enumerate(a_mags.T): 
        plt.plot(dists, mag, "+", label=mag_labels[mag_i])
        
    ids = tgt_info.index.values
    
    for star_i, star in enumerate(ids):
        plt.text(dists[star_i], -0.25, star, fontsize=6, rotation="vertical",
                     horizontalalignment="center")
          
    plt.xlabel("Dist (pc)")
    plt.ylabel("Extinction (mags)")
    plt.legend(loc="best")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/extinction_vs_distance.pdf")
    
    
def plot_distance_hists(tgt_info):
    """Function to plot a distance histogram of all targets.
    """
    plt.close("all")
    plt.figure()
    
    plt.hist(1000/tgt_info["Plx"][~np.isnan(tgt_info["Plx"])], bins=25, 
             alpha=0.75)
    plt.xlabel("Distance (pc)")
    plt.ylabel("# Stars")
    
    
def plot_vis2_fit(b_on_lambda, vis2, e_vis2, ldd_fit, ldd_pred, u_lld, target):
    """Function to plot squared calibrated visibilities, with curves for
    predicted diameter and fitted diameter.
    """
    # Import here to avoid mutual imports
    # TODO: remove dependency and need for this
    import reach.core as rch
    x = np.arange(1*10**7, 25*10**7, 10000)
    y1 = rch.calculate_vis2(x, ldd_fit, u_lld)
    y2 = rch.calculate_vis2(x, ldd_pred, u_lld)
    
    #plt.close("all")
    plt.figure()
    plt.errorbar(b_on_lambda, vis2, yerr=e_vis2, fmt=".", label="Data")
    plt.plot(x, y1, "--", label=r"Fit ($\theta_{\rm LDD}$=%f)" % ldd_fit)
    plt.plot(x, y2, "--", label=r"Predicted ($\theta_{\rm LDD}$=%f)" 
                                 % ldd_pred)
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(target + r" (%i vis$^2$ points)" % len(vis2))
    plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/vis2_fit.pdf")
    
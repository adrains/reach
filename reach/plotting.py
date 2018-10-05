"""File to contain various plotting functions of reach.
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_diameter_comparison(diam_series_1, diam_series_2, diam_label_1, 
                             diam_label_2):
    """
    """
    plt.scatter(diam_series_1, diam_series_2)
    plt.xlabel(diam_label_1 + "(mas)")
    plt.ylabel(diam_label_2 + "(mas)")
    

def plot_bv_intrinsic(grid):
    """
    """
    plt.close("all")
    plt.figure()
    plt.plot(grid["Teff"], grid["V"], "*-", label="V (Mamajek)")
    plt.plot(grid["Teff"], grid["skV"], "o", label="V (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["III"], "x", label="III (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["II"], "+", label="II (Schmidt-Kaler)")
    #plt.plot(grid["Teff"], grid["Ib"], label="Ib")
    #plt.plot(grid["Teff"], grid["Iab"], label="Iab")
    #plt.plot(grid["Teff"], grid["Ia"], label="Ia")
    
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
    plt.savefig("intrinsic_colours.pdf")
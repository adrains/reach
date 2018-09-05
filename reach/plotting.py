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
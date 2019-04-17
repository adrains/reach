"""File to contain various plotting functions of reach.
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.utils as rutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker

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
    
    
def plot_vis2_fit(b_on_lambda, vis2, e_vis2, ldd_fit, e_ldd_fit, ldd_pred, 
                  e_ldd_pred, u_lld, target):
    """Function to plot squared calibrated visibilities, with curves for
    predicted diameter and fitted diameter.
    """
    x = np.arange(1*10**6, 25*10**7, 10000)
    y_fit = rdiam.calculate_vis2(x, ldd_fit, u_lld)
    y_fit_low = rdiam.calculate_vis2(x, ldd_fit - e_ldd_fit, u_lld)
    y_fit_high = rdiam.calculate_vis2(x, ldd_fit + e_ldd_fit, u_lld)    
    
    y_pred = rdiam.calculate_vis2(x, ldd_pred, u_lld)
    y_pred_low = rdiam.calculate_vis2(x, ldd_pred - e_ldd_pred, u_lld)
    y_pred_high = rdiam.calculate_vis2(x, ldd_pred + e_ldd_pred, u_lld)
    
    #plt.close("all")
    plt.figure()
    
    # Plot the data points and best fit curve
    plt.errorbar(b_on_lambda, vis2, yerr=e_vis2, fmt=".", label="Data")
    
    plt.plot(x, y_fit, "--", 
             label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
    plt.fill_between(x, y_fit_low, y_fit_high, alpha=0.25)
    
    # Plot the predicted diameter with error
    plt.plot(x, y_pred, "--", 
             label=r"Predicted ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_pred, e_ldd_pred, e_ldd_pred/ldd_pred*100))
    plt.fill_between(x, y_pred_low, y_pred_high, alpha=0.25)
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(target + r" (%i vis$^2$ points)" % len(vis2))
    plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/vis2_fit.pdf")
    
    
def plot_all_vis2_fits(results, tgt_info):
    """Plot a single multi-page pdf of all fits using plot_vis2_fit
    """
    plt.close("all")
    with PdfPages("plots/bootstrapped_fits.pdf") as pdf:
        for star_i in np.arange(0, len(results)):
            #try:
            sci = results.iloc[star_i]["STAR"]
        
            pid = tgt_info[tgt_info["Primary"]==sci].index.values[0]

            n_bl = len(results.iloc[star_i]["BASELINE"])
            n_wl = len(results.iloc[star_i]["WAVELENGTH"])
            bl_grid = np.tile(results.iloc[star_i]["BASELINE"], n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], n_bl).reshape([n_bl, n_wl])
    
            b_on_lambda = (bl_grid / wl_grid).flatten()
            plot_vis2_fit(b_on_lambda, results.iloc[star_i]["VIS2"].flatten(), 
                          results.iloc[star_i]["e_VIS2"].flatten(),  
                          results.iloc[star_i]["LDD_FIT"], 
                          results.iloc[star_i]["e_LDD_FIT"], 
                          tgt_info.loc[pid, "LDD_VW3_dr"],
                          tgt_info.loc[pid, "e_LDD_VW3_dr"], 
                          tgt_info.loc[pid, "u_lld"], 
                          sci)
            pdf.savefig()
            plt.close()
            #except:
                #print("Failed on star #%i, %s" % (star_i, sci))
            
            

def plot_ldd_hists(n_ldd_fit, n_bins=10):
    """Function to plot a grid of histogram for LDD realisations from each 
    bootstrapping run.
    """
    plt.close("all")
    fig, axes = plt.subplots(4, 5)
    axes = axes.flatten()
    
    # For each science target, plot a histogram of N LDD realisations
    for sci_i, sci in enumerate(n_ldd_fit.keys()):
        ldd_percentiles = np.percentile(n_ldd_fit[sci], [50, 84.1, 15.9]) 
        err_ldd = np.abs(ldd_percentiles[1:] - ldd_percentiles[0])
    
        axes[sci_i].hist(n_ldd_fit[sci], n_bins)
        
        text_x = ldd_percentiles[0]
        text_y = axes[sci_i].get_ylim()[1] * 0.95
        
        axes[sci_i].set_title(sci)
        y_height = axes[sci_i].get_ylim()[1]
        axes[sci_i].vlines(ldd_percentiles[0], 0, y_height, 
                           linestyles="dashed")
        axes[sci_i].vlines(ldd_percentiles[1], 0, y_height, 
                           colors="red", linestyles="dotted")
        axes[sci_i].vlines(ldd_percentiles[2], 0, y_height, 
                           colors="red", linestyles="dotted")
        axes[sci_i].text(text_x, text_y, 
                         r"$\theta_{\rm LDD}=$%0.4f +%0.4f / -%0.4f" 
                         % (ldd_percentiles[0], err_ldd[0], err_ldd[1]),
                         horizontalalignment="center", fontsize=7)
        #axes[sci_i].set_xlabel("LDD (mas)")
    
    n_bs = len(n_ldd_fit[sci])
    fig.suptitle("LDD histograms for %i bootstrapping iterations" % n_bs)
    fig.tight_layout()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/ldd_hists.pdf")
    

def plot_bootstrapping_summary(results, bs_results, n_bins=20, 
                               plot_cal_info=True, sequences=None, 
                               complete_sequences=None, tgt_info=None, 
                               e_wl_frac=0.03):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    plt.close("all")
    with PdfPages("plots/bootstrapped_summary.pdf") as pdf:
        for star_i in np.arange(0, len(results)):
            # Get the science target name
            sci = results.iloc[star_i]["STAR"]
            period = results.iloc[star_i]["PERIOD"]
            sequence = results.iloc[star_i]["SEQUENCE"]
            
            if "SEQUENCE" == "combined":
                stitle = sci
            else:
                stitle = "%s (%s, %s)" % (sci, sequence, period)
            
            # -----------------------------------------------------------------
            # Plot vis^2 fits
            # -----------------------------------------------------------------
            n_bl = len(results.iloc[star_i]["BASELINE"])
            n_wl = len(results.iloc[star_i]["WAVELENGTH"])
            bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                              n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                              n_bl).reshape([n_bl, n_wl])
            
            b_on_lambda = (bl_grid / wl_grid).flatten()
            
            vis2 = results.iloc[star_i]["VIS2"].flatten()
            e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
            ldd_fit = results.iloc[star_i]["LDD_FIT"]
            e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
            
            ldd_pred = results.iloc[star_i]["LDD_PRED"]
            e_ldd_pred = results.iloc[star_i]["e_LDD_PRED"]
            u_lld = results.iloc[star_i]["u_LLD"]
            
            c_scale = results.iloc[star_i]["C_SCALE"]
            
            x = np.arange(1*10**6, 25*10**7, 10000)
            y_fit = rdiam.calc_vis2_ls(x, ldd_fit, c_scale, u_lld)
            y_fit_low = rdiam.calc_vis2_ls(x, ldd_fit - e_ldd_fit, c_scale, u_lld)
            y_fit_high = rdiam.calc_vis2_ls(x, ldd_fit + e_ldd_fit, c_scale, u_lld)    
    
            y_pred = rdiam.calc_vis2_ls(x, ldd_pred, 1, u_lld)
            y_pred_low = rdiam.calc_vis2_ls(x, ldd_pred - e_ldd_pred, 1, u_lld)
            y_pred_high = rdiam.calc_vis2_ls(x, ldd_pred + e_ldd_pred, 1, u_lld)
    
            fig, axes = plt.subplots(1, 2)
            axes = axes.flatten()
            
            # Setup lower panel for residuals
            divider = make_axes_locatable(axes[0])
            res_ax = divider.append_axes("bottom", size="20%", pad=0)
            axes[0].figure.add_axes(res_ax)
    
            # Plot the data points and best fit curve
            axes[0].errorbar(b_on_lambda, vis2, xerr=b_on_lambda*e_wl_frac,
                            yerr=e_vis2, fmt=".", 
                            label="Data", elinewidth=0.1, capsize=0.2, 
                            capthick=0.1)
    
            axes[0].plot(x, y_fit, "--", 
                     label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                           % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
            #axes[0].fill_between(x, y_fit_low, y_fit_high, alpha=0.25,
                                 #color="C1")
    
            # Plot the predicted diameter with error
            label=(r"Predicted ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_pred, e_ldd_pred, e_ldd_pred/ldd_pred*100))
            axes[0].plot(x, y_pred, "--", label=label)
            #axes[0].fill_between(x, y_pred_low, y_pred_high, alpha=0.25, 
                                 #color="C2")
    
            #axes[0].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            axes[0].set_ylabel(r"Visibility$^2$")
            axes[0].set_title(stitle + r" (%i vis$^2$ points)" % len(vis2))
            axes[0].legend(loc="best")
            axes[0].set_xlim([0.0,25E7])
            axes[0].set_ylim([0.0,1.1])
            axes[0].grid()
            
            # Plot residuals below the vis2 plot
            axes[0].set_xticks([])
            residuals = vis2 - rdiam.calc_vis2_ls(b_on_lambda, ldd_fit, c_scale,
                                                u_lld)
            
            res_ax.errorbar(b_on_lambda, residuals, xerr=b_on_lambda*e_wl_frac,
                            yerr=e_vis2, fmt=".", 
                            label="Residuals", elinewidth=0.1, capsize=0.2, 
                            capthick=0.1)
            res_ax.set_xlim([0.0,25E7])
            res_ax.hlines(0, 0, 25E7, linestyles="dotted")
            res_ax.set_ylabel("Residuals")
            res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            
            # -----------------------------------------------------------------
            # Plot calibrator angular diameters and magnitudes for diagnostic
            # purposes
            # -----------------------------------------------------------------
            if plot_cal_info:
                sci_h = tgt_info[tgt_info["Primary"]==sci]["Hmag"].values[0]
                sci_e_h = tgt_info[tgt_info["Primary"]==sci]["e_Hmag"].values[0]
                sci_jsdc = tgt_info[tgt_info["Primary"]==sci]["JSDC_LDD"].values[0]
                
                if sequence == "combined":
                    stars = set(sequences[(period, sci, "bright")][::2]
                                + sequences[(period, sci, "faint")][::2])
                else:
                    stars = sequences[(period, sci, sequence)][::2]
                    
                stars = [star.replace("_", "").replace(".","").replace(" ", "") 
                            for star in stars]
                            
                stars = rutils.get_unique_key(tgt_info, stars)
                
                # Print science details
                text_x = axes[0].get_xlim()[1] * 5/8 
                text_y = 3/4 + 0.125
                text = "C = %0.2f" % c_scale
                axes[0].text(text_x, text_y, text, fontsize="x-small",
                             horizontalalignment="center")
                
                text_x = axes[0].get_xlim()[1] * 5/8 
                text_y = 3/4 + 0.1
                
                text = (r"%s, $\theta_{\rm LDD}=%0.3f$, "
                        r"$\theta_{\rm JSDC}$=%0.3f, Hmag=%0.2f$\pm$ %0.2f" 
                        % (sci, ldd_fit, sci_jsdc, sci_h, sci_e_h))
                
                axes[0].text(text_x, text_y, text, fontsize="xx-small",
                             horizontalalignment="center")
                
                cal_ldd = []
                cal_h = []
                 
                # Print ESO's quality information about the observations
                if sequence == "bright" or sequence == "combined":
                    # Bright
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 + 0.075
                    
                    text = "Bright Quality: %s" % complete_sequences[(period, sci, "bright")][1]
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
                                 
                # Faint
                if sequence == "faint" or sequence == "combined":
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 + 0.05
                    
                    text = "Faint Quality: %s" % complete_sequences[(period, sci, "faint")][1]
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
                 
                             
                # Print calibrator details
                for star_i, star in enumerate(stars):
                    star_info = tgt_info.loc[star]
                    
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 - (0.025 * star_i)
                    
                    # Cross out stars we have ignored
                    if star_info["Quality"] == "BAD":
                        st = u"\u0336"
                        star = st.join(star) + st
                    
                    ldd_diff = star_info["LDD_pred"] - star_info["JSDC_LDD"]
                    
                    text = (r"%s, Hmag=%0.2f$\pm$ %0.2f, $\theta_{\rm LDD}=%0.3f\pm %0.3f$ (%s), "
                            r"$\theta_{\rm JSDC}=%0.3f\pm %0.3f$,   "
                            r"[$\theta_{\rm diff}=%0.3f$]" 
                            % (star, star_info["Hmag"], star_info["e_Hmag"],
                               star_info["LDD_pred"], 
                               star_info["e_LDD_pred"], star_info["LDD_rel"],
                               star_info["JSDC_LDD"], star_info["e_JSDC_LDD"],
                               ldd_diff))
                    
                    cal_ldd.append(star_info["LDD_pred"])
                    cal_h.append(star_info["Hmag"])
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
            
                                 
                # Print average 
                text = r"$\theta_{\rm LDD (AVG)}=%0.3f$" % np.nanmean(cal_ldd)
                axes[0].text(text_x, text_y-0.05, text, fontsize="x-small",
                                 horizontalalignment="center")
                                 
                text = r"Hmag$_{\rm AVG}=%0.3f$" % np.nanmean(cal_h)
                axes[0].text(text_x, text_y-0.075, text, fontsize="x-small",
                                 horizontalalignment="center")
                
            
            # -----------------------------------------------------------------
            # Plot histograms
            # -----------------------------------------------------------------
            axes[1].hist(bs_results[stitle]["LDD_FIT"].values.tolist(), n_bins)
        
            text_y = axes[1].get_ylim()[1]
        
            axes[1].set_title(stitle + r" (${\rm N}_{\rm bootstraps} = $%i)" 
                             % len(bs_results[stitle]["LDD_FIT"].values.tolist()))
            y_height = axes[1].get_ylim()[1]
            axes[1].vlines(ldd_fit, 0, y_height, linestyles="dashed")
            axes[1].vlines(ldd_fit-e_ldd_fit, 0, y_height, colors="red", 
                           linestyles="dotted")
            axes[1].vlines(ldd_fit+e_ldd_fit, 0, y_height, colors="red", 
                           linestyles="dotted")
            axes[1].text(ldd_fit, text_y, r"$\theta_{\rm LDD}=%0.4f \pm%0.4f$" 
                         % (ldd_fit, e_ldd_fit), horizontalalignment="center") 
            
            plt.gcf().set_size_inches(16, 9)
            pdf.savefig()
            plt.close()


def plot_single_vis2(results, e_wl_frac=0.03):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    for star_i in np.arange(0, len(results)):
        # Get the science target name
        sci = results.iloc[star_i]["STAR"]
        period = results.iloc[star_i]["PERIOD"]
        sequence = results.iloc[star_i]["SEQUENCE"]
        
        stitle = "%s (%s, P%s)" % (sci, sequence, period)
        
        # -----------------------------------------------------------------
        # Plot vis^2 fits
        # -----------------------------------------------------------------
        n_bl = len(results.iloc[star_i]["BASELINE"])
        n_wl = len(results.iloc[star_i]["WAVELENGTH"])
        bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                          n_wl).reshape([n_wl, n_bl]).T
        wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                          n_bl).reshape([n_bl, n_wl])
        
        b_on_lambda = (bl_grid / wl_grid).flatten()
        
        vis2 = results.iloc[star_i]["VIS2"].flatten()
        e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
        ldd_fit = results.iloc[star_i]["LDD_FIT"]
        e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
        
        u_lld = results.iloc[star_i]["u_LLD"]
        
        c_scale = results.iloc[star_i]["C_SCALE"]
        
        x = np.arange(1*10**6, 25*10**7, 10000)
        y_fit = rdiam.calc_vis2_ls(x, ldd_fit, c_scale, u_lld)

        plt.close("all")
        fig, ax = plt.subplots()
        
        # Setup lower panel for residuals
        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes("bottom", size="20%", pad=0)
        ax.figure.add_axes(res_ax)

        # Plot the data points and best fit curve
        ax.errorbar(b_on_lambda, vis2, xerr=b_on_lambda*e_wl_frac,
                        yerr=e_vis2, fmt=".", 
                        label="Data", elinewidth=0.1, capsize=0.2, 
                        capthick=0.1)

        ax.plot(x, y_fit, "--", 
                 label=r"Fit ($\theta_{\rm LDD}$=%0.3f mas)"# $\pm$ %f mas)" 
                       % (ldd_fit))#, e_ldd_fit))

        #axes[0].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
        ax.set_ylabel(r"Visibility$^2$")
        ax.set_title(stitle)
        ax.legend(loc="best")
        ax.set_xlim([0.0,10E7])
        ax.set_ylim([0.0, c_scale+0.1])
        ax.grid()
        
        # Plot residuals below the vis2 plot
        ax.set_xticks([])
        residuals = vis2 - rdiam.calc_vis2_ls(b_on_lambda, ldd_fit, c_scale,
                                            u_lld)
        
        res_ax.errorbar(b_on_lambda, residuals, xerr=b_on_lambda*e_wl_frac,
                        yerr=e_vis2, fmt=".", 
                        label="Residuals", elinewidth=0.1, capsize=0.2, 
                        capthick=0.1)
        res_ax.set_xlim([0.0,10E7])
        res_ax.hlines(0, 0, 25E7, linestyles="dotted")
        res_ax.set_ylabel("Residuals")
        res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
        
        # -----------------------------------------------------------------
        # Save figs
        # -----------------------------------------------------------------
        #plt.gcf().set_size_inches(16, 9)
        plt.savefig("plots/single_vis2/vis2_%s_%s_%s.pdf" % (sci, period, sequence))
        plt.close()


def plot_paper_vis2_fits(results, bs_results, n_rows=8, n_cols=2):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    plt.close("all")
    with PdfPages("paper/seq_vis2_plots.pdf") as pdf:
        # Figure out how many sets of plots are needed
        num_sets = int(np.ceil(len(bs_results) / n_rows / n_cols))
        n_rows_init = n_rows
        
        # For every set, save a page
        for set_i in np.arange(0, num_sets):
            # Ensure we don't have an incomplete set of subplots
            if set_i + 1 == num_sets:
                n_rows = int((len(bs_results) - set_i*n_rows*n_cols) / n_cols)
            
            # Setup the axes
            fig, axes = plt.subplots(n_rows, n_cols)#), sharex=True, sharey=True)
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            axes = axes.flatten()
    
            for star_i in np.arange(set_i*n_rows*n_cols, (set_i+1)*n_rows*n_cols):
                # Subplot index < n_rows
                plt_i = star_i % (n_rows * n_cols)
               
                # Might not be able to finish
                if star_i >= len(results):
                    axes[plt_i].axis("off")
                    continue
            
                # Get the science target name
                sci = results.iloc[star_i]["STAR"]
                period = results.iloc[star_i]["PERIOD"]
                sequence = results.iloc[star_i]["SEQUENCE"]
            
                stitle = "%s (%s, %s)" % (sci, sequence, period)
            
                print("%i, %i, %s %s %s" % (set_i, plt_i, sci, period, sequence))
            
                # -----------------------------------------------------------------
                # Plot vis^2 fits
                # -----------------------------------------------------------------
                n_bl = len(results.iloc[star_i]["BASELINE"])
                n_wl = len(results.iloc[star_i]["WAVELENGTH"])
                bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                                  n_wl).reshape([n_wl, n_bl]).T
                wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                                  n_bl).reshape([n_bl, n_wl])
            
                b_on_lambda = (bl_grid / wl_grid).flatten()
            
                vis2 = results.iloc[star_i]["VIS2"].flatten()
                e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
                ldd_fit = results.iloc[star_i]["LDD_FIT"]
                e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
            
                u_lld = results.iloc[star_i]["u_LLD"]
            
                c_scale = results.iloc[star_i]["C_SCALE"]
            
                x = np.arange(1*10**6, 25*10**7, 10000)
                y_fit = rdiam.calc_vis2_ls(x, ldd_fit, c_scale, u_lld)
                y_fit_low = rdiam.calc_vis2_ls(x, ldd_fit - e_ldd_fit, c_scale, u_lld)
                y_fit_high = rdiam.calc_vis2_ls(x, ldd_fit + e_ldd_fit, c_scale, u_lld)    
            
                # Setup lower panel for residuals
                divider = make_axes_locatable(axes[plt_i])
                res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
                axes[plt_i].figure.add_axes(res_ax, sharex=axes[plt_i])
    
                # Plot the data points and best fit curve
                axes[plt_i].errorbar(b_on_lambda, vis2, yerr=e_vis2, fmt=".", 
                                label="Data", elinewidth=0.1, capsize=0.2, 
                                capthick=0.1, markersize=0.5)
    
                axes[plt_i].plot(x, y_fit, "--", linewidth=0.25,
                         label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                               % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
                
                # Annotate the sequence name
                xx = (axes[plt_i].get_xlim()[1] - axes[plt_i].get_xlim()[0]) * 0.05
                yy = (axes[plt_i].get_ylim()[1] - axes[plt_i].get_ylim()[0]) * 0.05
                axes[plt_i].text(xx, yy, stitle, fontsize="xx-small")
                
                # Set up ticks
                axes[plt_i].set_xlim([0.0,10E7])
                axes[plt_i].set_ylim([0.0,1.1])
                
                axes[plt_i].set_xticklabels([])
                
                axes[plt_i].tick_params(axis="both", top=True, right=True)
                res_ax.tick_params(axis="y", right=True)
                
                maj_loc = plticker.MultipleLocator(base=0.2)
                min_loc = plticker.MultipleLocator(base=0.1)
                
                axes[plt_i].yaxis.set_major_locator(maj_loc)
                axes[plt_i].yaxis.set_minor_locator(min_loc)
                

                
            
                # Plot residuals below the vis2 plot
                residuals = vis2 - rdiam.calc_vis2_ls(b_on_lambda, ldd_fit, c_scale,
                                                    u_lld)
            
                res_ax.errorbar(b_on_lambda, residuals, yerr=e_vis2, fmt=".", 
                                label="Residuals", elinewidth=0.1, capsize=0.2, 
                                capthick=0.1, markersize=0.5)
                res_ax.set_xlim([0.0,10E7])
                res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
                #res_ax.set_ylabel("Residuals")
                #res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
                
                plt.setp(axes[plt_i].get_xticklabels(), fontsize="xx-small")
                plt.setp(axes[plt_i].get_yticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_xticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_yticklabels(), fontsize="xx-small")
                res_ax.xaxis.offsetText.set_fontsize("xx-small")
                res_ax.yaxis.offsetText.set_fontsize("xx-small")
                
                
                # Only show res_ax x labels on the bottom row
                #if not (plt_i >= (n_rows*n_cols - n_cols)):
                    #res_ax.set_xticklabels([])
                
                # Only show res_ax y labels if on left
                #if not (plt_i % n_cols == 0):
                    #res_ax.set_yticklabels([])
            # -----------------------------------------------------------------
            # Finalise
            # -----------------------------------------------------------------
            fig.text(0.5, 0.005, r"Spatial Frequency (rad$^{-1})$", ha='center')
            fig.text(0.005, 0.5, r"Visibility$^2$", va='center', rotation='vertical')
            
            plt.gcf().set_size_inches(8, 8*(n_rows/n_rows_init))
            plt.tight_layout(pad=1.0)
            pdf.savefig()
            plt.close()


 

def plot_lit_diam_comp(tgt_info):
    """Plot for paper comparing measured LDD vs any literature values
    """
    # Load in the literature diameters
    lit_diam_file = "data/literature_diameters.tsv"
    lit_diam_info = pd.read_csv(lit_diam_file, sep="\t", header=0)
    
    instruments = set(lit_diam_info[lit_diam_info["has_diam"]]["instrument"])
    
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="20%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # For every different instrument, plot the comparison between our results
    # and those from the literature
    for instrument in instruments:
        #print(instrument)
        mask = np.logical_and(lit_diam_info["has_diam"], 
                              lit_diam_info["instrument"]==instrument).values
        
        # Initialise arrays
        calc_diams = []
        e_calc_diams = []
        lit_diams = []
        e_lit_diams = []
        
        for index, star in lit_diam_info[mask].iterrows():
            # Get the two LDDs to compare
            lit_diams.append(star["theta_ldd"])
            e_lit_diams.append(star["e_theta_ldd"])
            calc_diams.append(tgt_info.loc[star["HD"]]["ldd_final"])
            e_calc_diams.append(tgt_info.loc[star["HD"]]["e_ldd_final"])
            
            # Plot the name of the star
            ax.annotate(star["Primary"], xy=(calc_diams[-1], lit_diams[-1]), 
                        xytext=(calc_diams[-1]+0.01, lit_diams[-1]+0.01), 
                        arrowprops=dict(facecolor="black", width=0.1, 
                                        headwidth=0.1),
                        fontsize="xx-small")
                        
        # Plot the points
        ax.errorbar(calc_diams, lit_diams, xerr=e_calc_diams, yerr=e_lit_diams, 
                    fmt="o", label=instrument, elinewidth=0.5, capsize=0.8, 
                    capthick=0.5)
            
        # Plot residuals
        ax.set_xticks([])
        residuals = np.array(lit_diams) / np.array(calc_diams)
            
        res_ax.errorbar(calc_diams, residuals, xerr=e_calc_diams, 
                        yerr=e_lit_diams, fmt="o", elinewidth=0.5, capsize=0.8,
                        capthick=0.5)
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10), np.arange(0, 10), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Setup the rest of the plot
    ax.set_ylabel(r"$\theta_{\rm Lit}$")  
    res_ax.set_xlabel(r"$\theta_{\rm PIONIER}$")   
    res_ax.set_ylabel(r"$\theta_{\rm Lit} / \theta_{\rm PIONIER}$")  
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/lit_diam_comp.pdf")    
    


def plot_colour_rel_diam_comp(tgt_info, colour_rel="V-W3", cbar="feh"):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    """
    # Format the colour relation
    colour_rel_col = "LDD_" + colour_rel.replace("-", "")
    
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    fit_diams = []
    e_fit_diams = []
    colour_rel_diams = []
    e_colour_rel_diams = []
    fehs = []
    teffs = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        # If star doesn't have a diameter using this relation, skip
        if np.isnan(star_data[colour_rel_col]) or not star_data["in_paper"]:
            continue
        elif colour_rel=="V-K" and star_data["LDD_rel"] != colour_rel_col:
            continue
        
        # Get the two LDDs to compare
        fit_diams.append(star_data["ldd_final"])
        e_fit_diams.append(star_data["e_ldd_final"])
        colour_rel_diams.append(star_data[colour_rel_col])
        e_colour_rel_diams.append(star_data["e_%s" % colour_rel_col])
        fehs.append(star_data["FeH_rel"])
        teffs.append(star_data["teff_final"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (fit_diams[-1]**2 + colour_rel_diams[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 0.1
        
        if len(xy_txt) > 0 and (xy < sep).any():
            xx = 0.025
            yy = 0.4
        else:
            xx = 0.025
            yy = 0.3 
        
        # Plot the name of the star
        ax.annotate(star_data["Primary"], xy=(fit_diams[-1], 
                    colour_rel_diams[-1]), 
                    xytext=(fit_diams[-1]+xx, colour_rel_diams[-1]-yy), 
                    arrowprops=dict(facecolor="black", width=0.1, 
                                    headwidth=0.1),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(fit_diams, colour_rel_diams, xerr=e_fit_diams, 
                yerr=e_colour_rel_diams, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(colour_rel_diams) / np.array(fit_diams)
        
    res_ax.errorbar(fit_diams, residuals, xerr=e_fit_diams, 
                    yerr=e_colour_rel_diams, fmt=".", elinewidth=0.5, 
                    capsize=0.8, capthick=0.5, zorder=1)
    
    # Overplot scatter points so we can have [Fe/H] as colours
    if cbar == "feh":
        
        scatter = ax.scatter(fit_diams, colour_rel_diams, c=fehs, marker="o", 
                             zorder=2)
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("[Fe/H]")
        res_ax.scatter(fit_diams, residuals, c=fehs, marker="o", zorder=2)
    
    # Overplot scatter points so we can have Teff as colours
    elif cbar == "teff":
        scatter = ax.scatter(fit_diams, colour_rel_diams, c=teffs, marker="o", 
                             zorder=2, cmap="magma")
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label(r"T$_{\rm eff}$")
        res_ax.scatter(fit_diams, residuals, c=teffs, marker="o", zorder=2, 
                       cmap="magma")
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10), np.arange(0, 10), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    loc = plticker.MultipleLocator(base=0.1)
    res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    
    # Setup the rest of the plot
    ax.set_ylabel(r"$\theta_{\rm %s}$" % colour_rel)  
    res_ax.set_xlabel(r"$\theta_{\rm PIONIER}$")   
    res_ax.set_ylabel(r"$\theta_{\rm %s} / \theta_{\rm PIONIER}$" % colour_rel)  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/colour_rel_diam_comp_%s_%s.pdf" % (colour_rel, cbar))     
            

    
def plot_vis2(oi_fits_file, star_id):
    """
    """
    
    #fig, ax = plt.figure()
    plt.close("all")
    
    vis2, e_vis2, baselines, wavelengths = rdiam.extract_vis2(oi_fits_file)
    
    n_bl = len(baselines)
    n_wl = len(wavelengths)
    bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
    wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
            
    b_on_lambda = (bl_grid / wl_grid).flatten()
    
    plt.errorbar(b_on_lambda, vis2.flatten(), yerr=e_vis2.flatten(), fmt=".")
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(r"%s (%i vis$^2$ points)" % (star_id, len(vis2.flatten())))
    #plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    #plt.gcf().set_size_inches(16, 9)
    #plt.savefig("plots/vis2_fit.pdf")
    

def plot_c_hist(results, n_bins=5):
    """Plot histograms of the scaling/intercept parameter C.
    """
    faint_cs = results[results["SEQUENCE"]=="faint"]["C_SCALE"].values.tolist()
    faint_cs.sort()
    faint_cs = faint_cs[:-1]
    bright_cs = results[results["SEQUENCE"]=="bright"]["C_SCALE"].values.tolist()
    
    plt.hist(faint_cs, bins=n_bins, label="Faint", alpha=0.60)
    plt.hist(bright_cs, bins=n_bins, label="Bright", alpha=0.60)
    
    plt.text(1.08, 5, r"C$_{\rm med}$ (bright) = %0.2f" % np.median(bright_cs))
    plt.text(1.08, 4.5, r"C$_{\rm med}$ (faint) = %0.2f" % np.median(faint_cs))
    
    plt.xlabel("C")
    plt.ylabel("#")
    plt.legend(loc="best")
    plt.savefig("plots/c_hist.png")
    
    
    
def presentation_vis2_plot():
    """Plot spatial frequency coverage of PAVO and POINIER for use as a visial
    aid when giving talks.
    """
    # CHARA
    chara_min_bl = 34
    chara_max_bl = 330
    chara_min_lambda = 630 * 10**-9
    chara_max_lambda = 950 * 10**-9
    chara_lims = np.array([chara_min_bl/chara_max_lambda, 
                                   chara_max_bl/chara_min_lambda])
                                
    # PIONIER
    vlti_min_bl = 58
    vlti_max_bl = 132
    vlti_min_lambda = 1533 * 10**-9
    vlti_max_lambda = 1773 * 10**-9
    vlti_lims = np.array([vlti_min_bl/vlti_max_lambda, 
                                     vlti_max_bl/vlti_min_lambda])
    
    # Diameters to plot
    ldds = [0.5, 1.0, 2.0, 4.0]
    u_lld = 0.3
    c_scale = 1
    xmax = 55*10**7
    nsteps = 25
    
    freqs = np.arange(1*10**6, xmax, 10000)
    chara_freqs = np.arange(chara_lims[0], chara_lims[1], 
                           (chara_lims[1]-chara_lims[0])/nsteps)
    vlti_freqs = np.arange(vlti_lims[0], vlti_lims[1], 
                          (vlti_lims[1]-vlti_lims[0])/nsteps)
    
    plt.close("all")
    
    for ldd in ldds:
        vis2 = rdiam.calc_vis2_ls(freqs, ldd, c_scale, u_lld)
        
        plt.plot(freqs, vis2, label=r"$\theta_{\rm LD}$ = %0.1f mas" % ldd)
    
        # PIONIER
        ldd_rad = ldd / 1000 / 3600 / 180 * np.pi
        vlti_vis2 = rdiam.calc_vis2_ls(vlti_freqs, ldd, c_scale, u_lld)
        plt.plot(vlti_freqs, vlti_vis2, ".", color="darkred") 
    
        # CHARA
        ldd_rad = ldd / 1000 / 3600 / 180 * np.pi
        chara_vis2 = rdiam.calc_vis2_ls(chara_freqs, ldd, c_scale, u_lld)
        plt.plot(chara_freqs, chara_vis2, "+", color="blue")
        
    plt.text(0.5*xmax, 0.95, "PAVO: 34-330m, R band", color="blue", ha='center')
    plt.text(0.5*xmax, 0.9, "PIONIER: 58-132m, H band", color="darkred", ha='center')
    
    plt.xlim([0.0, xmax])
    plt.ylim([0.0, 1.0])
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/presentation_vis2_vs_ldd.pdf")
    plt.savefig("plots/presentation_vis2_vs_ldd.png")
    
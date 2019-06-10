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
import matplotlib.cm as cm
import matplotlib.transforms as transforms

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
    
    
def plot_vis2_fit(sfreq, vis2, e_vis2, ldd_fit, e_ldd_fit, ldd_pred, 
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
    plt.errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", label="Data")
    
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
    
            sfreq = (bl_grid / wl_grid).flatten()
            plot_vis2_fit(sfreq, results.iloc[star_i]["VIS2"].flatten(), 
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
            hd_id = results.iloc[star_i]["HD"]
            period = results.iloc[star_i]["PERIOD"]
            sequence = results.iloc[star_i]["SEQUENCE"]
            
            print(sci)
            
            if sequence == "combined":
                stitle = sci
                star_id = sci
            else:
                stitle = "%s (%s, %s)" % (sci, sequence, period)
                star_id = (sci, sequence, period)
            
            # -----------------------------------------------------------------
            # Plot vis^2 fits
            # -----------------------------------------------------------------
            n_bl = len(results.iloc[star_i]["BASELINE"])
            n_wl = len(results.iloc[star_i]["WAVELENGTH"])
            bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                              n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                              n_bl).reshape([n_bl, n_wl])
            
            sfreq = (bl_grid / wl_grid).flatten()
            
            vis2 = results.iloc[star_i]["VIS2"].flatten()
            e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
            ldd_fit = results.iloc[star_i]["LDD_FIT"]
            e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
            
            ldd_pred = tgt_info.loc[hd_id]["LDD_pred"]
            e_ldd_pred = tgt_info.loc[hd_id]["e_LDD_pred"]
            
            u_lambdas = ['u_lambda_0', 'u_lambda_1', 'u_lambda_2', 'u_lambda_3',
                         'u_lambda_4', 'u_lambda_5']
            
            u_lld = np.mean(tgt_info.loc[hd_id][u_lambdas])
            
            #c_scale = results.iloc[star_i]["C_SCALE"]
            c_scale = 1
            
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
            axes[0].errorbar(sfreq, vis2, xerr=sfreq*e_wl_frac,
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
            residuals = vis2 - rdiam.calc_vis2_ls(sfreq, ldd_fit, c_scale,
                                                u_lld)
            
            res_ax.errorbar(sfreq, residuals, xerr=sfreq*e_wl_frac,
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
            axes[1].hist(bs_results[star_id]["LDD_FIT"].values.tolist(), n_bins)
        
            text_y = axes[1].get_ylim()[1]
        
            axes[1].set_title(stitle + r" (${\rm N}_{\rm bootstraps} = $%i)" 
                             % len(bs_results[star_id]["LDD_FIT"].values.tolist()))
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
        
        sfreq = (bl_grid / wl_grid).flatten()
        
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
        ax.errorbar(sfreq, vis2, xerr=sfreq*e_wl_frac,
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
        residuals = vis2 - rdiam.calc_vis2_ls(sfreq, ldd_fit, c_scale,
                                            u_lld)
        
        res_ax.errorbar(sfreq, residuals, xerr=sfreq*e_wl_frac,
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


def plot_paper_vis2_fits(results, n_rows=8, n_cols=2):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    plt.close("all")
    with PdfPages("paper/seq_vis2_plots.pdf") as pdf:
        # Figure out how many sets of plots are needed
        num_sets = int(np.ceil(len(results) / n_rows / n_cols))
        n_rows_init = n_rows
        
        # For every set, save a page
        for set_i in np.arange(0, num_sets):
            # Ensure we don't have an incomplete set of subplots
            if set_i + 1 == num_sets:
                n_rows = int((len(results) - set_i*n_rows*n_cols) / n_cols)
            
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
            
                # -------------------------------------------------------------
                # Plot vis^2 fits
                # -------------------------------------------------------------
                n_bl = len(results.iloc[star_i]["BASELINE"])
                n_wl = len(results.iloc[star_i]["WAVELENGTH"])
                bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                                  n_wl).reshape([n_wl, n_bl]).T
                wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                                  n_bl).reshape([n_bl, n_wl])
            
                sfreq = (bl_grid / wl_grid).flatten()
            
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
                axes[plt_i].errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
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
                residuals = vis2 - rdiam.calc_vis2_ls(sfreq, ldd_fit, c_scale,
                                                    u_lld)
            
                res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", 
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



def plot_joint_seq_paper_vis2_fits(tgt_info, results, n_rows=3, n_cols=2):
    """Plot the rescaled simultaneous fits for multiple sequences
    """
    #results = results.drop(3)
    #results.set_index(np.arange(len(results)))
       
    plt.close("all")
    with PdfPages("paper/joint_seq_vis2_plots.pdf") as pdf:
        # Figure out how many sets of plots are needed
        num_sets = int(np.ceil(len(results) / n_rows / n_cols))
        n_rows_init = n_rows
        
        # For every set, save a page
        for set_i in np.arange(0, num_sets):
            # Ensure we don't have an incomplete set of subplots
            if set_i + 1 == num_sets:
                n_rows = int((len(results) - set_i*n_rows*n_cols) / n_cols)
            
            # Setup the axes
            fig, axes = plt.subplots(n_rows, n_cols)
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            axes = axes.flatten()
    
            for star_i in np.arange(set_i*n_rows_init*n_cols, 
                                    (set_i+1)*n_rows_init*n_cols):
                # Subplot index < n_rows
                plt_i = star_i % (n_rows * n_cols)
               
                # Might not be able to finish
                if star_i >= len(results):
                    break
            
                # Get the science target name
                sci = results.iloc[star_i]["STAR"]
                hd_id = tgt_info[tgt_info["Primary"]==sci].index.values[0]
                
                period = results.iloc[star_i]["PERIOD"]
                sequence = results.iloc[star_i]["SEQUENCE"]
            
                stitle = "%s (%s, %s)" % (sci, sequence, period)
            
                print("%i, %i, [%i] %s %s %s" % (set_i, plt_i, star_i, sci, 
                                                 period, sequence))
            
                # Get the C params, and u_lambda values
                u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
                s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
                
                u_lambdas = tgt_info.loc[hd_id][u_lambda_cols].values
                s_lambdas = tgt_info.loc[hd_id][s_lambda_cols].values
                
                c_scale = results.iloc[star_i]["C_SCALE"]
                
                n_points = [12] * len(c_scale)
                
                c_array = np.hstack([c_scale[ni]*np.ones(n) 
                             for ni, n in enumerate(n_points)])
                
                #colours = ["mistyrose", "coral", "orangered", "red", 
                #           "firebrick", "maroon"]
                cmap = cm.get_cmap("magma")
                colours = [cmap(i) for i in np.arange(0.96,0,-0.16)]
                           
                wl_um = [1.533, 1.581, 1.629, 1.677, 1.725, 1.773]
                wl_lbl = [r"%s$\,\mu$m" % wl for wl in wl_um]
                
                # -------------------------------------------------------------
                # Plot vis^2 fits
                # -------------------------------------------------------------
                n_bl = len(results.iloc[star_i]["BASELINE"])
                n_wl = len(results.iloc[star_i]["WAVELENGTH"])
                
                # Setup lower panel for residuals
                divider = make_axes_locatable(axes[plt_i])
                res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
                axes[plt_i].figure.add_axes(res_ax, sharex=axes[plt_i])
                
                residuals_all = []
                e_vis2_all = []
                
                # For each wavelength dimension
                for wl_i in np.arange(6):
                    # Need to do 1 plot per wavelength channel
                    bls = results.iloc[star_i]["BASELINE"]
                    wls = results.iloc[star_i]["WAVELENGTH"]
                    sfreq = (bls / wls[wl_i])[:len(c_array)]
                    
                    vis2 = results.iloc[star_i]["VIS2"][:, wl_i]
                    e_vis2 = results.iloc[star_i]["e_VIS2"][:, wl_i]
                    ldd_fit = results.iloc[star_i]["LDD_FIT"]
                    e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
                    
                    # Add a mask to not plot any bad data
                    valid_i = (((vis2 >= 0) & (e_vis2 > 0) 
                               & ~np.isnan(vis2)))[:len(c_array)]
                    
                    # Normalise vis2 and scale ldd_fit
                    # TODO: Fix the uncertainty over the length of each seq
                    vis2 = vis2[:len(c_array)] / c_array
                    e_vis2 = e_vis2[:len(c_array)]
                    ldd_fit = ldd_fit * s_lambdas[wl_i]
                    
                    u_lambda = u_lambdas[wl_i]
                    
                    # Apply mask
                    vis2 = vis2[valid_i]
                    e_vis2 = e_vis2[valid_i]
                    sfreq = sfreq[valid_i]
            
                    x = np.arange(1*10**6, 25*10**7, 10000)
                    y_fit = rdiam.calc_vis2_ls(x, ldd_fit, 1.0, u_lambda) 
    
                    # Plot the data points and best fit curve
                    axes[plt_i].errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
                                    label=wl_lbl[wl_i], elinewidth=0.1, capsize=0.2, 
                                    capthick=0.1, markersize=0.5, color=colours[wl_i])
    
                    axes[plt_i].plot(x, y_fit, "--", linewidth=0.25, 
                                     color=colours[wl_i])
                
                    # Plot residuals below the vis2 plot
                    residuals = vis2 - rdiam.calc_vis2_ls(sfreq, ldd_fit, 1.0,
                                                    u_lambda)
            
                    res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", 
                                elinewidth=0.1, capsize=0.2, 
                                capthick=0.1, markersize=0.5, color=colours[wl_i])
                                
                    #axes[plt_i].legend(loc="best", fontsize="xx-small")
                    
                    # Record all points for figuring out axis ticks later
                    residuals_all = np.hstack((residuals_all, residuals))
                    e_vis2_all = np.hstack((e_vis2_all, e_vis2))
                    
                # Annotate the sequence name
                xx = (axes[plt_i].get_xlim()[1] - axes[plt_i].get_xlim()[0]) * 0.05
                yy = (axes[plt_i].get_ylim()[1] - axes[plt_i].get_ylim()[0]) * 0.05
                axes[plt_i].text(xx, yy, rutils.format_id(sci), 
                                 fontsize="xx-small")
                
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
                
                # Work out the residual axis spacing
                res_sep_maj = np.abs(np.max(residuals_all + e_vis2_all) 
                               - np.min(residuals_all - e_vis2_all)) / 4
                               
                res_sep_maj = np.round(res_sep_maj*2, 2) / 2
                res_sep_min = res_sep_maj / 2
                
                res_maj_loc = plticker.MultipleLocator(base=res_sep_maj)
                res_min_loc = plticker.MultipleLocator(base=res_sep_min)
                res_ax.yaxis.set_major_locator(res_maj_loc)
                res_ax.yaxis.set_minor_locator(res_min_loc)
                
                res_ax.set_xlim([0.0,10E7])
                res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
                
                plt.setp(axes[plt_i].get_xticklabels(), fontsize="xx-small")
                plt.setp(axes[plt_i].get_yticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_xticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_yticklabels(), fontsize="xx-small")
                res_ax.xaxis.offsetText.set_fontsize("xx-small")
                res_ax.yaxis.offsetText.set_fontsize("xx-small")
                
            # -----------------------------------------------------------------
            # Finalise
            # -----------------------------------------------------------------
            fig.text(0.5, 0.005, r"Spatial Frequency (rad$^{-1})$", ha='center')
            fig.text(0.005, 0.5, r"Visibility$^2$", va='center', rotation='vertical')
            
            plt.gcf().set_size_inches(8, 11*(n_rows/n_rows_init))
            plt.tight_layout(pad=1.0)
            pdf.savefig()
            plt.close()    


def plot_sidelobe_vis2_fit(tgt_info, results, sci="lamSgr", star_i=15):
    """Plot the zoomed in fitted sidelobe
    """
    plt.close("all")
    # Setup the axes
    fig, axes = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Get the science target name
    hd_id = tgt_info[tgt_info["Primary"]==sci].index.values[0]

    # Get the C params, and u_lambda values
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    
    u_lambdas = tgt_info.loc[hd_id][u_lambda_cols].values
    s_lambdas = tgt_info.loc[hd_id][s_lambda_cols].values
    
    c_scale = results.iloc[star_i]["C_SCALE"]
    
    n_points = [12] * len(c_scale)
    
    c_array = np.hstack([c_scale[ni]*np.ones(n) 
                 for ni, n in enumerate(n_points)])
    
    cmap = cm.get_cmap("magma")
    colours = [cmap(i) for i in np.arange(0.96,0,-0.16)]
               
    wl_um = [1.533, 1.581, 1.629, 1.677, 1.725, 1.773]
    wl_lbl = [r"%s$\,\mu$m" % wl for wl in wl_um]
    
    # -----------------------------------------------------------------
    # Plot vis^2 fits
    # -----------------------------------------------------------------
    n_bl = len(results.iloc[star_i]["BASELINE"])
    n_wl = len(results.iloc[star_i]["WAVELENGTH"])
    
    # Setup lower panel for residuals
    divider = make_axes_locatable(axes)
    res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
    axes.figure.add_axes(res_ax, sharex=axes)
    
    # For each wavelength dimension
    for wl_i in np.arange(6):
        # Need to do 1 plot per wavelength channel
        bls = results.iloc[star_i]["BASELINE"]
        wls = results.iloc[star_i]["WAVELENGTH"]
        sfreq = (bls / wls[wl_i])[:len(c_array)]
        
        vis2 = results.iloc[star_i]["VIS2"][:, wl_i]
        e_vis2 = results.iloc[star_i]["e_VIS2"][:, wl_i]
        ldd_fit = results.iloc[star_i]["LDD_FIT"]
        e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
        
        # Normalise vis2 and scale ldd_fit
        # TODO: Fix the uncertainty over the length of each seq
        vis2 = vis2[:len(c_array)] / c_array
        e_vis2 = e_vis2[:len(c_array)]
        ldd_fit = ldd_fit * s_lambdas[wl_i]

        u_lambda = u_lambdas[wl_i]

        x = np.arange(1*10**6, 25*10**7, 10000)
        y_fit = rdiam.calc_vis2_ls(x, ldd_fit, 1.0, u_lambda) 

        # Plot the data points and best fit curve
        axes.errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
                        label=wl_lbl[wl_i], elinewidth=0.1, capsize=0.2, 
                        capthick=0.1, markersize=4.0, color=colours[wl_i])

        axes.plot(x, y_fit, "--", linewidth=0.25, color=colours[wl_i])
    
        # Plot residuals below the vis2 plot
        residuals = vis2 - rdiam.calc_vis2_ls(sfreq, ldd_fit, 1.0,
                                        u_lambda)

        res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", elinewidth=0.1, 
                    capsize=0.2, capthick=0.1, markersize=4.0, 
                    color=colours[wl_i])
                    
        axes.legend(loc="best", fontsize="xx-small")
    
    # Set up ticks and axes
    axes.set_xlim([5.5E7,9.5E7])
    axes.set_ylim([0.0,0.016])
    
    axes.set_xticklabels([])
    
    axes.tick_params(axis="both", top=True, right=True)
    res_ax.tick_params(axis="y", right=True)
    
    maj_loc = plticker.MultipleLocator(base=0.004)
    min_loc = plticker.MultipleLocator(base=0.00025)
    
    axes.yaxis.set_major_locator(maj_loc)
    axes.yaxis.set_minor_locator(min_loc)
    axes.set_ylabel(r"Visibility$^2$", va='center', rotation='vertical')
    
    res_maj_loc = plticker.MultipleLocator(base=0.005)
    res_min_loc = plticker.MultipleLocator(base=0.001)
    
    res_ax.yaxis.set_major_locator(res_maj_loc)
    res_ax.yaxis.set_minor_locator(res_min_loc)
    
    res_ax.set_xlim([5.5E7,9.5E7])
    res_ax.set_ylim([-0.01,0.01])
    res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
    res_ax.set_ylabel("Residuals")
    
    res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
    
    plt.setp(axes.get_xticklabels(), fontsize="xx-small")
    plt.setp(axes.get_yticklabels(), fontsize="xx-small")
    plt.setp(res_ax.get_xticklabels(), fontsize="xx-small")
    plt.setp(res_ax.get_yticklabels(), fontsize="xx-small")
    res_ax.xaxis.offsetText.set_fontsize("xx-small")
    res_ax.yaxis.offsetText.set_fontsize("xx-small")
        
    plt.tight_layout(pad=1.0)
    plt.savefig("paper/lam_sgr_sidelobe.pdf")  

 

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
            ax.annotate(rutils.format_id(star["Primary"].replace(" ", "")), 
                        xy=(calc_diams[-1], lit_diams[-1]), 
                        xytext=(calc_diams[-1]+0.01, lit_diams[-1]+0.01), 
                        #arrowprops=dict(facecolor="black", width=0.1, 
                        #                headwidth=0.1),
                        fontsize="xx-small")
                        
        # Plot the points
        ax.errorbar(calc_diams, lit_diams, xerr=e_calc_diams, yerr=e_lit_diams, 
                    fmt="o", label=instrument, elinewidth=0.5, capsize=0.8, 
                    capthick=0.5)
            
        # Plot residuals
        ax.set_xticks([])
        residuals = np.array(lit_diams) / np.array(calc_diams)
        err_res = np.array(e_calc_diams) / np.array(calc_diams)
            
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
    plt.savefig("paper/lit_diam_comp.pdf")    
    


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
            yy = 0.2
        else:
            xx = 0.025
            yy = 0.15 
        
        # Plot the name of the star
        ax.annotate(rutils.format_id(star_data["Primary"]), xy=(fit_diams[-1], 
                    colour_rel_diams[-1]), 
                    xytext=(fit_diams[-1]+xx, colour_rel_diams[-1]-yy), 
                    #arrowprops=dict(facecolor="black", width=0.1, 
                    #                headwidth=0.1),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(fit_diams, colour_rel_diams, xerr=e_fit_diams, 
                yerr=e_colour_rel_diams, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(colour_rel_diams) / np.array(fit_diams)
    err_res = np.array(e_fit_diams) / np.array(fit_diams)
        
    res_ax.errorbar(fit_diams, residuals, xerr=err_res, 
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
    plt.savefig("paper/colour_rel_diam_comp_%s_%s.pdf" % (colour_rel, cbar))     
            


def plot_casagrande_teff_comp(tgt_info):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    """
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    final_teffs = []
    e_final_teffs = []
    casagrande_teffs = []
    e_casagrande_teffs = []
    fehs = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        
        if star_data["Primary"] in ["gamPav", "HD187289"]:
            continue
        
        # Get the two LDDs to compare
        final_teffs.append(star_data["teff_final"])
        e_final_teffs.append(star_data["e_teff_final"])
        casagrande_teffs.append(star_data["teff_casagrande"])
        e_casagrande_teffs.append(star_data["e_teff_casagrande"])
        fehs.append(star_data["FeH_rel"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (final_teffs[-1]**2 + casagrande_teffs[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 50
        
        if len(xy_txt) > 0 and (xy < sep).any():
            xx = 25
            yy = 200
        else:
            xx = 25
            yy = 100
        
        # Plot the name of the star
        ax.annotate(rutils.format_id(star_data["Primary"]),  
                    xy=(final_teffs[-1],casagrande_teffs[-1]), 
                    xytext=(final_teffs[-1]+xx, casagrande_teffs[-1]-yy), 
                    #arrowprops=dict(facecolor="black", width=0.0, 
                                    #headwidth=0.0),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(final_teffs, casagrande_teffs, xerr=e_final_teffs, 
                yerr=e_casagrande_teffs, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(casagrande_teffs) - np.array(final_teffs)
        
    res_ax.errorbar(final_teffs, residuals, xerr=e_final_teffs, 
                    yerr=e_casagrande_teffs, fmt=".", elinewidth=0.5, 
                    capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(final_teffs, casagrande_teffs, c=fehs, marker="o", 
                         zorder=2)
                         

    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("[Fe/H]")
    res_ax.scatter(final_teffs, residuals, c=fehs, marker="o", zorder=2)
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10000, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"T$_{\rm eff, Casagrande+2010}$")  
    res_ax.set_xlabel(r"T$_{\rm eff, PIONIER}$")  
    res_ax.set_ylabel(r"T$_{\rm eff, residuals}$")  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("paper/teff_comp_casagrande.pdf")  


def plot_lit_teff_comp(tgt_info):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    """
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    final_teffs = []
    e_final_teffs = []
    lit_teffs = []
    e_lit_teffs = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        
        if star_data["Primary"] in ["gamPav", "HD187289"]:
            continue
        
        # Get the two LDDs to compare
        #final_teffs.append(star_data["teff_final"])
        #e_final_teffs.append(star_data["e_teff_final"])
        final_teffs.append(star_data["teff_casagrande"])
        e_final_teffs.append(star_data["e_teff_casagrande"])
        lit_teffs.append(star_data["Teff"])
        e_lit_teffs.append(star_data["e_teff"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (final_teffs[-1]**2 + lit_teffs[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 0.1
        
        if len(xy_txt) > 0 and (xy < sep).any():
            xx = 0.025
            yy = 0.4
        else:
            xx = 0.025
            yy = 0.3 
        
        # Plot the name of the star
        ax.annotate(star_data["Primary"], xy=(final_teffs[-1], 
                    lit_teffs[-1]), 
                    xytext=(final_teffs[-1]+xx, lit_teffs[-1]-yy), 
                    arrowprops=dict(facecolor="black", width=0.1, 
                                    headwidth=0.1),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(final_teffs, lit_teffs, xerr=e_final_teffs, 
                yerr=e_lit_teffs, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(lit_teffs) - np.array(final_teffs)
        
    res_ax.errorbar(final_teffs, residuals, xerr=e_final_teffs, 
                    yerr=e_lit_teffs, fmt=".", elinewidth=0.5, 
                    capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(final_teffs, lit_teffs, marker="o", 
                         zorder=2)
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10000, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"T$_{\rm eff, literature}$")  
    res_ax.set_xlabel(r"T$_{\rm eff, PIONIER}$")  
    res_ax.set_ylabel(r"T$_{\rm eff, residuals}$")  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/teff_comp_lit_casagrande.pdf")  

    
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
            
    sfreq = (bl_grid / wl_grid).flatten()
    
    plt.errorbar(sfreq, vis2.flatten(), yerr=e_vis2.flatten(), fmt=".")
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(r"%s (%i vis$^2$ points)" % (star_id, len(vis2.flatten())))
    #plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    #plt.gcf().set_size_inches(16, 9)
    #plt.savefig("plots/vis2_fit.pdf")



def plot_fbol_comp(tgt_info):
    """Plot a comparison of the sampled values of fbol from each filter to 
    check whether they are consistent or not.
    """
    plt.close("all")
    plt.figure()
    
    # Define bands to reference, construct new headers
    bands = ["Hp", "BT", "VT", "BP", "RP"]
    e_bands = ["e_%s" % band for band in bands] 
    f_bol_bands = ["f_bol_%s" % band for band in bands] 
    e_f_bol_bands = ["e_f_bol_%s" % band for band in bands] 
    
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData
    
    tf = 7.5
    
    colours = ["green", "blue", "orange", "deepskyblue", "red"]
    
    mask = np.logical_and(tgt_info["Science"], tgt_info["in_paper"])
    
    ids = tgt_info[mask]["Primary"]
    fbol = tgt_info[mask][f_bol_bands]
    e_fbol = tgt_info[mask][e_f_bol_bands]
    
    
    
    for band_i, (fband, e_fband) in enumerate(zip(f_bol_bands, e_f_bol_bands)):
        plt.errorbar(ids, fbol[fband], yerr=e_fbol[e_fband], 
                     fmt=".", zorder=1, label="", ecolor="black",
                     transform=trans+offset(-tf*band_i))
        plt.scatter(ids, fbol[fband], s=2**6, c=colours[band_i], label=bands[band_i],
                    zorder=2, transform=trans+offset(-tf*band_i), 
                    marker="$%s$" % bands[band_i])
                        
    plt.xlabel("Star")
    plt.ylabel(r"Flux (ergs s$^{-1}$ cm $^{-2}$)")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("fbol_comp.pdf")
    
    

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
    
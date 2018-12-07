"""
"""
from __future__ import division, print_function

import glob
import numpy as np
import pandas as pd
import reach.pndrs as rpndrs
import reach.utils as rutils
import reach.diameters as rdiam
import matplotlib.pylab as plt
from decimal import Decimal
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

def run_vis2_diagnostics(complete_sequences, base_path):
    """Inspect vis2 as a function of time for baseline dropouts.
    """
    # For each sequence, go through the fringe files and plot vis2 over time 
    # on a per baseline basis. This should give ~37 plots.
    
    # Construct a data cube of dimensions [n_seq, n_bl, n_vis2] 
    
    #n_seq = len(complete_sequences)
    #n_bl = 6
    #n_vis2 = 60
    
    #seq_bl_vis2 = np.zeros([n_seq, n_bl, n_vis, n_vis2])
    
    vis2_per_bl = {}
    
    for seq_i, seq in enumerate(complete_sequences.keys()):
        print("Sequence %i --> %s" % (seq_i, seq))
        # Get the files
        fringe_files = [data[7] for data in complete_sequences[seq][2] if data[8]=="FRINGE"]
        fringe_files = [ff.replace(".fits.Z", "_oidata.fits") for ff in fringe_files]
        fringe_files = [ff.replace("all_sequences", "complete_sequences") for ff in fringe_files]
        fringe_files = [ff.replace("/PIONI", "_v3.73_abcd/PIONI") for ff in fringe_files]
        fringe_files.sort()                     
        
        # Create empty dataframe, and pre-allocate memory
        cols = ["MJD", "AT1-AT2", "BL 1-2", "AT1-AT3", "BL 1-3", 
                "AT1-AT4", "BL 1-4", "AT2-AT3", "BL 2-3", "AT2-AT4", "BL 2-4",
                "AT3-AT4", "BL 3-4"] 
        seq_recs = pd.DataFrame(index=np.arange(0, len(fringe_files)), columns=cols)
        
        # Extract from each one
        for ff_i, oi_fits_file in enumerate(fringe_files):
            #oi_fits_file = ff.replace(".fits.Z", "_oidata.fits")
            
            # Open file and populate dataframe
            with fits.open(oi_fits_file, memmap=False) as oifits:
                # Get the telescope/station data
                tels = oifits[3].data
                
                # Step through the vis2 data
                for bs_i, baseline_data in enumerate(oifits[4].data):
                    # Save MJD data
                    seq_recs.loc[ff_i, "MJD"] = baseline_data[2]
                    
                    # Get column name, and save vis2 data
                    tel_index = baseline_data[8] - [1,1]
                    
                    tel_col = "%s-%s" % (tels[tel_index[0]][0], 
                                         tels[tel_index[1]][0])
                   
                    seq_recs.loc[ff_i, tel_col] = baseline_data[4]
                    
                    # Save the UV baseline data
                    uv = np.sqrt(baseline_data[6]**2 + baseline_data[7]**2)
                    bl_col = "BL %i-%i" % (baseline_data[8][0], baseline_data[8][1])
                    seq_recs.loc[ff_i, bl_col] = uv
                    
        # Save the pandas array and move on
        vis2_per_bl[seq] = seq_recs
        
    return vis2_per_bl
  
            
def plot_vis2_diagnostics(vis2_per_bl):
    """
    """
    plt.close("all")
    fig, axes = plt.subplots(6, 7)
    axes = axes.flatten()
    
    vis2_cols = ["AT1-AT2", "AT1-AT3", "AT1-AT4", "AT2-AT3", "AT2-AT4", 
                "AT3-AT4"] 
    bl_cols = ["BL 1-2", "BL 1-3", "BL 1-4", "BL 2-3", "BL 2-4", "BL 3-4"] 
    
    for seq_i, seq in enumerate(vis2_per_bl.keys()):
        for vis2_col, bl_col in zip(vis2_cols, bl_cols):
            # Average the vis2 across the wavelength dimension
            mean_vis2 = np.mean(np.vstack(vis2_per_bl[seq][vis2_col].values), 
                                axis=1)
            
            # Average the baseline data to get a rough idea of relative lengths
            mean_bl = np.mean(vis2_per_bl[seq][bl_col].values)
            
            # Construct the label for the legend
            label = vis2_col + "(%i m)" % mean_bl
            
            axes[seq_i].plot(vis2_per_bl[seq]["MJD"], mean_vis2, ".-", 
                             label=label)

        axes[seq_i].set_title(seq)
        
        axes[seq_i].set_ylim([0,1])
        axes[seq_i].legend(loc="best")
        
    fig.suptitle("vis^2 vs MJD")
    #fig.tight_layout()
    plt.gcf().set_size_inches(32, 32)
    plt.savefig("plots/vis2_vs_time.pdf")
    
    
    
def plot_vis2_diagnostics_time_wl(vis2_per_bl):
    """
    """
    plt.close("all")
    
    vis2_cols = ["AT1-AT2", "AT1-AT3", "AT1-AT4", "AT2-AT3", "AT2-AT4", 
                "AT3-AT4"] 
    bl_cols = ["BL 1-2", "BL 1-3", "BL 1-4", "BL 2-3", "BL 2-4", "BL 3-4"] 
    
    wavelengths = [1.533e-06, 1.581e-06, 1.629e-06, 1.677e-06, 1.725e-06, 1.773e-06]
    
    with PdfPages("plots/vis2_vs_time_vs_baseline.pdf") as pdf:
        for seq_i, seq in enumerate(vis2_per_bl.keys()):
            # Initialise subplot
            fig, axes = plt.subplots(3, 3)
            axes = axes.flatten()
    
            for bl_i, (vis2_col, bl_col) in enumerate(zip(vis2_cols, bl_cols)):
                # Retrive all the vis2 values for this baseline
                vis2 = np.vstack(vis2_per_bl[seq][vis2_col].values)
                
                # For each wavelength *column*, plot
                for wl_i in np.arange(0, 6):
                    axes[bl_i].plot(vis2_per_bl[seq]["MJD"], vis2[:, wl_i], ".-", 
                                 label="%0.3E m" % Decimal(wavelengths[wl_i]))

            
                # Average the baseline data to get a rough idea of relative lengths
                mean_bl = np.mean(vis2_per_bl[seq][bl_col].values)
            
                # Construct the label for the legend
                title = vis2_col + "(%i m)" % mean_bl
            
                #axes[seq_i].plot(vis2_per_bl[seq]["MJD"], mean_vis2, ".-", 
                #                 label=label)
                axes[bl_i].set_title(title)
                axes[bl_i].set_ylim([0,1])
                axes[bl_i].legend(loc="top", fontsize="xx-small")
            
            fig.suptitle(seq)
            #fig.suptitle("vis^2 vs MJD")
            fig.tight_layout()
            plt.gcf().set_size_inches(16, 9)
            #plt.savefig("plots/vis2_vs_time.pdf")
            pdf.savefig()
            plt.close("all")
            

def calibrate_calibrators(sequences, complete_sequences, base_path, tgt_info,
                          n_pred_ldd, e_pred_ldd):
    """
    Assume that every sequence has three calibrators - we can simply do three
    loops of the calibration routine, turning the science off each time, and 
    flipping each calibrator to science in order going through.
    
    To do this, we'll need to flip all the science targets to "BAD" so that
    they're ignored. Hopefully this doesn't affect the kappa matrices. Should
    probably also flip them to calibrator status in case the pipeline complains
    that an entire science target is being ignored.
    """
    run_local = False
    already_calibrated = False
    do_random_ifg_sampling = False
    do_gaussian_diam_sampling = False
    test_one_seq_only = False
    do_ldd_fitting = False
    n_bootstraps = 1
    pred_ldd_col = "LDD_VW3_dr"
    e_pred_ldd_col = "e_LDD_VW3_dr"
    base_path = "/priv/mulga1/arains/pionier/complete_sequences/%s_v3.73_abcd/"
    results_path = "/home/arains/code/reach/diagnostics/"
    
    # Get a list of the calibrators, in order, turn one calibrator per sequence
    # to science per calibration run - should need three runs
    calibrators = np.vstack(sequences.values())[:,::2]
    
    # Reset any currently "BAD" quality targets
    tgt_info["Quality"] = [None] * len(tgt_info)
    
    # Set all science targets to "BAD" so that we ignore them when calibrating
    science = rutils.get_unique_key(tgt_info, np.vstack(sequences.values())[:,1])
    tgt_info.loc[science, "Quality"] = ["BAD"] * len(science)
    
    # Write nightly pndrs.i scripts
    for cal_i in np.arange(0,3):
        print("Now running for set number %i of calibrators" % cal_i)
        
        # Set all science targets to "BAD" so that we ignore them when calibrating
        tgt_info["Science"] = [False] * len(tgt_info)
        
        # Get the cal_i-th column of calibrators, and get their unique IDs
        cal_ids = rutils.get_unique_key(tgt_info, calibrators[:,cal_i])
        
        # Flip each of the calibrators in cals_to_sci to be a science target
        tgt_info.loc[cal_ids, "Science"] = [True] * len(cal_ids)
        
        # Run calibration (no bootstrapping)
        rpndrs.run_n_bootstraps(sequences, complete_sequences, base_path, 
                                    tgt_info,  n_pred_ldd, e_pred_ldd, 
                                    n_bootstraps, run_local=run_local, 
                                    already_calibrated=already_calibrated,
                                    do_random_ifg_sampling=do_random_ifg_sampling,
                                    results_path=results_path, 
                                    do_ldd_fitting=do_ldd_fitting)
                                    
        # Compile results and look at visibility curves
        #oifiles = glob.glob("results/*oidata


def plot_calibrator_vis2(cal_folder="diagnostics/"):
    """
    """
    cal_oifits = glob.glob(cal_folder + "*fits")
    plt.close("all")
    with PdfPages("plots/calibrator_vis2.pdf") as pdf:
        # Initialise subplot
        fig, axes = plt.subplots(12, 12)
        axes = axes.flatten()
        
        for cal_i, cal in enumerate(cal_oifits):
            cal_id = cal.split("SCI")[-1].split("oidata")[0].replace("_","")
            #rplt.plot_vis2(cal, cal_id)
            
            vis2, e_vis2, baselines, wavelengths = rdiam.extract_vis2(cal)
    
            n_bl = len(baselines)
            n_wl = len(wavelengths)
            bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
            
            b_on_lambda = (bl_grid / wl_grid).flatten()
    
            axes[cal_i].errorbar(b_on_lambda, vis2.flatten(), 
                                 yerr=e_vis2.flatten(), fmt=".", 
                                 elinewidth=0.1, markersize=0.25)
            
            #axes[cal_i].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            #axes[cal_i].set_ylabel(r"Visibility$^2$")
            axes[cal_i].set_title(r"%s (%i vis$^2$ points)" % (cal_id, len(vis2.flatten())), fontsize="xx-small")
            #plt.legend(loc="best")
            axes[cal_i].set_xlim([0.0,25E7])
            axes[cal_i].set_ylim([0.0,1.0])
            axes[cal_i].tick_params(axis="both", labelsize="xx-small")
            axes[cal_i].grid()
            
        #fig.tight_layout()
        plt.gcf().set_size_inches(16, 16)
        pdf.savefig()
        #plt.close("all")
            
            
def inspect_bootstrap_iterations(oifits_files):
    """
    """
    times = []
    baselines = []
    wavelengths = []
    stations = []
    
    for file_i, file in enumerate(oifits_files):
        with fits.open(file, memmap=False) as oifits:
            wavelengths.append(tuple(oifits[2].data["EFF_WAVE"]))
            times.append(tuple(oifits[4].data["MJD"]))
            
            bl = np.sqrt(oifits[4].data["UCOORD"]**2 + oifits[4].data["VCOORD"]**2)
            bl.sort()
            
            baselines.append(tuple(bl))
            
            sta = tuple([tuple(pair) for pair in oifits[4].data["STA_INDEX"]])
            
            stations.append(sta)
            
    return set(times), set(baselines), set(wavelengths), set(stations)
    
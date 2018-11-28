"""
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import reach.pndrs as rpndrs
import matplotlib.pylab as plt
from astropy.io import fits

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
        
        #import pdb
        #pdb.set_trace()
        
        # Create empty dataframe, and pre-allocate memory
        cols = ["MJD", "AT1-AT2", "AT1-AT3", "AT1-AT4", "AT2-AT3", "AT2-AT4", "AT3-AT4"] 
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
                    
                    #uv = np.sqrt(baseline_data[6]**2 + baseline_data[7]**2)
                    
                    # Get column name, and save vis2 data
                    sta_index = baseline_data[8] - [1,1]
                    
                    #import pdb
                    #pdb.set_trace()
                    
                    sta_str = "%s-%s" % (tels[sta_index[0]][0], tels[sta_index[1]][0])
                   
                    seq_recs.loc[ff_i, sta_str] = baseline_data[4]
                    
        # Save the pandas array and move on
        vis2_per_bl[seq] = seq_recs
        
    return vis2_per_bl
            
def plot_vis2_diagnostics(vis2_per_bl):
    """
    """
    plt.close("all")
    fig, axes = plt.subplots(6, 7)
    axes = axes.flatten()
    
    for seq_i, seq in enumerate(vis2_per_bl.keys()):
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT1-AT2"].values),axis=1), ".-", label="AT1-AT2")
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT1-AT3"].values),axis=1), ".-", label="AT1-AT3")
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT1-AT4"].values),axis=1), ".-", label="AT1-AT4")
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT2-AT3"].values),axis=1), ".-", label="AT2-AT3")
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT2-AT4"].values),axis=1), ".-", label="AT2-AT4")
        axes[seq_i].plot(vis2_per_bl[seq]["MJD"], np.mean(np.vstack(vis2_per_bl[seq]["AT3-AT4"].values),axis=1), ".-", label="AT3-AT4")
        axes[seq_i].set_title(seq)
        
        axes[seq_i].set_ylim([0,1])
        
    fig.suptitle("vis^2 vs MJD")
    #fig.tight_layout()
    plt.gcf().set_size_inches(32, 32)
    plt.savefig("plots/vis2_vs_time.pdf")
"""Module to handle interacting with the PIONIER data reduction pipeline, pndrs
"""
from __future__ import division, print_function
import os
import glob
import datetime
import numpy as np
import pandas as pd
import reach.diameters as rdiam
from shutil import copyfile
from astropy.io import fits
from astropy.time import Time
from collections import OrderedDict

# -----------------------------------------------------------------------------
# pndrs Affiliated Functions
# -----------------------------------------------------------------------------
def save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                pred_ldd, e_pred_ldd,
                base_path="/priv/mulga1/arains/pionier/complete_sequences/",
                dir_suffix="_v3.73_abcd", run_local=False):
    """This is a function to create and save the oiDiam.fits files referenced
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_oiDiam.fits containing an
    empty primary HDU, and a fits table with LDD and e_LDD for each star listed
    alphabetically.
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequence: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    dir_suffix: str
        String suffix on the end of each folder of calibrated data
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    ldd_col: str
        Column of predicted LDD from tgt_info to use.
    
    e_ldd_col:
        Column of predicted e_LDD from tgt_info to use.
    """
    print("\n", "-"*79, "\n", "\tSaving Nightly oidiam files\n", "-"*79)
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
    
    diam_files_written = 0
    
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
        # Note that several stars are observed multiple times under different
        # primary IDs, so we need to check HD and Bayer IDs as well
        for star in nights[night]:
            prim_id = tgt_info[tgt_info["Primary"]==star].index
            
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["Bayer_ID"]==star].index
                
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["HD_ID"]==star].index
            
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
        
        # We need to compile entries for multiple targets with the same name
        # due to the non-unique/inconsistent IDs initially sent to ESO. These
        # are stored in the following columns of the input table.
        ref_ids = ["Ref_ID_1", "Ref_ID_2", "Ref_ID_3"]
        
        recs = []
        
        # For each non-null reference ID, collate magnitude, LDD, and sci/cal
        # Rename the reference ID column in the pandas dataframe, then stack
        for ref_id in ref_ids:
            
            rec = tgt_info.loc[ids][tgt_info.loc[ids][ref_id].notnull()]
            rec = rec[["Hmag", "Kmag", "Vmag", "Science", ref_id]]
            
            # Insert the diameters - these are now coming from a separate data
            # structure to facilitate potential bootstrapping. The variable
            # appropriate_ids are only those IDs found to have the given ref_id
            # since the ldd data structures don't know about this
            appropriate_ids = rec.index.values

            rec.insert(0,"pred_LDD", pred_ldd[appropriate_ids].values)    
            rec.insert(1,"e_pred_LDD", e_pred_ldd[appropriate_ids].values[0])     
                       
            rec.rename(columns={ref_id:"Ref_ID"}, inplace=True)
            
            if len(rec) > 0:
                recs.append(rec.copy(deep=True))

        rec = pd.concat(recs)

        # Invert, as column is for calibrator status
        rec.Science =  np.abs(rec.Science - 1)
        rec["INFO"] = np.repeat("(V-W3) diameter from Boyajian et al. 2014",
                                len(rec))

        rec.insert(0,"TARGET_ID", np.arange(1,len(rec)+1))
        
        max_id = np.max([len(id) for id in rec["Ref_ID"]])
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
            fname = dir + "/" + night + "_oiDiam.fits" 
            hdu.writeto(fname, output_verify="warn", overwrite=True)
            
            # Done, move to the next night
            print("...wrote %s, %s" % (night, nights[night]))
            diam_files_written += 1
        else:
            # The directory does not exist, flag
            print("...directory '%s' does not exist" % dir)
    print("%i oiDiam.fits files written" % diam_files_written)    
    return nights


def save_nightly_pndrs_script(complete_sequences, tgt_info, 
            base_path="/priv/mulga1/arains/pionier/complete_sequences/",
            dir_suffix="_v3.73_abcd", run_local=False):
    """This is a function to create and save the pndrs script files referenced
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_pndrsScript.i containing a list
    of pndrs commands to run in order to customise the calibration procedure.
    
    Important here are the following commands:
        - Ignore some observations: oiFitsFlagOiData
        - Split the night: oiFitsSplitNight

    Parameters
    ----------
    complete_sequence: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    dir_suffix: str
        String suffix on the end of each folder of calibrated data
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    """
    print("\n", "-"*79, "\n", "\tSaving Nightly pndrs Scripts\n", "-"*79)
    
    # Figure out what targets share nights
    # Of the form nights[night] = [mjd1, mjd2, ..., mjdn]
    sequence_times = {}
    
    for seq in complete_sequences.keys():
        # Get the string representing the night, YYYY-MM-DD
        night = complete_sequences[seq][0]
        
        # Get the datetime objects representing the first and last observations
        # of each sequence, and add or subtract a small increment as to bracket
        # the entire sequence between the time range. Convert these to MJD.
        delta = datetime.timedelta(seconds=10)
        first_ob = Time(complete_sequences[seq][2][0][4] - delta).mjd
        last_ob = Time(complete_sequences[seq][2][-1][4] + delta).mjd
        
        if night not in sequence_times:
            sequence_times[night] = [first_ob, last_ob]
        else:
            sequence_times[night] += [first_ob, last_ob]
            sequence_times[night].sort()
    
    # These lines are written to YYYY-MM-DD_pndrsScript.i alongside the MJD
    # to split upon
    line_split_1 = 'yocoLogInfo, "Split night to isolate SCI-CAL sequences";'
    line_split_2 = 'oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;'
    
    # These lines are written to exclude bad calibrators, with the variable
    # 'startend' being a list with an MJD range to exclude
    line_exclude_1 = 'yocoLogInfo,"Ignore bad calibrators";'
    line_exclude_2 = ('oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis,' 
                      'tlimit=startend;')
    
    pndrs_scripts_written = 0
    no_script_nights = 0
    
    # Get a list of the target durations
    durations = calculate_target_durations(complete_sequences)
    bad_durations = select_only_bad_target_durations(durations, tgt_info)
    
    for night in sequence_times:
        # It is only meaningful to write a script if we need to split the night
        # (i.e. if more than one sequence has been observed, that is there are
        # 4 or more MJD entries) or we have bad calibrators to exclude 
        if len(sequence_times[night]) <= 2 and len(bad_durations[night]) < 1:
            no_script_nights += 1
            continue 
        
        # Save the fits file to the night directory
        if not run_local:
            dir = base_path + night + dir_suffix
        else:
            dir = "test/"
        
        # This night requires a script to be written. When splitting the night,
        # we can neglect the first and last times as there are no observations
        # before or after these times respectively, and we only need one of any
        # pair of star1 end MJD and star2 start MJD    
        if os.path.exists(dir):
            fname = dir + "/" + night + "_pndrsScript.i" 
            
            with open(fname, "w") as nightly_script:
                # Split the night
                if len(sequence_times[night]) > 2:
                    nightly_script.write(line_split_1 + "\n")
                    cc = "cc = %s;\n" % sequence_times[night][1:-1:2]
                    nightly_script.write(cc)
                    nightly_script.write(line_split_2)
                
                # Rule out bad calibrators
                # Note that this currently assumes only one bad calibrator per
                # science target - fix is to use star_i in string formatting
                if len(bad_durations[night]) >= 1:
                    for star_i, bad_cal in enumerate(bad_durations[night]):
                        nightly_script.write(line_exclude_1 + "\n")
                        startend = "startend = %s;\n" % bad_cal[1:]
                        nightly_script.write(startend)
                        nightly_script.write(line_exclude_2 + "\n")
            
            # Done, move to the next night
            print("...wrote %s, night split into %s, bad calibrators: %s" 
                  % (night, len(sequence_times[night])//2, 
                     len(bad_durations[night])))
            pndrs_scripts_written += 1
            
        else:
            # The directory does not exist, flag
            print("...directory '%s' does not exist" % dir)
            
    print("%i pndrs.i scripts written" % pndrs_scripts_written)
    print("%i no script nights" % no_script_nights)        



def calculate_target_durations(complete_sequences):
    """For each night of observations, return the start and end time of 
    *sequential* observations associated with a given target.
    
    A typical CAL1-SCI1-CAL2-SCI2-CAL3 sequence observes each target 5 times 
    before moving on to the next target in the sequence. This function gets
    the first and last times of each block for the purpose of later excluding 
    bad calibrators.
    
    Parameters
    ----------
    complete_sequence: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    Returns
    -------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
    """
    # Initialise results dict
    sequence_durations = {}
    
    # Time difference to go before start of first observations, or after end of
    # last observation
    delta = datetime.timedelta(seconds=10)
    
    for seq in complete_sequences.keys():
        # Get a mapping of all target IDs to their times
        times = [(ob[2], ob[4]) for ob in complete_sequences[seq][2]]
        
        durations = [[times[0][0], Time(times[0][1] - delta).mjd, 0]]
        
        night = complete_sequences[seq][0]
        
        tgt_i = 0
        
        for (tgt, time) in times:
            # Same target
            if tgt == durations[tgt_i][0]:
                # Update the end time
                durations[tgt_i][2] = Time(time + delta).mjd
            
            # We've moved on
            else:
                tgt_i += 1
                durations.append([tgt, Time(time - delta).mjd, 0])
            
        # All done
        sequence_durations[night] = durations
        
    return sequence_durations


def select_only_bad_target_durations(sequence_durations, tgt_info):
    """Takes the output of calculate_target_durations, and compares to the 
    target quality values in tgt_info, returning only durations for only those
    targets which we wish to exclude from the calibration process.
    
    Parameters
    ----------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
        
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    Returns
    -------
    bad_durations: dict
        Dict of same form as sequence_durations, but containing only the 
        calibrators we wish to exclude.
    """
    # Initialise results dict
    bad_durations = {}
    
    for night in sequence_durations:
        bad_durations[night] = []
        
        for star in sequence_durations[night]:
            # Get the star info, making sure to check primary, bayer, and HD
            # IDs given the non-unique IDs used
            prim_id = tgt_info[tgt_info["Primary"]==star[0]].index
        
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["Bayer_ID"]==star[0]].index
            
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["HD_ID"]==star[0]].index
        
            try:
                assert len(prim_id) > 0
            except:
                print("...failed on %s, %s" % (night, star))
            
            # Check if it is a bad calibrator, and if so add to return dict
            if tgt_info.loc[prim_id[0]]["Quality"] == "BAD":
                bad_durations[night].append(star)
                
    return bad_durations
    

def calibrate_all_observations(reduced_data_folders):
    """Calls the PIONIER data reduction pipeline for each folder of reduced
    data from within Python.
    
    Parameters
    ----------
    reduced_data_folders: string array
        List of folder paths to run the calibration pipeline on
    """
    # List to record times for the start and end of each night to calibrate
    times = []
    
    # Run the PIONIER calibration pipeline for every folder with reduced data
    # TODO: capture the output and inspect for errors
    for night_i, ob_folder in enumerate(reduced_data_folders):
        # Record the start time
        times.append(datetime.datetime.now())    
    
        # Navigate to the night folder and call pndrsCalibrate from terminal
        night = ob_folder.split("/")[-2].split("_")[0]
        print("\n", "-"*79, "\n", "\tCalibrating %s, night %i/%i\n" % (night, 
              night_i+1, len(reduced_data_folders)), "-"*79)
        os.system("(cd %s; pndrsCalibrate >> cal_log.txt)" % ob_folder)
        
        # Record and the end time and print duration
        times.append(datetime.datetime.now()) 
        cal_time = (times[-1] - times[-2]).total_seconds() 
        print("\n\nNight calibrated in %02d:%04.1f\n" 
              % (int(np.floor(cal_time/60.)), cal_time % 60.))
    
    # All nights finished, print summary          
    total_time = (times[-1] - times[0]).total_seconds()    
    print("Calibration finished, %i nights in %02d:%04.1f\n" 
          % (len(reduced_data_folders),int(np.floor(total_time/60.)), 
             total_time % 60.))
        

def move_sci_oifits(obs_path="/priv/mulga1/arains/pionier/complete_sequences/",
                    new_path="/home/arains/code/reach/results/"):
    """Used to collect the calibrated oiFits files of all science targets after
    running the PIONIER data reduction pipeline. 
    
    Parameters
    ----------
    obs_path: string
        Base directory, will move any SCI_oifits files one directory deeper.
    
    new_path: string
        Folder to move the results to.
    """
    sci_oi_fits = glob.glob(obs_path + "*/*SCI*oidataCalibrated.fits")
    
    print("\n", "-"*79, "\n", "\tCopying complete sequences\n", "-"*79)
    
    for files_copied, oifits in enumerate(sci_oi_fits):
        if os.path.exists(new_path):
            print("...copying %s" % oifits.split("/")[-1])
            copyfile(oifits, new_path + oifits.split("/")[-1])
            files_copied += 1
    
    print("%i files copied" % files_copied)
    

def initialise_interferograms():
    """
    """    
    # Randomly sample, move, rename
    pass
    
    
def run_one_calibration_set(sequences, complete_sequences, base_path, 
                            tgt_info, pred_ldd, e_pred_ldd,
                            run_local=False, already_calibrated=False):
    """
    (8) Write YYYY-MM-DD_oiDiam.fits files for each night of observing
    (9) Run pndrsCalibrate for each night of observing
    (10) Fit angular diameters to vis^2 of all science targets
    """
    # Intialise interferograms
    # Select the reduced interferograms which should be used for calibration
    initialise_interferograms()
    
    if not run_local and not already_calibrated:
        # Save oiDiam files
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info,
                                  pred_ldd, e_pred_ldd)
        
        # Run Calibration
        obs_folders = [base_path % night for night in nights.keys()]
        calibrate_all_observations(obs_folders)

        # Move oifits files back to central location (reach/results by default)
        move_sci_oifits()
    
    elif run_local and not already_calibrated:
        # Save oiDiam files for local inspection
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                                  pred_ldd, e_pred_ldd, 
                                  run_local=run_local)
    
    # Collate calibrated vis2 data
    if not run_local:
        vis2, e_vis2, baselines, wavelengths = rdiam.collate_vis2_from_file()
    else:
        path = "/Users/adamrains/code/reach/results/"
        vis2, e_vis2, baselines, wavelengths = rdiam.collate_vis2_from_file(path)
    
    # Fit LDD
    ldd_fits = rdiam.fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info)
    
    return vis2, e_vis2, baselines, wavelengths, ldd_fits
    
    
def run_n_bootstraps(sequences, complete_sequences, base_path, tgt_info,
                     n_guassian_ldd, e_pred_ldd, n_bootstraps,
                     run_local=False, already_calibrated=False):
    """
    """
    # Initialise data structures for results
    n_vis2 = {}
    n_baselines = {}
    n_ldd_fit = {}
    
    times = []
    
    # Bootstrap n times
    for b_i in np.arange(0, n_bootstraps):
        times.append(datetime.datetime.now())  
        print("\nBootstrapping iteration %i\n" % b_i)
        
        # Run a single calibration run
        vis2, e_vis2, baselines, wavelengths, ldd_fits = \
            run_one_calibration_set(sequences, complete_sequences, base_path, 
                                    tgt_info, n_guassian_ldd.iloc[b_i], 
                                    e_pred_ldd, run_local=run_local, 
                                    already_calibrated=already_calibrated)
                                    
        # Collate results
        for sci in vis2.keys():
            if sci in n_vis2.keys():
                n_vis2[sci].append(vis2[sci])
                n_baselines[sci].append(baselines[sci])
                n_ldd_fit[sci].append(ldd_fits[sci][0])
            else:
                n_vis2[sci] = [vis2[sci]]
                n_baselines[sci] = [baselines[sci]]
                n_ldd_fit[sci] = [ldd_fits[sci][0]]
                
        times.append(datetime.datetime.now())  
        b_i_time = (times[-1] - times[-2]).total_seconds() 
        print("\n\nBoostrap %i done in %02d:%04.1f\n" 
              % (b_i, int(np.floor(b_i_time/60.)), b_i_time % 60.))
                
    # All done
    print("\n", "-"*79, "\n", "\tBootstrapping Complete\n", "-"*79)
    
    for sci in n_ldd_fit.keys():
        # Predicted results
        #sci_ldd_pred = tgt_info
        #sci_e_ldd_pred = np.std(n_ldd_fit[sci])
        #sci_percent_pred = sci_e_ldd_fit / sci_ldd_fit * 100
        
        # Fitting results
        sci_ldd_fit = np.mean(n_ldd_fit[sci])
        sci_e_ldd_fit = np.std(n_ldd_fit[sci])
        sci_percent_fit = sci_e_ldd_fit / sci_ldd_fit * 100
        
        print("%-12s\tLDD = %f +/- %f (%0.2f%%)" % (sci, sci_ldd_fit, 
                                                      sci_e_ldd_fit,
                                                      sci_percent_fit))
                                                      
    return n_vis2, n_baselines, n_ldd_fit, wavelengths
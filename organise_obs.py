"""
Script to parse the ESO run summary and produce a set of complete bright/faint
concatenations
"""
import csv
import glob
import os
from shutil import copyfile
from collections import OrderedDict
from datetime import datetime

# -----------------------------------------------------------------------------
# Import and separate observation log into nights
# -----------------------------------------------------------------------------
# Find of the text logs
#all_logs = glob.glob("/priv/mulga2/mireland/pionier/*/PIONI*.NL.txt")
all_logs = glob.glob("/priv/mulga1/arains/pionier/p99/*/PIONI*.NL.txt")
#all_logs = glob.glob("/Users/adamrains/Downloads/PIONI*.txt")
all_logs.sort()

# Initialise dictionary to store observations
night_log = OrderedDict()

# Generate an observational log from these text files
for obs_log in all_logs:
    # Get the time of the observation (truncated to seconds)
    # Example filename: PIONI.2017-09-06T08:36:53.372.NL.txt
    yyyymmddhhMMss = obs_log.split("/")[-1][6:25]
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%H:%M:%S")
    #ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%I_%M_%S")
    
    # Read everything in the file
    with open(obs_log) as file:
        content = file.readlines()

    # Remove newline characters from the 
    content = [row.strip() for row in content]
    
    # Get the night of observation by way of subfolder
    night = obs_log.split("/")[-2]
    
    # Extract the relevant information
    for row in content:
        # Night grade
        if row[:6] == "Grade:":
            grade = row[-1]
        # Target name
        elif row[:7] == "Target:":
            target = row.split(" ")[-1]
        # OB - ESO observation ID
        elif row[:3] == "OB:":
            OB = row.split(" ")[-1]
        # Container - ESO concatenation ID
        elif row[:10] == "Container:":
            container = row.split(" ")[-1]
            
    # Strip out the unique, non-file format specific part of the filename
    ob_fn = obs_log.split("/")[-1].replace(".NL.txt", "")
    
    # Select details of observation to save
    ob_details = [container, OB, target, grade, ob_time, ob_fn]
    
    # Now store the observation details in the dictionary
    # If night entry exists, append observation
    if night in night_log.keys():
        night_log[night].append(ob_details)
    # Night entry  does not exist, create it
    else:
        night_log[night] = [ob_details]
    
    file.close()

# Can check that two datetime objects are close in time by subtracting them   
# datetime1 - datetime2 --> datetime.timedelta(...)
# datetime.timedelta(...).seconds


"""
file = "p99_log.txt"

observations = []

# Column definitions
# Retrieved OBs / ReducedOB / IDDate from > to / OB Name / Grade / Weather
with open(file) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        observations.append(line[1:])

# Order observations by night
night_log = OrderedDict()

for observation in observations:
    # If night entry exists, append observation
    if observation[2] in night_log.keys():
        night_log[observation[2]].append(observation)
    # Night entry  does not exist, create it
    else:
        night_log[observation[2]] = [observation]
"""
# -----------------------------------------------------------------------------
# Read in and separate out the bright and faint sci-cal sequences
# -----------------------------------------------------------------------------
# Read in each sequence
bright_list_file = "p99_bright.txt"
faint_list_file = "p99_faint.txt"

bright_list = []
faint_list = []

with open(bright_list_file) as csv_file:
    for line in csv.reader(csv_file):
        bright_list.append(line[0].replace(" ", "_"))
        
with open(faint_list_file) as csv_file:
    for line in csv.reader(csv_file):
        faint_list.append(line[0].replace(" ", "_"))
        
# Order each sequence
bright_sequences = {}
faint_sequences = {}

for i in xrange(0, len(bright_list), 4):
    bright_sequences[bright_list[i]] = [bright_list[i+1], bright_list[i],
                                        bright_list[i+2], bright_list[i],
                                        bright_list[i+3]]
    
    faint_sequences[faint_list[i]] = [faint_list[i+1], faint_list[i],
                                      faint_list[i+2], faint_list[i],
                                      faint_list[i+3]]

all_sequences = [bright_sequences, faint_sequences]
seq_label = ["bright", "faint"]

# -----------------------------------------------------------------------------
# Check night by night to see which sequences were completed
# -----------------------------------------------------------------------------
# Observation grades as follow:
# - A: fully within constraints
# - B: mostly within constraints (~10% violation)
# - C: out of constraints, will be repeated
# - D: out of constraints, will not be repeated
# - X: aborted
# - ? or "-": error, will be repeated
# Given this, A and B observations will be accepted
def check_and_save_good_sequence(complete_sequences, sequence, concatentation):
    """
    """
    # Only add if all observations in sequence are good
    grade = (night_log[night][ob_i][3] + night_log[night][ob_i+1][3]
             + night_log[night][ob_i+2][3] + night_log[night][ob_i+3][3]
             + night_log[night][ob_i+4][3])
         
    # Determine start and end times of the concatenation
    start = night_log[night][ob_i][4]
    end = night_log[night][ob+4][4]
    
    # If all obs have acceptable grades the sequence exists
    if ("C" not in grade and "D" not in grade and "-" not in grade 
        and "?" not in grade and "X" not in grade):
        complete_sequences.append([concatentation, seq_label[seq_i], night, start, end, 
                                   grade])  
    


complete_sequences = {}

# For every night of observations...
for night in night_log.keys():
    # For every sequence (i.e. bright, faint)...
    for seq_i, sequence in enumerate(all_sequences):
        # For every science target...
        for sci in sequence:
            # Step through the nightly observations attempting to match
            # the CAL-SCI-CAL-SCI-CAL sequences where each CAL or SCI has many
            # different files (N exposures, darks, kappa matrices)
            # Do this by building up the list of observations until you either
            # reach the end of the final calibrator (i.e. complete sequence) or 
            # something else happens (i.e. broken sequence)
            ob_i = 0            # The ith observation that night
            tgt_i = 0           # The ith target in the CAL-SCI sequence
            concatenation = []  # Current list of obs from CAL-SCI sequence
            
            # For every observation in the night...
            while ob_i < len(night_log[night]):
                # Adding entry for current target, increment ob_i
                if sequence[sci][tgt_i] in night[ob_i][2]:
                    concatenation.append(night[ob_i])
                    ob_i += 1
                
                # Adding entry for next target, increment ob_i and tgt_i
                elif ((tgt_i + 1 < len(sequence[sci])) 
                    and sequence[sci][tgt_i+1] in night[ob_i][2]):
                    concatenation.append(night[ob_i])
                    ob_i += 1
                    tgt_i += 1
                    
                # Completed sequence, increment ob_i and reset tgt_i
                elif ((tgt_i + 1 == len(sequence[sci]))         # Last tgt &&
                    and (ob_i + 1 == len(night)                 # Last ob OR
                    or (sequence[sci][tgt_i] not in night[ob_i][2]))):    # Non-seq ob
                    
                    all_grades = [concatenation[i][3] for i in concatenation]
                    
                    # If all obs have acceptable grades the sequence exists
                    if ("C" not in grade and "D" not in grade 
                        and "-" not in grade and "?" not in grade 
                        and "X" not in grade):
                        complete_sequences[(sci, seq_label[seq_i], grade)] = concatentation
                        #complete_sequences.append([concatentation, seq_label[seq_i], night, start, end, 
                                   #grade])  
                
                # Broken sequence, discard concatenation, reset tgt_i
                else:
                    tgt_t = 0
                    concatenation = []
            
                """
                # If observation is the current (tgt_i) target, add to 
                # concatenation and increment observation counter
                if sci[tgt_i] in night[ob_i][2]:
                    concatenation.append(night[ob_i])
                    ob_i += 1
                    
                # If observation is *next* target in concatenation, add to seq
                # and increment both the target and observation counters
                elif tgt_i + 1 < len(sci) and sci[tgt_i+1] in night[ob_i][2]:
                    concatenation.append(night[ob_i])
                    ob_i += 1
                    tgt_i += 1
                    
                # If observation is not in sequence, but last observation was
                # of the last calibrator in the sequence, consider the seq
                # complete and save
                elif tgt_i + 1 == len(sci) and sci[tgt_i] not in night[ob_i][2]:
                    pass
                
                # If not current target, and not at end of sequence, consider
                # incomplete and discard current progress and reset (but don't
                # reset ob counter, as we want to test this ob again from seq 
                # start)
                else:
                    tgt_i = 0
                    concatenation = []
                
                
                
                try:

                    # Determine if the correct sequence exists in order
                    if (sequence[sci][0] in night_log[night][ob_i][2] 
                        and sequence[sci][1] in night_log[night][ob_i+1][2]
                        and sequence[sci][2] in night_log[night][ob_i+2][2]
                        and sequence[sci][3] in night_log[night][ob_i+3][2]
                        and sequence[sci][4] in night_log[night][ob_i+4][2]):
                    	print "good seuqence"
                        
                except:
                    # We've reached the end of the night, can't fit a full 
                    # concatenation in
                    break
                """
# Done
#complete_sequences.sort()

# -----------------------------------------------------------------------------
# Copy the completed sequences to a new directory structure
# -----------------------------------------------------------------------------
"""
Reduced data takes the form: PIONI.2017-09-06T08:44:52.816_XXX_*.*

Where each single observation has a set of corresponding pdf diagnostic plots,
as well as the reduced (uncalibrated) data
"""
# Test that the directory exists before creating it
#if not os.path.exists(""):
#    os.makedirs("")
    
# Copy the file
#copyfile("", "")

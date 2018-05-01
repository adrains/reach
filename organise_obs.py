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
all_logs = glob.glob("/priv/mulga1/arains/pionier/P99/*/PIONI*.NL.txt")
#all_logs = glob.glob("/Users/adamrains/Downloads/PIONI*.txt")
all_logs.sort()

# Initialise dictionary to store observations
night_log = OrderedDict()

# Generate an observational log from these text files
for obs_log in all_logs:
    # Get the time of the observation (truncated to seconds)
    # Example filename: PIONI.2017-09-06T08:36:53.372.NL.txt
    yyyymmddhhMMss = obs_log.split("/")[-1][6:25]
    #ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%I:%M:%S")
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%I_%M_%S")
    
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
    ob_details = [container, OB, target, grade, yyyymmddhhMMss, ob_fn]
    
    # Now store the observation details in the dictionary
    # If night entry exists, append observation
    if night in night_log.keys():
        night_log[night].append(observation)
    # Night entry  does not exist, create it
    else:
        night_log[night] = [observation]
    
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

complete_sequences = []

# For every night of observations...
for night in night_log.keys():
    # For every sequence (i.e. bright, faint)...
    for seq_i, sequence in enumerate(all_sequences):
        # For every science target...
        for sci in sequence:
            # For every observation in the night...
            for ob_i, observation in enumerate(night_log[night]):
                try:
                    # Determine if the correct sequence exists in order
                    if (sequence[sci][0] in night_log[night][ob_i][4] 
                        and sequence[sci][1] in night_log[night][ob_i+1][4]
                        and sequence[sci][2] in night_log[night][ob_i+2][4]
                        and sequence[sci][3] in night_log[night][ob_i+3][4]
                        and sequence[sci][4] in night_log[night][ob_i+4][4]):
                    
                        # Only add if all observations in sequence are good
                        grade = (night_log[night][ob_i][5]
                                 + night_log[night][ob_i+1][5]
                                 + night_log[night][ob_i+2][5]
                                 + night_log[night][ob_i+3][5]
                                 + night_log[night][ob_i+4][5])
                             
                        # Determine start and end times of the concatenation
                        start = night_log[night][ob_i][3].split(" ")[0]
                        end = night_log[night][ob_i+4][3].split(" ")[-1]
                        
                        # If all obs have acceptable grades the sequence exists
                        if ("C" not in grade and "D" not in grade 
                            and "-" not in grade and "?" not in grade
                            and "X" not in grade):
                            complete_sequences.append([sci, seq_label[seq_i], 
                                                       night, start, end, 
                                                       grade])  
                except:
                    # We've reached the end of the night, can't fit a full 
                    # concatenation in
                    break

# Done
complete_sequences.sort()

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
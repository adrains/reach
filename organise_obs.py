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
complete_sequences = {}

# For every night of observations...
for night in night_log.keys():
    print "\n-----------------------------"
    print night, "\n-----------------------------"
    # For every sequence (i.e. bright, faint)...
    for seq_i, sequence in enumerate(all_sequences):
        print "\n"
        print "-------", seq_label[seq_i], "-------"
        # For every science target...
        for sci in sequence:
            print ""
            print sci,
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
                if sequence[sci][tgt_i] in night_log[night][ob_i][2]:
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    print "1",
                # Adding entry for next target, increment ob_i and tgt_i
                elif ((tgt_i + 1 < len(sequence[sci])) 
                    and sequence[sci][tgt_i+1] in night_log[night][ob_i][2]):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    tgt_i += 1
                    print 2,
                # Broken sequence, discard concatenation, reset tgt_i
                else:
                    ob_i += 1
                    tgt_i = 0
                    concatenation = []                
                
                # Now check to see if we have completed the sequence. Two ways
                # that this can be the case:
                #  1 - On last target of sequence, next is unrelated
                #  2 - On last target of sequence, reached end of night
                # Completed sequence, increment ob_i and reset tgt_i
                if ((tgt_i + 1 == len(sequence[sci]))    # Last tgt &&
                    and (ob_i == len(night_log[night])   # Last ob OR
                    or (sequence[sci][tgt_i] not in night_log[night][ob_i][2]))):    # Non-seq ob
                    print 3,
                    all_grades = [observation[3] for observation in concatenation]
                    grade = "".join(all_grades)
                    
                    # If all obs have acceptable grades the sequence exists
                    if ("C" not in grade and "D" not in grade 
                        and "-" not in grade and "?" not in grade 
                        and "X" not in grade):
                        complete_sequences[(sci, seq_label[seq_i], grade)] = concatenation
                        print "[DONE, %s]" % grade 
                    else:
                        print "[BAD GRADE %s]" % grade  
                        tgt_i = 0


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

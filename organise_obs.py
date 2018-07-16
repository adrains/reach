"""
Script to parse the ESO run summary and produce a set of complete bright/faint
concatenations
"""
from __future__ import division, print_function
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
all_logs = glob.glob("/priv/mulga1/arains/pionier"
                     "/all_sequences/*/PIONI*.NL.txt")
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
            target = row.split(" ")[-1].replace("_", "")
        # OB - ESO observation ID
        elif row[:3] == "OB:":
            OB = row.split(" ")[-1]
        # Container - ESO concatenation ID
        elif row[:10] == "Container:":
            container = row.split(" ")[-1]
        # Run - the ESO period
        elif row[:4] == "Run:":
            run = row.split(" ")[-1]
            
    # Strip out the unique, non-file format specific part of the filename
    ob_fn = obs_log.split("/")[-1].replace(".NL.txt", "")
    
    # Select details of observation to save
    ob_details = [container, OB, target, grade, ob_time, ob_fn, run]
    
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


# -----------------------------------------------------------------------------
# Read in and separate out the bright and faint sci-cal sequences
# -----------------------------------------------------------------------------
# Read in each sequence
bright_list_files = ["p99_bright.txt", "p101_bright.txt"]
faint_list_files = ["p99_faint.txt", "p101_faint.txt"]

bright_list = []
faint_list = []

for bright_list_file in bright_list_files:
    with open(bright_list_file) as csv_file:
        for line in csv.reader(csv_file):
            bright_list.append(line[0].replace(" ", "_"))

for faint_list_file in faint_list_files:        
    with open(faint_list_file) as csv_file:
        for line in csv.reader(csv_file):
            faint_list.append(line[0].replace(" ", "_"))

# For consistency, remove any underscores
bright_list = [tgt.replace("_", "") for tgt in bright_list]
faint_list = [tgt.replace("_", "") for tgt in faint_list]
        
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

missing_sequences = [(key, "bright") for key in bright_sequences]
missing_sequences.extend([(key, "faint") for key in faint_sequences])
missing_sequences = set(missing_sequences)

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
def is_good_grade(new_grade):
    """Tests the grade
    """
    is_good_grade = True
    
    bad_grades = ["C", "D", "-", "?", "X"]
    
    for grade in bad_grades:
        if grade in new_grade:
            is_good_grade = False
            
    return is_good_grade


complete_sequences = {}

# For every night of observations...
for night in night_log.keys():
    print("\n\n---------------------------------------------------------")
    print(night, "\n---------------------------------------------------------")
    # For every sequence (i.e. bright, faint)...
    for seq_i, sequence in enumerate(all_sequences):
        print("-------", seq_label[seq_i], "-------")
        # For every science target...
        for sci in sequence:
            print("")
            print(sci, end="   ")
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
                # Get the grade of the observation to be considered
                grade = night_log[night][ob_i][3]
                obs_tar = night_log[night][ob_i][2] 
                
                sequence_added_to = True
                
                # First element of sequence
                # - Add to concatenation and increment to next observation
                if (len(concatenation) == 0 and sequence[sci][tgt_i] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    print("0", end="")
                
                # Continuation of current target
                # - Add to concatenation and increment to next observation
                elif (len(concatenation) > 0 
                    and sequence[sci][tgt_i] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    print("1", end="")
                    
                # No more obs for current target, check next in sequence
                # - If not at last ob, add to concatenation and increment ob
                elif (len(concatenation) > 0 and tgt_i + 1 < len(sequence[sci])
                    and sequence[sci][tgt_i+1] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    tgt_i += 1
                    print(2, end="")
                    
                else:
                    sequence_added_to = False
                
                # Now we should either have finished a sequence, or found a
                # broken sequence. Two ways to complete a sequence, otherwise
                # we consider it broken and move on:
                #  1 - On last target of sequence, next is unrelated
                #  2 - On last target of sequence, reached end of night
                # If the last target in the sequence, and we either did not add
                # to anything or we are at the end of the night
                if (tgt_i + 1 == len(sequence[sci]) 
                    and len(concatenation) >= 29
                    and (not sequence_added_to 
                    or ob_i < len(night_log[night]))):
                    # Note that there is the case where the grade is bad on the
                    # final target
                    print(3, end="")
                    
                    all_grades = [ob[3] for ob in concatenation]
                    grade = "".join(all_grades)
                    
                    # Record the period of the sequence
                    period = concatenation[0][6]
                    
                    end_time = concatenation[-1][4].isoformat()
                    key = (period, sci, seq_label[seq_i], end_time, grade)
                    complete_sequences[key] = concatenation
                    print(" [DONE, %s]" % grade)
                    
                    if (sci, seq_label[seq_i]) in missing_sequences:
                        missing_sequences.remove((sci, seq_label[seq_i]))
                    
                    tgt_i = 0
                    
                    # Only add to the counter if we added to the sequence
                    if not sequence_added_to:
                        ob_i += 1
                    
                    concatenation = [] 
                    
                # We didn't add to the sequence, and we're not on the last 
                # target of the sequence - consider the concatenation broken
                # and move on
                elif not sequence_added_to:
                    
                    if len(concatenation) > 100:
                        print(" [Failed on: %s vs %s, %s, %i] on SEQ %s" %  
                                (night_log[night][ob_i][2], 
                                sequence[sci][tgt_i], grade, ob_i,
                                sequence[sci]), end="") 
                    
                    # Here is where we decide whether to move to the next 
                    # observation. We should *not* move on if:
                    # - The observation is an earlier member of the 
                    #   concatenation currently in progress *and* has a good
                    #   grade
                    # - In all other cases we want to increment the ob_i
                    #   counter: it is only when you have adjacent broken
                    #   sequences that things get messy
                    if not is_good_grade(grade):
                        ob_i += 1
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("A")
                        
                    elif (len(concatenation) > 0 
                        and night_log[night][ob_i][2] in sequence[sci]):
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("B")
                        
                    else:
                        ob_i += 1
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("C")
                    
                    concatenation = []             
                    
        print("\n") 

# -----------------------------------------------------------------------------
# Manually account for any out of sequence, but complete, sequences
# -----------------------------------------------------------------------------
# Note: this code should be in a imported file

# First star: del Pav. This was the only sequence observed on 27/08/17, and 
# seems to have a mismatch between what the logs *say* are good observations,
# and what are marked as good per the ESO grades (i.e. A or B). The HR7732 data
# is fine per the logs, but marked as C. Ignore the first saturated del Pav
# sequence, but take everything else, mindful that the guiding was lost during
# the final del Pav exposure and we have 4x the number of observations required
# (but don't currently know how to assess them as good or bad).
concatenation = night_log["2017-08-27"][:6]
concatenation.extend(night_log["2017-08-27"][16:])

# Determine grades, period, and end time, then save to dict as usual
grade = "".join([observation[3] for observation in concatenation])
period = concatenation[0][6]
end_time = concatenation[-1][4].isoformat()

key = (period, "delPav", "faint", end_time, grade)
complete_sequences[key] = concatenation

missing_sequences.remove(("delPav", "faint"))

# Second star: lam Sgr. First calibrator observed within requirements, but 1st
# science was not. Everything else per the log seems to be okay, with HR6838
# observed again to bracket the science observations.
concatenation = night_log["2017-08-26"][:6]
concatenation.extend(night_log["2017-08-27"][16:51])

# Determine grades, period, and end time, then save to dict as usual
grade = "".join([observation[3] for observation in concatenation])
period = concatenation[0][6]
end_time = concatenation[-1][4].isoformat()

key = (period, "lamSgr", "faint", end_time, grade)
complete_sequences[key] = concatenation

missing_sequences.remove(("lamSgr", "faint"))

# Third star: Tau Ceti (bright) in period 99, which is missed because of the
# assumption that each star has a single unique bright and faint sequence, 
# which breaks down for Tau Cet given we removed a bad calibrator from it. It's
# easier to account for this here than changing the data structures used above.
concatenation = night_log["2017-08-26"][67:101]

# Determine grades, period, and end time, then save to dict as usual
grade = "".join([observation[3] for observation in concatenation])
period = concatenation[0][6]
end_time = concatenation[-1][4].isoformat()

key = (period, "TauCet", "bright", end_time, grade)
complete_sequences[key] = concatenation

missing_sequences.remove(("TauCet", "bright"))

# -----------------------------------------------------------------------------
# Summarise keys for easy inspection
# -----------------------------------------------------------------------------
obs_keys = complete_sequences.keys()
obs_keys.sort()

print("\n\n-------------------------\nSummary\n-------------------------")
print("%i/%i Unique Complete Sequences\n" % (len(obs_keys), 
                                           (len(bright_sequences) + 
                                            len(faint_sequences))))
print("%-16s%-12s%-12s%-22s%-10s\n" % ("Period", "Target", "Sequence", "Time", 
                                       "Grade")) 

for ob in obs_keys:
    print("%-16s%-12s%-12s%-22s%-10s" % (ob[0], ob[1], ob[2], ob[3], ob[4])) 
   

print("\n\n----------------------\nMissing Sequences\n----------------------")  
print("%i Missing Sequences\n" % len(missing_sequences))
    
for sequence in missing_sequences:
    print("%-12s%-12s" % (sequence[0], sequence[1]))

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

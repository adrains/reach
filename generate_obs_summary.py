import csv
from collections import OrderedDict

# -----------------------------------------------------------------------------
# Import and separate observation log into nights
# -----------------------------------------------------------------------------
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

for night in night_log.keys():
    # Bright
    for sci in bright_sequences:
        for ob_i, observation in enumerate(night_log[night]):
            try:
                if (bright_sequences[sci][0] in night_log[night][ob_i][4] 
                    and bright_sequences[sci][1] in night_log[night][ob_i+1][4]
                    and bright_sequences[sci][2] in night_log[night][ob_i+2][4]
                    and bright_sequences[sci][3] in night_log[night][ob_i+3][4]
                    and bright_sequences[sci][4] in night_log[night][ob_i+4][4]):
                    
                    # Only add if all observations in sequence are good
                    grade = (night_log[night][ob_i][5]
                             + night_log[night][ob_i+1][5]
                             + night_log[night][ob_i+2][5]
                             + night_log[night][ob_i+3][5]
                             + night_log[night][ob_i+4][5])
                             
                    if ("C" not in grade and "D" not in grade 
                        and "-" not in grade and "?" not in grade
                        and "X" not in grade):
                        complete_sequences.append([sci, "bright", night, grade]) 
            except:
                # We've reached the end of the night, can't fit a full 
                # concatenation in
                break
    
    # Faint
    for sci in faint_sequences:
        for ob_i, observation in enumerate(night_log[night]):
            try:
                if (faint_sequences[sci][0] in night_log[night][ob_i][4] 
                    and faint_sequences[sci][1] in night_log[night][ob_i+1][4]
                    and faint_sequences[sci][2] in night_log[night][ob_i+2][4]
                    and faint_sequences[sci][3] in night_log[night][ob_i+3][4]
                    and faint_sequences[sci][4] in night_log[night][ob_i+4][4]):
                
                    # Only add if all observations in sequence are good
                    grade = (night_log[night][ob_i][5]
                             + night_log[night][ob_i+1][5]
                             + night_log[night][ob_i+2][5]
                             + night_log[night][ob_i+3][5]
                             + night_log[night][ob_i+4][5])
                         
                    if ("C" not in grade and "D" not in grade 
                        and "-" not in grade and "?" not in grade
                        and "X" not in grade):
                        complete_sequences.append([sci, "faint", night, grade]) 
            except:
                # We've reached the end of the night, can't fit a full 
                # concatenation in
                break

# Done
complete_sequences.sort()
print complete_sequences
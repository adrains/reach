"""Helper functions to assist with writing the PIONIER paper (e.g. making 
LaTeX tables).

Note that it might be helpful to plot dates the tables were generated to ease
copying and pasting?
"""
from __future__ import division, print_function
import numpy as np
import reach.utils as rutils
from collections import OrderedDict


def make_table_results(results):
    """
    """
    columns = OrderedDict([("Star", ""),
                           ("HD", ""),
                           ("Period", ""),
                           ("Sequence", ""),
                           (r"$\theta_{\rm LD}$", "(mas)"),
                           ("C", ""),
                           (r"R ($R_\odot$)", ""), 
                           (r"T$_{\rm eff}$", "(K)")])
                           
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star, row in results.iterrows():
        table_row = ""
        
        id = row["STAR"]
        period = row["PERIOD"]
        sequence = row["SEQUENCE"]
        
        # Step through column by column
        table_row += "%s & " % id
        table_row += "%s & " % row["HD"].replace("HD", "")
        table_row += "%s & " % period
        table_row += "%s & " % sequence
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["LDD_FIT"], row["e_LDD_FIT"])
        table_row += "%0.3f & " % row["C_SCALE"]
        table_row += r"%0.2f $\pm$ %0.2f &" % (row["R_STAR"], row["e_R_STAR"])
        table_row += r"%0.0f $\pm$ %0.0f " % (row["Teff_VTmag"], row["e_Teff_VTmag"])
        
        table_rows.append(table_row + r"\\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_results.tex", table_1, fmt="%s")
    
    

def make_table_observation_log(tgt_info, complete_sequences, sequences):
    """
    """
    columns = OrderedDict([("HD", ""),
                           ("UT Date", ""),
                           ("ESO", "Period"),
                           ("Sequence", "Type"),
                           ("Baseline", ""), 
                           #("Spectral", "channels"),
                           ("Calibrator", "HD")])
                           
    header = []
    table_rows = {}
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    
    # Populate the table for every science target
    for seq in complete_sequences:
        table_row = ""
        
        star_id = seq[1]
        ut_date = complete_sequences[seq][0]
        period = seq[0]
        seq_type = seq[2]
        baselines = complete_sequences[seq][2][0][9]
        cals = [target.replace("_", "").replace(".","").replace(" ", "") 
                for target in sequences[seq][::2]]
        cals = ("%s, %s, %s" % tuple(rutils.get_unique_key(tgt_info, cals))).replace("HD", "")
        
        
        table_row = ("%s & "*len(columns)) % (star_id, ut_date, period,  
                                              seq_type, baselines, cals)
                
        table_rows[(ut_date, star_id)] = table_row + r"\\"
        
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Sort by UT
    ut_sorted = table_rows.keys()
    ut_sorted.sort()
    
    sorted_rows = [table_rows[row] for row in ut_sorted]
    
    # Write the tables
    table_1 = header + sorted_rows[:30] + footer
    table_2 = header + sorted_rows[30:] + footer
    
    np.savetxt("paper/table_observations_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_observations_2.tex", table_2, fmt="%s")
    

def make_table_targets(tgt_info):
    """
    Columns:
    - Common ID
    - HD ID
    - SpT
    - Vmag
    - Hmag
    - Teff
    - Logg
    - [Fe/H]
    - Mass?
    - Parallax
    - Existing angular diameter
    """
    # Column names and associated units
    columns = OrderedDict([("Star", ""), 
                           ("HD", ""),
                           ("SpT", ""),
                           ("VTmag", "(mag)"), 
                           ("Hmag", "(mag)"),
                           ("T$_{\\rm eff}$", "(K)"),
                           ("logg", "(dex)"), 
                           ("[Fe/H]", "(dex)"),
                           ("Ref", ""),
                           ("Plx", "(mas)"),
                           ("Mission", "")])
    
    labels = ["Primary", "index", "SpT", "VTmag", "Hmag", "Teff", "logg", 
              "FeH_rel", "Plx"]
               
    
    table_rows = []
    
    # Construct the header of the table
    table_rows.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    table_rows.append("\hline")
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    table_rows.append("\hline")
    
    # Populate the table for every science target
    for star_i, star in tgt_info[tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Step through column by column
        table_row += "%s & " % star["Primary"]
        table_row += "%s & " % star.name.replace("HD", "")
        table_row += "%s & " % star["SpT"]
        table_row += "%0.2f & " % star["VTmag"]
        table_row += "%0.2f & " % star["Hmag"]
        table_row += r"%0.0f $\pm$ %0.0f & " % (star["Teff"], star["e_teff"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["logg"], star["e_logg"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["FeH_rel"], star["e_FeH_rel"])
        table_row += "TODO &"
        
        
        # Parallax is not from Gaia DR2
        if np.isnan(star["Plx"]):
            table_row += r"%0.2f $\pm$ %0.2f &" % (star["Plx_alt"], star["e_Plx_alt"])
            table_row += "\\textit{Hipparcos}"
        
        # From Gaia DR2
        else:
            table_row += r"%0.2f $\pm$ %0.2f &" % (star["Plx"], star["e_Plx"])
            table_row += "\\textit{Gaia}"
        
        table_rows.append(table_row + r"\\")
        
    # Finish the table
    table_rows.append("\hline")
    table_rows.append("\end{tabular}")
    
    # Write the table
    np.savetxt("paper/table_targets.tex", table_rows, fmt="%s")
    
    


def make_table_calibrators(tgt_info, sequences):
    """
    Columns
    - Angular diameter relation used
    - Status (e.g. bad (binary))
    """
    # Column names and associated units
    columns = OrderedDict([("HD", ""),
                           ("SpT", ""),
                           ("VTmag", "(mag)"), 
                           ("Hmag", "(mag)"),
                           ("E(B-V)", "(mag)"),
                           ("$\\theta_{\\rm pred}$", "(mas)"),
                           ("$\\theta_{\\rm LD}$ Rel", ""),
                           ("Used", ""),
                           ("Plx", "(mas)"),
                           ("Mission", ""),
                           ("Target/s", "")])
    
    labels = ["index", "SpT", "VTmag", "Hmag", "Quality", "Target/s"]
               
    
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    
    # Populate the table for every science target
    for star_i, star in tgt_info[~tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Find which science target/s the calibrator is associated with
        scis = []
        for seq in sequences:
            cals = [target.replace("_", "").replace(".","").replace(" ", "") 
                    for target in sequences[seq]]
                        
            if star.name in rutils.get_unique_key(tgt_info, cals):
                scis.append(seq[1])
                
        scis = list(set(scis))
        scis.sort()
        
        # Step through column by column
        table_row += "%s & " % star.name.replace("HD", "")
        
        # Make SpT have a smaller font if it's long
        if len(str(star["SpT"])) > 5:
            table_row += "{\\tiny %s } & " % star["SpT"]
        else:
            table_row += "%s & " % star["SpT"]
            
        table_row += "%0.2f & " % star["VTmag"]
        table_row += "%0.2f & " % star["Hmag"]
        table_row += "%0.3f & " % star["eb_v"]
        table_row += "%0.3f & " % star["LDD_pred"]
        table_row += ("%s & " % star["LDD_rel"]).split("_")[-1]
        
        # Determine whether the star was used as a calibrator
        if star["Quality"] == "BAD":
            table_row += "N & "
        else:
            table_row += "Y & "
        
        # Parallax is not from Gaia DR2
        if np.isnan(star["Plx"]):
            table_row += r"%0.2f $\pm$ %0.2f &" % (star["Plx_alt"], star["e_Plx_alt"])
            table_row += "\\textit{Hipparcos} &"
        
        # From Gaia DR2
        else:
            table_row += r"%0.2f $\pm$ %0.2f &" % (star["Plx"], star["e_Plx"])
            table_row += "\\textit{Gaia} &"        
        
        table_row += ("%s, "*len(scis) % tuple(scis))[:-2]
        
        table_rows.append(table_row + r"\\")
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows[:62] + footer
    table_2 = header + table_rows[62:] + footer
    
    np.savetxt("paper/table_calibrators_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_calibrators_2.tex", table_2, fmt="%s")

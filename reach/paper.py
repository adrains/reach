"""Helper functions to assist with writing the PIONIER paper (e.g. making 
LaTeX tables).

Note that it might be helpful to plot dates the tables were generated to ease
copying and pasting?
"""
from __future__ import division, print_function
import numpy as np
import reach.utils as rutils
from collections import OrderedDict

def make_table_final_results(tgt_info):
    """
    """
    columns = OrderedDict([("Star", ""),
                           #("HD", ""),
                           ("u", ""),
                           (r"$\theta_{\rm LD}$", "(mas)"),
                           (r"R", "($R_\odot$)"), 
                           (r"f$_{\rm bol}$", r"(ergs s$^{-1}$ cm $^{-2}$)"),
                           (r"T$_{\rm eff}$", "(K)"),
                           ("L", ("($L_\odot$)"))])
                           
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
    for star_i, row in tgt_info[tgt_info["Science"]].iterrows():
        
        # Only continue if we have data on this particular star
        if not row["in_paper"]:
            continue
        
        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % row["Primary"]
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["u_lld"], row["e_u_lld"])
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["ldd_final"], row["e_ldd_final"])
        table_row += r"%0.3f $\pm$ %0.3f &" % (row["r_star_final"], row["e_r_star_final"])
        table_row += r"%0.3E $\pm$ %0.3E &" % (row["f_bol_final"], row["e_f_bol_final"])
        table_row += r"%0.0f $\pm$ %0.0f & " % (row["teff_final"], row["e_teff_final"])
        table_row += r"%0.3f $\pm$ %0.3f " % (row["L_star_final"], row["e_L_star_final"])
        
        table_rows.append(table_row + r"\\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_final_results.tex", table_1, fmt="%s")


def make_table_seq_results(results):
    """
    """
    columns = OrderedDict([("Star", ""),
                           #("HD", ""),
                           ("Period", ""),
                           ("Sequence", ""),
                           (r"$\theta_{\rm LD}$", "(mas)"),
                           ("C", "")])
                           
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
        #table_row += "%s & " % row["HD"].replace("HD", "")
        table_row += "%s & " % period
        table_row += "%s & " % sequence
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["LDD_FIT"], row["e_LDD_FIT"])
        table_row += "%0.3f " % row["C_SCALE"]
        
        table_rows.append(table_row + r"\\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_sequence_results.tex", table_1, fmt="%s")
    
    
def make_table_fbol(tgt_info):
    """
    """
    columns = OrderedDict([("Star", ""),
                           ("HD", ""),
                           (r"f$_{\rm bol}$ (MARCS)", r"(ergs s$^{-1}$ cm $^{-2}$)"),
                           (r"$\sigma_{f_{\rm bol}} (\zeta)$", r"(\%)")])
                           #(r"f$_{\rm bol} (avg)$", r"(ergs s$^{-1}$ cm $^{-2}$)")])
                     
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")    
    
    
    bands = [r"H$_p$", r"B$_T$", r"V$_T$", r"B$_P$", r"R$_P$"]
    
    f_bols = ["f_bol_Hpmag", "f_bol_BTmag", "f_bol_VTmag", "f_bol_BPmag", 
             "f_bol_RPmag"]
    e_f_bols = ["e_f_bol_Hpmag", "e_f_bol_BTmag", "e_f_bol_VTmag", 
               "e_f_bol_BPmag", "e_f_bol_RPmag"]
    
     # Populate the table for every science target
    for star_i, star in tgt_info[tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Step through column by column
        table_row += "%s & " % star["Primary"]
        table_row += "%s & " % star.name.replace("HD", "")
        
        if not np.isnan(star["f_bol_Hpmag"]):
            table_row += r"H$_p$: %.3E & " % star["f_bol_Hpmag"]
        
            e_pc_f_bol_hp = star["e_f_bol_Hpmag"] / star["f_bol_Hpmag"]
        
            table_row += r"%.2f \\" % (e_pc_f_bol_hp * 100)
        else:
            table_row += r"H$_p$: %s & %s \\" % ("-"*13, "-"*5)
        
        #table_row += r"%.3E $\pm$ %0.3E \\" % (star["f_bol_final"], 
        #                                       star["e_f_bol_final"])
        
        table_rows.append(table_row)
        
        # Now have a separate row for each of the remaining filters
        for band_i in np.arange(1, len(bands)):
            table_row = r" & & %s: %.3E &" % (bands[band_i], star[f_bols[band_i]])
            
            e_pc_f_bol = star[e_f_bols[band_i]] / star[f_bols[band_i]]
        
            table_row += r"%.2f \\" % (e_pc_f_bol*100)
            
            table_rows.append(table_row)
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows[:60] + footer
    table_2 = header + table_rows[60:] + footer
    
    # Write the table
    np.savetxt("paper/table_fbol_1.tex", table_1, fmt="%s")      
    np.savetxt("paper/table_fbol_2.tex", table_2, fmt="%s")     
    
    

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
                
        table_rows[(ut_date, star_id)] = table_row[:-2] + r"\\"
        
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Sort by UT
    ut_sorted = table_rows.keys()
    ut_sorted.sort()
    
    sorted_rows = [table_rows[row] for row in ut_sorted]
    
    # Write the table
    table = header + sorted_rows + footer
    np.savetxt("paper/table_observations.tex", table, fmt="%s")
    

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
                           ("RA", "(hh mm ss.ss)"),
                           ("DEC", "(dd mm ss.ss)"),
                           ("SpT", ""),
                           ("VTmag", "(mag)"), 
                           ("Hmag", "(mag)"),
                           ("T$_{\\rm eff}$", "(K)"),
                           ("logg", "(dex)"), 
                           ("[Fe/H]", "(dex)"),
                           ("vsini", r"(km$\,$s$^{-1}$)"),
                           ("Ref", ""),
                           ("Plx", "(mas)")])#,
                           #("Mission", "")])         
    
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
        
        # Format RA and DEC
        ra_hr = np.floor(star["RA"] / 15)
        ra_min = np.floor((star["RA"] / 15 - ra_hr) * 60)
        ra_sec = ((star["RA"] / 15 - ra_hr) * 60 - ra_min) * 60
        ra = "%02i %02i %05.2f" % (ra_hr, ra_min, ra_sec)
        
        dec_deg = np.floor(star["DEC"])
        dec_min = np.floor((star["DEC"] - dec_deg) * 60)
        dec_sec = ((star["DEC"] - dec_deg) * 60 - dec_min) * 60
        dec = "%02i %02i %05.2f" % (dec_deg, dec_min, dec_sec)
        
        # Step through column by column
        table_row += "%s & " % star["Primary"]
        table_row += "%s & " % star.name.replace("HD", "")
        table_row += "%s & " % ra
        table_row += "%s & " % dec
        table_row += "%s & " % star["SpT"]
        table_row += "%0.2f & " % star["VTmag"]
        table_row += "%0.2f & " % star["Hmag"]
        table_row += r"%0.0f $\pm$ %0.0f & " % (star["Teff"], star["e_teff"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["logg"], star["e_logg"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["FeH_rel"], star["e_FeH_rel"])
        table_row += r"%0.2f $ &" % star["vsini"]
        table_row += "TODO &"
        
        
        # Parallax is not from Gaia DR2
        if np.isnan(star["Plx"]):
            table_row += r"%0.2f $\pm$ %0.2f" % (star["Plx_alt"], star["e_Plx_alt"])
            #table_row += "\\textit{Hipparcos}"
        
        # From Gaia DR2
        else:
            table_row += r"%0.2f $\pm$ %0.2f" % (star["Plx"], star["e_Plx"])
            #table_row += "\\textit{Gaia}"
        
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

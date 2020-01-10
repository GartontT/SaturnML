# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:40:07 2020

Display the theta component of the magnetic field measured by Cassini in blue 
with all Titan flyby intervals highlighted in red.

@author: tmg1v19
"""

#=====Import necessary libraries=====#
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd

#=====Identifies directory path to this script=====#
dir_path = os.path.dirname(os.path.realpath(__file__))

#=====Read in catalogue of Titan flybys and Cassini data=====#
Titan_flybys = pd.read_csv(dir_path + "/Events/Titan_Flyby_catalogue.txt", 
                           sep = "\t", header = 7)
SCAS_events = pd.read_csv(dir_path + "/Events/SCAS_catalogue.txt", 
                           sep = "\t", header = 4)
Cass_data = pd.read_csv(dir_path + "/Cassini_Observations/Cass_data.txt", 
                        sep = "\t", header = 457)

#=====Convert date component of pandas arrays to datetime objects=====#
cass_date = pd.to_datetime(Cass_data["Timestamp(UTC)"], 
                                     format="%d/%m/%Y %H:%M:%S.%f")
Titan_dates = pd.to_datetime(Titan_flybys["EventTime(UTC)"], 
                                          format="%Y %b %d (%j), %H:%M:%S")

SCAS_st = pd.to_datetime(SCAS_events["EventStart(UTC)"], 
                                          format="%Y %b %d (%j), %H:%M")
SCAS_en = pd.to_datetime(SCAS_events["EventEnd(UTC)"], 
                                          format="%Y %b %d (%j), %H:%M")

#=====Create an hour long interval for Titan flybys=====#
Titan_st = Titan_dates-np.timedelta64(30, 'm')
Titan_en = Titan_dates+np.timedelta64(30, 'm')

#=====Establish a matplotlib figure environment and plot Cassini data=====#
fig = plt.figure()
plt.plot_date(cass_date, Cass_data["BY_KRTP(nT)"], linestyle = 'solid', 
                                   label = '$B_\\theta$')
plt.ylabel("$|B|\ (nT)$")
plt.xlabel("Date")

#=====Loop through Titan intervals to highlight each interval=====#
for i in range(len(Titan_st)):
    if i == 0:
        plt.axvspan(*mdates.date2num([Titan_st[i], Titan_en[i]]), 
                    color = 'orange', alpha = 0.5, label = 'Titan Flybys')
    
    else:
        plt.axvspan(*mdates.date2num([Titan_st[i], Titan_en[i]]), 
                    color = 'orange', alpha = 0.5)
        
#=====Loop through SCAS intervals to highlight each interval=====#
for i in range(len(SCAS_st)):
    if i == 0:
        plt.axvspan(*mdates.date2num([SCAS_st[i], SCAS_en[i]]), color = 'red', 
                    alpha = 0.5, label = 'SCAS Intervals')
    
    else:
        plt.axvspan(*mdates.date2num([SCAS_st[i], SCAS_en[i]]), color = 'red', 
                    alpha = 0.5)

#=====Display legend of data and close figure=====#
plt.legend()
plt.show()

#=====EOF=====#

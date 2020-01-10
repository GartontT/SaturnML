# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:40:07 2020

Display the theta component of the magnetic field measured by Cassini in blue 
with all Titan flyby intervals highlighted in red 

@author: tmg1v19
"""

#=====Import necessary libraries=====#
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

#=====Read in catalogue of Titan flybys and Cassini data=====#
Titan_flybys = pd.read_csv("Events/Titan_Flyby_catalogue.txt", sep = "\t", 
                         header = 7)
Cass_data = pd.read_csv("Cassini_Observations/06001T000031_06365T235931_mrdcd_sdfgmc_krtp_1m.txt", 
                        sep = "\t", header = 457)

#=====Convert date component of pandas arrays to datetime objects=====#
cass_date = pd.to_datetime(Cass_data["Timestamp(UTC)"], 
                                     format="%d/%m/%Y %H:%M:%S.%f")
Titan_dates = pd.to_datetime(Titan_flybys["EventTime(UTC)"], 
                                          format="%Y %b %d (%j), %H:%M:%S")

#=====Create an hour long interval for Titan flybys=====#
rec_st = Titan_dates-np.timedelta64(30,'m')
rec_en = Titan_dates+np.timedelta64(30,'m')

#=====Establish a matplotlib figure environment and plot Cassini data=====#
fig = plt.figure()
plt.plot_date(cass_date, Cass_data["BY_KRTP(nT)"], linestyle = 'solid', 
                                   label = '$B_\\theta$')
plt.ylabel("$|B|\ (nT)$")
plt.xlabel("Date")

#=====Loop through Titan intervals to highlight each interval=====#
for i in range(len(Titan_dates)):
    if i == 0:
        plt.axvspan(*mdates.date2num([rec_st[i], rec_en[i]]), color = 'red', 
                    alpha = 0.5, label = 'Titan Flybys')
    
    else:
        plt.axvspan(*mdates.date2num([rec_st[i], 
                                     rec_en[i]]), color = 'red', alpha = 0.5)

#=====Display legend of data and close figure=====#
plt.legend()
plt.show()

#=====EOF=====#
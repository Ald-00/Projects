#!/usr/bin/python3

   # Copyright 2023 University of Twente

   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at

       # http://www.apache.org/licenses/LICENSE-2.0

   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.



# Implementation of the algorithm presented in:
# M.E.T. Gerards et al., "Demand Side Management using Profile Steering", IEEE PowerTech, 2015, Eindhoven.
# https://research.utwente.nl/en/publications/demand-side-management-using-profile-steering
# Interactive demo: https://foreman.virt.dacs.utwente.nl/~gerardsmet/vis/ps.html

# Optimization of devices (OptAlg file) based on the work of Thijs van der Klauw)
# https://ris.utwente.nl/ws/portalfiles/portal/12378855/thesis_T_van_der_Klauw.pdf

# Plots 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter


# Importing models
from dev.load import Load
from dev.battery import Battery
from dev.electricvehicle import ElectricVehicle
from dev.heatpump import HeatPump

# Import Profile Steering
from profilesteering import ProfileSteering
from PSCollector import BatteryCollector

import numpy as np
import statistics

import fairness
from dataReader import DataReader
from simulations import AllEqual, AllDifferent, OneDifferent, IncreasingLoad

# simulation where all batteries are equal in capacity
# sim1 = AllEqual(sim_name="All-equal", capacity=5000, max_load=5000, load_mult=1)
# sim1.run(10)
# sim1.showPlot()
"""
NOTES: Not much can be said about this one
"""

# simulation where the capacity of the batteries is too low to balance the grid
# sim2 = AllEqual(sim_name="Not-enough-capacity",capacity=400)
# sim2.run(10)
# sim2.showPlot()
"""
NOTES: Very fair, because all batteries are used excessively.
"""

# simulation where the capacity is more than enough to balance the grid
# sim3 = AllEqual(sim_name="More-than-enough-capacity",capacity=25000)
# sim3.run(42)
# sim3.showPlot(rolling_average=True)
"""
NOTES: a few batteries are used excessively, very unfair.
"""

# sim3_shuffle = AllEqual(sim_name="More-than-enough-capacity-shuffle", shuffle=True, capacity=25000)
# sim3_shuffle.run(42)
# sim3_shuffle.showPlot(rolling_average=True)
"""
NOTES: 	this is a lot more fair over multiple days. Every single day, one battery still gets treated unfairly
		Every day this is a different battery, meaning the fairness more or less evens out over time
		
		When a rolling average is used for the fairness inputs, the fairness goes up significantly
"""

sim4 = AllDifferent(sim_name="All-different",start_size=12500, step_size=-500)
sim4.run(20)
sim4.showPlot(rolling_average=True)
"""
NOTES: At low capacities shows the difference between proportional and normal. usage is multiple of capacity for all batteries
	   Otherwise shows unfairness
"""

# sim4_shuffle = AllDifferent(sim_name="All-different-shuffle", shuffle=True, start_size=6260, step_size=1)
# sim4_shuffle.run(28)
# sim4_shuffle.showPlot(rolling_average=True)
"""
Notes:	Shuffling the batteries does not appear to have any effects when the values are different.
"""

# sim5 = OneDifferent(sim_name="One-different",capacity=1000, outlier_capacity=6000)
# sim5.run(10)
# sim5.showPlot()
"""
NOTES: 	C=1000, C_outlier=6000 shows big difference between jain and Gini results
		C=400, C_outlier=500 mostly fair because small difference & it seems battery is already used as much as possible
"""

# sim6 = IncreasingLoad(sim_name="increasing-load",capacity=1000)
# sim6.run(100)
# sim6.showPlot()


"""
TODO:
DONE: make sure battery capacities are carried through to the next day

Test without EVs and heatpumps
	
	
Test higher base load only
	- behaves just as if the battery capacity was smaller. indicates some sort of relation between load, capacity and fairness
	

Test with completely empty or full batteries at the start of the day
	- profile steering seems to try and end at the exact same value it started at. e.g it starts empty, it wants to end empty
	- behaves very strange when batteries are full. This is because battery sets tagetSoC to initialSoC, meaning it always aims to be full
	TODO: change targetSoC behaviour
	
Test with very high or low charge/discharge rates
"""
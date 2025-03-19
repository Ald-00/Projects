
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
import operator

class BaseSim:
	def __init__(self, households=23, max_iters=100, sim_name = "", shuffle=False):
		# Initialisation
		self.devices = []
		self.simName = sim_name

		# amount of days simulated
		self.days = 0

		# contains the fairness results
		self.fairnessResults_proportional = []
		self.fairnessResults_normal = []

		# Settings
		self.intervals = 96
		# desired_profile = [0]*intervals		# d in the PS paper
		# power_profile = [0]*intervals		# x in the PS paper

		self.e_min = 0.001  # e_min in the PS paper
		self.max_iters = max_iters  # maximum number of iterations
		self.shuffle = shuffle
		self.households = households

	def setupLoads(self, EVs : list, heatPumps : list, maxLoad=5000, loadMult=1):
		csvReader = DataReader('data_aardehuizen_without_carport.csv')
		loads = csvReader.readAllLoads(self.intervals)
		for i in range(0, self.households):
			l = Load(name=f"Load{i}", loads=[x * loadMult for x in loads[i]])
			l.max = maxLoad
			self.devices.append(l)

		for i in range(len(EVs)):
			self.devices.append(EVs[i])

		for i in range(len(heatPumps)):
			self.devices.append(heatPumps[i])

	def setupBatteries(self, batteries : list):
		if len(batteries) != self.households:
			raise Exception("n-batteries does not match n-households")

		for i in range(0, self.households):
			self.devices.append(batteries[i])

	def run1(self):
		"""
		Runs 1 simulation day
		"""

		self.days = self.days + 1

		desired_profile = [0] * self.intervals  # d in the PS paper
		power_profile = [0] * self.intervals  # x in the PS paper

		ps = ProfileSteering(self.devices)
		power_profile = ps.init(desired_profile)
		power_profile = ps.iterative(self.e_min, self.max_iters, self.shuffle, fairness="jain")
		ps.collectAll()

		self.fairnessResults_normal.append(fairness.getFairnessInputs(ps, False))
		self.fairnessResults_proportional.append(fairness.getFairnessInputs(ps))

		# update the SOC to be the value from the previous day
		for dev in self.devices:
			if type(dev) == BatteryCollector:
				# print(dev.profile)
				dev.initialSOC = dev.updateSOC()


	def run(self, days=7):
		"""
		Runs this simulation
		"""
		print(f"Running {self.simName} simulation")
		for i in range(days):
			self.run1()
			print(f"Day \x1b[1;31m{self.days}\x1b[0m, Jains: \x1b[1;33m{fairness.jain(self.fairnessResults_normal[i])}\x1b[0m, "
			  f"Ginis: \x1b[1;32m{fairness.gini(self.fairnessResults_normal[i])}\x1b[0m, "
			  f"QoE:   \x1b[1;35m{fairness.hoẞfeld(self.fairnessResults_normal[i], max(self.fairnessResults_normal[i]))}\x1b[0m" )

	def showPlot(self, rolling_average=False):
		"""
		Just show the generic plot function defined in the base class
		"""
		self.plotFairness(rolling_average)
		self.plotInputs()
		plt.show()

	def plotInputs(self):
		# iax is the input axis
		# cax is the capacity axis
		fig, iax = plt.subplots(layout='constrained')
		cax = iax.twinx()

		capacities = []
		for dev in self.devices:
			if type(dev) == Battery or type(dev) == BatteryCollector:
				capacities.append(dev.capacity)

		sums = []
		for house in np.transpose(self.fairnessResults_normal):
			sums.append(np.sum(house))

		x = np.arange(len(sums))  # the label locations
		width = 0.2  # the width of the bars
		capWidth = 0.4
		multiplier = 0

		cline = cax.bar(x, capacities, 2 * capWidth, color="#FAC898", zorder=1, label="Capacity")
		sline = iax.bar(x, sums, 2 * width, zorder=5, label="ESS usage")

		iax.legend([sline, cline], ["Usage", "Capacity"])
		cax.set_ybound(0, 20000)

		cax.set_ylabel("Capacity (Wh)")
		iax.set_ylabel("ESS usage (Wh)")
		iax.set_xlabel("House")
		iax.set_xticks(np.arange(len(sums), step=2))

		cax.yaxis.set_major_formatter(ticker.FuncFormatter(format_with_units))
		iax.yaxis.set_major_formatter(ticker.FuncFormatter(format_with_units))

		iax.set_zorder(cax.get_zorder() + 1)
		iax.patch.set_visible(False)


	def plotFairness(self, rolling_average=False):
		print(self.fairnessResults_normal)
		print(self.fairnessResults_proportional)

		jain_values = []
		prop_jain_values = []
		gini_values = []
		prop_gini_values = []
		# Ain't nobody care about Hoßfeld
		# hossfeld_values = []

		if rolling_average:
			self.fairnessResults_normal = self.rolling_average(self.fairnessResults_normal)
			self.fairnessResults_proportional = self.rolling_average(self.fairnessResults_proportional)

		for day in self.fairnessResults_proportional:
			prop_jain_values.append(fairness.jain(day))
			prop_gini_values.append(fairness.gini(day))
		for day in self.fairnessResults_normal:
			jain_values.append(fairness.jain(day))
			gini_values.append(fairness.gini(day))

		# Graphing time!!

		fig, axis = plt.subplots(2, 2)

		axis[1][1].set_xlabel("Day")
		axis[1][1].set_ylabel("Proportional fairness input")
		axis[1][0].set_xlabel("Day")
		axis[1][0].set_ylabel("Fairness input")


		axis[1][0].yaxis.set_major_formatter(ticker.FuncFormatter(format_with_units))

		xs = np.arange(len(self.fairnessResults_normal))

		circle_error = 0.05
		allMax_proportional = max(map(max, self.fairnessResults_proportional))
		allMin_proportional = min(map(min, self.fairnessResults_proportional))
		allMax_normal = max(map(max, self.fairnessResults_normal))
		allMin_normal = min(map(min, self.fairnessResults_normal))
		margin_proportional = circle_error * (allMax_proportional - allMin_proportional)
		margin_normal = circle_error * (allMax_normal - allMin_normal)

		# The max difference between 2 values for them to combine into a circle,
		# proportional to the maximum and minimum values of the entire week
		print("HAHAHAHA DEEMSS!!!")
		print(allMax_proportional, allMin_proportional, allMax_normal, allMin_normal)
		print(f"Margins: {margin_proportional}, {margin_normal}")

		for i, day in enumerate(self.fairnessResults_proportional):
			# Calculate sizes based on closeness of y-values (within 0.1)
			sizes = []
			
			for house in day:
				size = sum(abs(house - other_house) <= margin_proportional for other_house in day)
				sizes.append(size * 40)  # Scale for visibility

			xs2 = np.repeat(i, len(day))

			# Create scatter plot
			axis[1][1].scatter(x=xs2, y=day, s=sizes, alpha=0.7, color="#d62728")

		for i, day in enumerate(self.fairnessResults_normal):
			sizes = []

			for house in day:
				size = sum(abs(house - other_house) <= margin_normal for other_house in day)
				sizes.append(size * 40)  # Scale for visibility

			xs2 = np.repeat(i, len(day))

			# Create scatter plot
			axis[1][0].scatter(x=xs2, y=day, s=sizes, alpha=0.7, color="#d62728")

		axis[0][1].set_ylabel("Proportional fairness")
		axis[0][0].set_ylabel("Normal Fairness")

		axis[0][1].plot(prop_jain_values, color="#FA7E19", label="Jain proportional")
		axis[0][1].plot(prop_gini_values, color="#378A4D", label="Gini proportional")
		axis[0][0].plot(jain_values, color="#FA7E19", label="Jain")
		axis[0][0].plot(gini_values, color="#378A4D", label="Gini")

		# Dotted lines on opposite graphs
		# axis[0][1].plot(prop_jain_values, color="#FA7E19", label="Jain proportional", linestyle="dotted")
		# axis[0][1].plot(prop_gini_values, color="#378A4D", label="Gini proportional", linestyle="dotted")
		# axis[0][0].plot(jain_values, color="#FA7E19", label="Jain", linestyle="dotted")
		# axis[0][0].plot(gini_values, color="#378A4D", label="Gini", linestyle="dotted")
		axis[0][1].set_ybound(0, 1)
		axis[0][0].set_ybound(0, 1)
		axis[0][1].legend(prop={'size': 8})
		axis[0][0].legend(prop={'size': 8})

		axis[0][0].set_ylim(0, 1.1)
		axis[0][1].set_ylim(0, 1.1)

		# axis[0][0]

	def rolling_average(self, fg : list):
		padding = [0,0,0]
		fg = np.transpose(fg)
		weekly_avg = []
		for house in fg:
			weekly_avg.append((padding + [(house[i-3] + house[i-2] + house[i-1] + house[i] + house[i+1] + house[i+2] + house[i+3]) / 7.0 for i in range(3, len(house)-3)] + padding))
		return np.transpose(weekly_avg)

# Number formatter because for some reason you have to make that yourself
def format_with_units(x, pos):
	if x >= 1e6:
		return f'{x / 1e6:.0f}M'
	elif x >= 1e3:
		return f'{x / 1e3:.0f}k'
	else:
		return f'{x:.0f}'


class AllEqual(BaseSim):
	"""
	Simulation where all batteries are equal
	"""
	def __init__(self,  households=23, max_iters=100, sim_name="AllEqual", shuffle=False, capacity=4000, max_load=5000, load_mult=1):
		"""
		Setup the simulation.
		inits base class and sets up the grid
		"""

		super().__init__(households, max_iters, sim_name, shuffle)

		hp = []
		ev = []
		bat = []

		for i in range(0, households):
			# setup EV and heatpumps
			ev.append(ElectricVehicle(name=f"ElectricVehicle{i}"))
			hp.append(HeatPump(name=f"HeatPump{i}"))

			# setup battery
			battery = BatteryCollector(name=f"Battery{i}")
			battery.capacity = capacity
			battery.initialSoC = battery.capacity /2
			battery.promised = battery.capacity
			bat.append(battery)

		# add the devices
		self.setupLoads(EVs=ev, heatPumps=hp, maxLoad=max_load, loadMult=load_mult)
		self.setupBatteries(bat)

class AllDifferent(BaseSim):
	""" Simulation where all batteries are a different value """
	def __init__(self,  households=23, max_iters=100, sim_name="AllDifferent", shuffle=False, start_size=500, step_size=500):
		"""
		Setup the simulation.
		inits base class and sets up the grid
		"""
		super().__init__(households, max_iters, sim_name, shuffle)

		hp = []
		ev = []
		bat = []

		for i in range(0, households):
			# setup EV and heatpumps
			ev.append(ElectricVehicle(name=f"ElectricVehicle{i}"))
			hp.append(HeatPump(name=f"HeatPump{i}"))

			# setup battery
			battery = BatteryCollector(name=f"Battery{i}")
			battery.capacity = start_size + (i * step_size)
			battery.initialSoC = battery.capacity / 2
			battery.promised = battery.capacity
			bat.append(battery)

		# add the devices
		self.setupLoads(EVs=ev, heatPumps=hp)
		self.setupBatteries(bat)

class OneDifferent(BaseSim):
	""" Simulation where just one battery has a different value """
	def __init__(self,  households=23, max_iters=100, sim_name="OneDifferent", shuffle=False, capacity=4000, outlier_capacity=5000):
		"""
		Setup the simulation.
		inits base class and sets up the grid
		"""
		super().__init__(households, max_iters, sim_name, shuffle)

		hp = []
		ev = []
		bat = []

		for i in range(0, households):
			# setup EV and heatpumps
			ev.append(ElectricVehicle(name=f"ElectricVehicle{i}"))
			hp.append(HeatPump(name=f"HeatPump{i}"))

			# setup battery
			battery = BatteryCollector(name=f"Battery{i}")
			if i == 0:
				battery.capacity = outlier_capacity
			else:
				battery.capacity = capacity
			battery.initialSoC = battery.capacity / 2
			battery.promised = battery.capacity
			bat.append(battery)

		# add the devices
		self.setupLoads(EVs=ev, heatPumps=hp)
		self.setupBatteries(bat)

class IncreasingLoad(BaseSim):
	"""
	Simulation where all batteries are equal
	"""
	def __init__(self,  households=23, max_iters=100, sim_name="AllEqual", shuffle=False, capacity=4000):
		"""
		Setup the simulation.
		inits base class and sets up the grid
		"""

		super().__init__(households, max_iters, sim_name, shuffle)

		self.hp = []
		self.ev = []
		self.bat = []

		for i in range(0, households):
			# setup EV and heatpumps
			self.ev.append(ElectricVehicle(name=f"ElectricVehicle{i}"))
			self.hp.append(HeatPump(name=f"HeatPump{i}"))

			# setup battery
			battery = BatteryCollector(name=f"Battery{i}")
			battery.capacity = capacity
			battery.initialSoC = battery.capacity /2
			battery.promised = battery.capacity
			self.bat.append(battery)

	def run1(self):
		self.devices = []
		self.setupLoads(EVs=self.ev, heatPumps=self.hp, maxLoad=5000*(self.days+1), loadMult=0.5*(self.days+1))
		self.setupBatteries(self.bat)
		print(f"day {self.days+1}, testing with {len(self.devices)} devices, {len(self.bat)} of which are batteries. The load multiplier is {0.5*(self.days + 1)}")
		super().run1()

	def run(self, days):
		super().run(days=days)
		f = open("fairness_inputs.txt", "w")
		for line in self.fairnessResults_normal:
			f.write(str(line) + "\n")
		f.close()


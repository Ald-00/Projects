import operator
import numpy as np


def jain(values):
	if sum(values) == 0:
		return 1.0

	return sum(values) ** 2 / (len(values) * sum(map(operator.mul, values, values)))

def gini(values):
	differenceSum = 0

	if sum(values) == 0:
		return 1.0

	for a in values:
		for b in values:
			differenceSum += abs(a - b)
	
	return 1 - differenceSum / (2 * len(values) * sum(values))

def hoẞfeld(values, maximum = 1):
	stdDev = np.std(values)
	# print(stdDev)

	return 1 - (2 * stdDev) / maximum

def getFairnessInputs(profiles, proportional=True):
	fairness_inputs = []

	for i in profiles.batteries:
		# values range between 0: battery not used at all, to 1, usage=battery capacity to >1 usage>=capacity
		if proportional:
			fairness_inputs.append(i.getUsage() / i.capacity)
		else:
			fairness_inputs.append(i.getUsage())

	return fairness_inputs
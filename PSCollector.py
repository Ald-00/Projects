import operator
import matplotlib.pyplot as plt

from dev.load import Load
from dev.battery import Battery
from dev.electricvehicle import ElectricVehicle
from dev.heatpump import HeatPump


class BatteryCollector(Battery):
    def __init__(self, name=""):
        super().__init__(name)

    def getDelivered(self):
        delivered = 0

        for entry in self.profile:
            if entry >= 0:
                delivered += entry

        return delivered

    def getUsage(self, useCandidate=False):
        delivered = 0
        profile = self.candidate if useCandidate else self.profile

        for entry in profile:
            delivered += abs(entry)

        return delivered

    def updateSOC(self):
        self.initialSoC = self.initialSoC + sum(self.profile)

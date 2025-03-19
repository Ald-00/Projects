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

import operator
import matplotlib.pyplot as plt

from dev.load import Load
from dev.electricvehicle import ElectricVehicle
from dev.heatpump import HeatPump
from PSCollector import BatteryCollector

from fairness import jain, gini

import random

def getColor(dev):
    if type(dev) == HeatPump: return "red"
    if type(dev) == ElectricVehicle: return "magenta"
    if type(dev) == Load: return "gray"
    # Batteries have a weird type, this is the easiest way of doing this
    return "blue"


class ProfileSteering():

    def __init__(self, devices):
        self.devices = devices
        self.p = []  # p in the PS paper
        self.x = []  # x in the PS paper

        self.profileData = {}
        self.batteryData = {}
        self.iter = 0

        self.batteries = []

    def init(self, p):
        # Set the desired profile and reset xrange
        self.p = list(p)
        self.x = [0] * len(p)

        # Ask all devices to propose an initial planning
        for device in self.devices:
            r = device.init(p)  # request device to create a planning
            self.x = list(
                map(operator.add, self.x, r))  # Perform the summation by adding the overall profile to the planning

        # print("Initial planning", self.x)
        return self.x

    # Fairness should be either "none", "jain", or "gini"
    def iterative(self, e_min, max_iters, shuffle, fairness="none"):

        batteries_used = 0
        profile_factor = 1
        fairness_factor = 0

        #? We're waiting every night

        # Iterative Loop
        for i in range(0, max_iters):  # Note we deviate here slightly by also definint a maximum number of iterations

            devices = [x for x in self.devices]
            if shuffle:
                random.shuffle(devices)

            # Init
            best_improvement = 0
            best_device = None
            best_battery_improvement = 0
            best_battery = None

            # difference profile
            diff = list(map(operator.sub, self.x, self.p))  # diff = x - p

            # Fairness
            batteries = [b for b in self.devices if type(b) == BatteryCollector]
            old_usages = [b.getUsage() for b in batteries]
            old_jain = jain(old_usages)
            old_gini = gini(old_usages)

            # request a new candidate profile from each device
            for device in devices:
                profile_improvement = device.plan(diff)

                # Fairness
                if fairness == "none" or type(device) != BatteryCollector or fairness_factor == 0:
                    improvement = profile_improvement

                else:
                    my_usage = device.getUsage(useCandidate=True)
                    new_usages = [device.getUsage(useCandidate=True) if b==device else b.getUsage() for b in batteries]
                    
                    new_jain = jain(new_usages)
                    new_gini = gini(new_usages)
                    fairness_improvement = new_jain - old_jain if fairness == "jain" else new_gini - old_gini
                    
                    improvement = fairness_improvement * fairness_factor + profile_improvement * profile_factor
                    
                    # print("\tPI: {}, FI: {}, Total: {}".format(profile_improvement * profile_factor, fairness_improvement * fairness_factor, improvement))
                    # use maximum of first iteration
                
                if type(device) == BatteryCollector and improvement > best_battery_improvement:
                    best_battery_improvement = improvement
                    best_battery = device

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_device = device

            # Now set the winner (best scoring device) and update the planning
            if best_device is not None:
                if type(best_device) == BatteryCollector:
                    batteries_used += 1
                diff = best_device.accept()
                self.x = list(map(operator.add, self.x, diff))
                if fairness_factor == 0 and type(best_device) == BatteryCollector:
                    fairness_factor = best_improvement
                    print("Soooo the fairness factor is \x1b[1;34m{}\x1b[0m now".format(fairness_factor))

            # Now check id the improvement is good enough
            if best_improvement < e_min:
                break  # Break the loop
                # Omg thanks for that useful comment✨!
        
        print("Used \x1b[1;92m{}\x1b[0m batteries for this day".format(batteries_used))
        return self.x  # Return the profile

    def collectAll(self):
        for dev in self.devices:
            self.collectProfileData(dev)

            if type(dev) == BatteryCollector:
                self.batteries.append(dev)
                self.collectBatteryData(dev)

    def collectProfileData(self, dev):
        self.profileData[f"{dev.name}"] = dev.profile

    def collectBatteryData(self, dev):
        """
        stores a list that shows the SOC over time
        """
        soc = dev.initialSoC
        socProfile = []

        # Integrate the profile
        for entry in dev.profile:
            soc += entry
            socProfile.append(round((soc / dev.capacity) * 100, 1))

        # The difference in SoC is the battery usage for now
        self.batteryData[f"{dev.name}"] = socProfile
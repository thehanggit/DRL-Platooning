from platoon_src.intersectionController import IntersectionController
from platoon_src.platoon import Platoon
from platoon_src.vehicle import Vehicle
from platoon_src.simlib import flatten

import traci
import math

class SimulationManager():

    def __init__(self, pCreation=True, iCoordination=True, iZipping=True):
        self.intersections = []
        self.platoons = list()
        self.platoonCreation = pCreation
        self.vehicles = list()
        self.maxStoppedVehicles = dict()
        if iCoordination:
            for intersection in traci.trafficlight.getIDList():
                controller = IntersectionController(intersection, iZipping)
                self.intersections.append(controller)
    
    def createPlatoon(self, vehicles):
        # Creates a platoon with the given vehicles
        platoon = Platoon(vehicles)
        self.platoons.append(platoon)

    def getActivePlatoons(self):
        # Gets all active platoons
        return [p for p in self.platoons if p.isActive()]

    def getAllVehiclesInPlatoons(self):
        # Gets all vehicles in every active platoon
        return flatten(p.getAllVehiclesByName() for p in self.getActivePlatoons())

    def getAverageLengthOfAllPlatoons(self):
        count = 0
        length = len(self.platoons)
        for platoon in self.platoons:
            if platoon._disbandReason != "Merged" and platoon._disbandReason != "Reform required due to new leader":
                count = count + platoon.getNumberOfVehicles()
            else:
                length = length - 1
        return count/length

    def getPlatoonByLane(self, lane):
        # Gets platoons corresponding to a given lane
        return [p for p in self.getActivePlatoons() if lane == p.getLane()]

    def getPlatoonByVehicle(self, v):
        return [p for p in self.getActivePlatoons() if v in p.getAllVehiclesByName()]

    def getReleventPlatoon(self, vehicle):
        # Returns a single platoon that is most relevent to the given
        # vehicle by looking to see if the car in front is part of a platoon
        # It also checks that the platoon is heading in the right direction
        leadVeh = vehicle.getLeader()
        if leadVeh and leadVeh[1] < 10:
            possiblePlatoon = self.getPlatoonByVehicle(leadVeh[0])
            if possiblePlatoon:
                if possiblePlatoon[0].checkVehiclePathsConverge([vehicle]) and vehicle not in possiblePlatoon[0].getAllVehicles():
                    return possiblePlatoon[0]
    
    """ ccyen """
    def get_timing(self, member_list):
        Arrival_vec = []
        Green_Windows = []
        next_s = 0
        
        if member_list:
            for veh in member_list:
                
                allVehicles = traci.vehicle.getIDList()
                
                if veh in allVehicles:
                    memberID = veh.getName()
                    
                    data = traci.vehicle.getNextTLS(memberID)
                    #print("This is DATA")
                    #print(data)
                    #print("\n")
                    
                    if data:
                        ts_id = data[0][0] # traffic light ID
                        d = data[0][2] # distance to the coming intersection
                        
                        ### Get Green Windows ###
                        remaining_duration = int(traci.trafficlight.getNextSwitch(ts_id) - traci.simulation.getTime())
                        if data[0][3] == 'G':
                            next_s = int(remaining_duration + 20 + 1)
                            
                            for t in range(1, remaining_duration + 1):
                                Green_Windows.append(t)
                            for t in range(next_s, next_s + 30):
                                Green_Windows.append(t)
                        else:
                            next_s = int(remaining_duration + 1)
                            
                            for t in range(next_s, next_s + 30):
                                Green_Windows.append(t)
                            for t in range(next_s + 20 + 30, next_s + 20 + 30 + 30):
                                Green_Windows.append(t)
                        
                        
                        ### Get t_e, t_c, t_l ###
                        v = veh.getSpeed()
                        v_lim = veh.getMaxSpeed()
                        v_coast = 5 # default value in SUMO
                        #a_max = 1.5 # preset constant which denotes the maximum changing rate of speed
                        #jerk_max = 10 # preset constant which denotes the maximum changing rate of acceleration or deceleration
                        
                        a_max = veh.getAcceleration()
                        
                        Acc_history = veh.getAccHistory()
                        jerk_max = 0
                        if Acc_history:
                            Acc_list = list(Acc_history.items())
                            jerk_max = Acc_list[-1][1] / (Acc_list[-1][0] - 0) if Acc_list[-1] == Acc_list[0] else (Acc_list[-1][1] - Acc_list[-2][1]) / (Acc_list[-1][0] - Acc_list[-2][0])
                        
                        
                        if v > v_coast:
                            alpha = min((2 * a_max) / (v_lim - v), math.sqrt((2 * jerk_max) / (v_lim - v)))
                            beta = min((2 * a_max) / (v - v_coast), math.sqrt((2 * jerk_max) / (v - v_coast)))
                            
                            pi_over_2_alpha = math.pi / 2 * alpha
                            pi_over_2_beta = math.pi / 2 * beta
                            
                            t_c = math.ceil(d / v)
                            t_e = math.ceil(((d - (v * pi_over_2_alpha)) / v_lim) + pi_over_2_alpha)
                            t_l = math.ceil(((d - (v * pi_over_2_beta)) / v_coast) + pi_over_2_beta)
                                    
                            if t_c in Green_Windows:
                                t_arr = t_c
                            elif t_e in Green_Windows or t_c in Green_Windows:
                                t_arr = t_e
                            elif t_c in Green_Windows or t_l in Green_Windows:
                                t_arr = t_c
                            else:
                                t_arr = next_s
                        else:
                            t_arr = next_s
                    else:
                        t_arr = -1
                else:
                    t_arr = -1
                
                Arrival_vec.append(t_arr)
                     
        return Arrival_vec
    
    
    """ ccyen """
    def get_nextTLS_info(self, leader_list):
        info = {}
        tls_vec = {}
        #print(traci.trafficlight.getIDCount())
        #print(leader_list)
        
        if leader_list:
            for leaderID in leader_list:
                info[leaderID] = []
                """
                ts_id = [-1, -1, -1, -1]
                distance = [-999, -999, -999, -999]
                status = [0, 0, 0, 0]
                """
                ts_id = [-1]
                distance = [-999]
                status = [0]
                tls_vec[leaderID] = []
                data = traci.vehicle.getNextTLS(leaderID)
                #print("This is DATA")
                #print(data)
                #print("\n")
                for item in data:
                    #print("This is ITEM")
                    #print(item)
                    
                    tls_vec[leaderID].append(item[0])
                    
                    if item[0] == 't':
                        ts_id[0] = 1
                        distance[0] = item[2]
                        status[0] = 1 if item[3] == 'G' else 0
                    
                info[leaderID] = ts_id + distance + status
                #print(info[leaderID])
                #print(tls_vec[leaderID])
                    
        return info, tls_vec
    
    """ ccyen """
    def setReform(self, p, split):
        p.setSplit(split)
    
    """ ccyen """
    def setMaxSpeedForAPlat(self, p, newSpeed):
        p.setTargetSpeed(newSpeed)
    
    def setMaxGapForAPlat(self, maxGap):
        for p in self.getActivePlatoons():
            p.setGap(maxGap)
                
    def handleSimulationStep(self):
        allVehicles = traci.vehicle.getIDList()
        # Check mark vehicles as in-active if they are outside the map
        stoppedCount = dict()
        for v in self.vehicles:
            if v.getName() not in allVehicles:
                v.setInActive()
            # Get information concerning the number of vehicles queueing on each lane
            if v.isActive() and v.getSpeed() == 0:
                lane = v.getEdge()
                if lane in stoppedCount:
                    stoppedCount[lane] = stoppedCount[lane] + 1
                else:
                    stoppedCount[lane] = 1
        
        """ ccyen """
        # Record acceleration data for each vehicle       
        for v in self.vehicles:
            if v.isActive():
                v.recordAcceleration()
        
        # Gather statistics for amount of vehicles stopped per lane
        for lane in stoppedCount:
            if lane in self.maxStoppedVehicles:
                if stoppedCount[lane] > self.maxStoppedVehicles[lane]:
                    self.maxStoppedVehicles[lane] = stoppedCount[lane]
            else:
                self.maxStoppedVehicles[lane] = stoppedCount[lane]

        # Update all platoons active status
        for p in self.getActivePlatoons():
            p.updateIsActive()

        if self.platoonCreation:
            # See whether there are any vehicles that are not
            # in a platoon that should be in one
            vehiclesNotInPlatoons = [v for v in allVehicles if v not in self.getAllVehiclesInPlatoons()]

            for vehicleID in vehiclesNotInPlatoons:
                vehicle = Vehicle(vehicleID)
                self.vehicles.append(vehicle)
                vehicleLane = vehicle.getLane()
                # If we're not in a starting segment (speed starts as 0)
                possiblePlatoon = self.getReleventPlatoon(vehicle)
                if possiblePlatoon:
                    possiblePlatoon.addVehicle(vehicle)
                else:
                    self.createPlatoon([vehicle, ])

        # If we're doing intersection management, update each controller and add any new platoons into their
        # control
        if self.intersections:
            for inControl in self.intersections:
                inControl.removeIrreleventPlatoons()
                inControl.findAndAddReleventPlatoons(self.getActivePlatoons())
                inControl.update()
        
        if self.platoonCreation:
            # Handles a single step of the simulation
            # Update all active platoons in the scenario
            for platoon in self.getActivePlatoons():
                platoon.update()
                if platoon.canMerge() and platoon.isActive():
                    lead = platoon.getLeadVehicle().getLeader()
                    if lead:
                        leadPlatoon = self.getPlatoonByVehicle(lead[0])
                        if leadPlatoon:
                            leadPlatoon[0].mergePlatoon(platoon)
        
        """ ccyen """
        if self.platoonCreation:
            for platoon in self.getActivePlatoons():
                platoon.update()
                if platoon.canSplit() and platoon.isActive():
                    platoon_ahead, platoon_behind = platoon.splitPlatoon()
                
                    if platoon_ahead:
                        self.createPlatoon(platoon_ahead)
                        # platoon_ahead speedup
                        for p_a in self.getActivePlatoons():
                            if platoon_ahead[0] in p_a.getAllVehicles():
                                p_a.setTargetSpeed(p_a.getSpeed() + 5)
                    
                    if platoon_behind:
                        self.createPlatoon(platoon_behind)
                        # platoon_behind slow down
                        for p_b in self.getActivePlatoons():
                            if platoon_behind[0] in p_b.getAllVehicles():
                                p_b.setTargetSpeed(p_b.getSpeed() - 5)
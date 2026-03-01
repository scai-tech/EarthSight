import shelve
from typing import Dict, List, Optional, no_type_check, TYPE_CHECKING
import sys
from time import time as time_now
from typing import Dict, List
from collections.abc import Iterable

from matplotlib import pyplot as plt
from src.earthsightgs import EarthSightGroundStation
from src.metrics import Metrics
from src.utils import Time, Print
from src.routing import Routing
from src.satellite import Satellite
from src.station import Station
from src.data import Data
from src.packet import Packet
from src.topology import Topology
from src.node import Node
from src.transmission import Transmission
from src import log
from multiprocessing import Pool

import src.const as const

class Simulator:
    """
    Main class that runs simulator

    Attributes:
        timestep (float/seconds) - timestep in seconds. CURRENTLY MUST BE AN INTEGER NUMBER
        startTime (Time) - Time object to when to start
        endTime (Time) - Time object for when to end (simulation will end on this timestep. i.e if end time is 12:00 and timeStep is 10, last run will be 11:59:50)
        satList (List[Satellite]) - List of Satellite objects
        gsList (List[Station]) - List of Station objects
        topologys (Dict[str, Topology]) - dictionary of time strings to Topology objects
        recreated (bool) - wether or not this simulation was recreated from saved file. If true, then don't compute for t-1 timestep
    """
    def __init__(self, timeStep: float, startTime: Time, endTime: Time, satList: 'List[Satellite]', gsList: 'List[Station]', recreated: bool = False) -> None:

        self.timeStep = timeStep
        self.startTime = startTime
        self.time = self.startTime.copy()
        self.endTime = endTime
        self.satList = satList
        self.gsList = gsList
        self.recreated = recreated
        self.topologys: 'Dict[str, Topology]' = {}
        #log.clear_logging_file()
    
    @staticmethod
    def parallel_sat_loads(sat : Satellite, timestep):
        sat.load_data(timestep)
        sat.load_packet_buffer()

    @staticmethod
    def parallel_gs_loads(gs : EarthSightGroundStation, timestep):
        gs.load_data(timestep)
        gs.load_packet_buffer()

    @staticmethod
    def parallel_sat_power(sat : Satellite, timeStep):
        sat.generate_power(timeStep)
        sat.use_regular_power(timeStep)

    @staticmethod
    def parallel_propogation(sat : Satellite, time):
        sat.update_orbit(time)

    def run(self) -> None:
        """
        At inital, load one time step of data into object
        then schedule based off of new data
        send info
        """
        
        time : Time = self.time
        log.update_logging_time(time)        
        step_i = 0
        while time < self.endTime:
            step_i += 1
            if step_i % 60 == 0:
                Metrics.metr().print()
                for sat, data in EarthSightGroundStation.rcv_data.items():
                    log.Log("Satellite {} has delays {}".format(sat.id, data))

            s : Time = time_now()
            print("Simulation at", time.to_str())
            log.update_logging_time(time)
            
            # for now, single threaded. easily parallelzable with threadpoolexecutor
            for sat in self.satList:
                Simulator.parallel_propogation(sat, time)

            for sat in self.satList:
                Simulator.parallel_sat_loads(sat, self.timeStep)

            for gs in self.gsList:
                Simulator.parallel_gs_loads(gs, self.timeStep)

            for sat in self.satList:
                Simulator.parallel_sat_power(sat, self.timeStep)

            topology = Topology(time, self.satList, self.gsList)
            routing = Routing(topology, self.timeStep)

            Transmission(routing.bestUpLinks, topology, self.satList, self.gsList, self.timeStep, uplink=True)
            Transmission(routing.bestDownLinks, topology, self.satList, self.gsList, self.timeStep, uplink=False)

            self.logAtTimestep()
            time.add_seconds(self.timeStep)
            print("Timestep took", time_now() - s)
            
        log.close_logging_file()

    def run_parallel(self) -> None:
        time: Time = self.time
        log.update_logging_time(time)
        step_i = 0

        with Pool() as pool:
            while time < self.endTime:
                step_i += 1
                if step_i % 60 == 0:
                    for sat, data in EarthSightGroundStation.rcv_data.items():
                        print(f"Satellite {sat.id} has delays {data}")
                        log.Log(f"Satellite {sat.id} has delays {data}")

                s: Time = time_now()
                print("Simulation at", time.to_str())
                Metrics.metr().print()
                log.update_logging_time(time)

                # Parallel execution of satellite propagation
                pool.starmap(Simulator.parallel_propogation, [(sat, time) for sat in self.satList])

                # Parallel execution of satellite loads
                pool.starmap(Simulator.parallel_sat_loads, [(sat, self.timeStep) for sat in self.satList])

                # Parallel execution of ground station loads
                pool.starmap(Simulator.parallel_gs_loads, [(gs, self.timeStep) for gs in self.gsList])

                # Parallel execution of satellite power calculations
                pool.starmap(Simulator.parallel_sat_power, [(sat, self.timeStep) for sat in self.satList])

                topology = Topology(time, self.satList, self.gsList)
                routing = Routing(topology, self.timeStep)

                Transmission(routing.bestUpLinks, topology, self.satList, self.gsList, self.timeStep, uplink=True)
                Transmission(routing.bestDownLinks, topology, self.satList, self.gsList, self.timeStep, uplink=False)

                self.logAtTimestep()
                time.add_seconds(self.timeStep)
                print("Timestep took", time_now() - s)

        log.close_logging_file()
            

    def logAtTimestep(self):
        # for sat in self.satList:
        #    log.Log("Satellite Memory", sat.percent_of_memory_filled(), sat, len(sat.transmitPacketQueue), len(sat.receivePacketQueue))

        # for gs in self.gsList:
        #    log.Log("Iot Memory", len(gs.transmitPacketQueue), gs)

        log.update_logging_file()




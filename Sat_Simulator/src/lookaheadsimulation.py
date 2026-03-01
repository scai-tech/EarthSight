import copy
from typing import Dict, List, Optional, no_type_check, TYPE_CHECKING
from time import time as time_now
from typing import Dict, List
from src.utils import Time, Print, PriorityQueueWrapper
from src.routing import Routing
from src.satellite import Satellite
from src.station import Station
from src.data import Data
from src.packet import PriorityPacket
from src.topology import Topology
from src.node import Node
from src.transmission import Transmission
from src import log
import concurrent.futures
from src.nodeDecorator import NodeDecorator
from src.query import SpatialQueryEngine
from src.formula import overall_confidence_dnf
from src.receiveGS import ReceiveGS
from src.utils import Time
    
class LookaheadSimulator:
    """
    A lightweight version of the main Simulator used for predictive scheduling.

    Runs a simplified simulation over a future time window (typically 6 hours) to
    estimate satellite-ground station contacts and determine what priority images
    will be transmitted during those contacts. Rather than performing full-fidelity
    simulation, it wraps each satellite and ground station in lightweight decorators
    (LookaheadSatellite and LookaheadGS) that approximate behavior using query
    formula probabilities instead of actual ML inference.

    The result is a transmission_log that maps each satellite ID to a list of
    predicted contact windows and the priority breakdown of data expected to be
    downlinked in each window.

    Attributes:
        timeStep (float): Simulation timestep in seconds. Must be an integer number.
        startTime (Time): Time object indicating when the lookahead window begins.
        endTime (Time): Time object indicating when the lookahead window ends. The
            simulation runs up to but not including this timestep.
        time (Time): The current simulation time, advancing from startTime to endTime.
        id (int): Unique identifier for this lookahead simulation instance.
        transmission_log (Dict[int, List]): Maps each satellite ID to a list of
            predicted contact entries. Each entry is [Time, Dict[int, float]] recording
            the contact time and the amount of data (by priority 1-10) transmitted.
        satList (List[LookaheadSatellite]): Wrapped satellite objects for lookahead.
        gsList (List[LookaheadGS]): Wrapped ground station objects for lookahead.
    """
    sim_id = 0
    def __init__(self, timeStep: float, startTime: Time, endTime: Time, satList: 'List[Satellite]', gsList: 'List[Station]', engine : SpatialQueryEngine = None) -> None:
        """
        Initialize the lookahead simulator.

        Creates LookaheadSatellite and LookaheadGS wrappers around the provided
        satellites and ground stations, and initializes an empty transmission log
        for each satellite.

        Args:
            timeStep (float): Simulation timestep in seconds. Must be an integer value.
            startTime (Time): The start time of the lookahead window.
            endTime (Time): The end time of the lookahead window. Simulation runs up
                to but not including this timestep.
            satList (List[Satellite]): List of Satellite objects to wrap for lookahead.
            gsList (List[Station]): List of Station (ground station) objects to wrap.
            engine (SpatialQueryEngine, optional): The spatial query engine used by
                LookaheadSatellite to determine which queries intersect a satellite's
                ground footprint. Defaults to None.
        """
        self.timeStep = timeStep
        self.startTime = startTime
        self.endTime = endTime
        self.time = self.startTime.copy()

        self.id = LookaheadSimulator.sim_id
        LookaheadSimulator.sim_id += 1

        self.transmission_log = {sat.id: [] for sat in satList}
        self.satList = [LookaheadSatellite(sat, engine=engine, lookahead_time=self.time, sim_id=self.sim_id) for sat in satList]
        self.gsList = [LookaheadGS(gs, self.transmission_log, self.time) for gs in gsList]

        
    @staticmethod
    def parallel_sat_loads(sat, timestep):
        """
        Load data and refresh the packet buffer for a single satellite.

        Intended to be callable in parallel across multiple satellites. Invokes
        the satellite's load_data (which generates images and populates priority
        counts) followed by load_packet_buffer (which converts priority counts
        into transmit-ready packets).

        Args:
            sat (LookaheadSatellite): The satellite to load data for.
            timestep (float): The simulation timestep duration in seconds.
        """
        sat.load_data(timestep)
        sat.load_packet_buffer()

    @staticmethod
    def parallel_gs_loads(gs, timestep):
        """
        Load data and refresh the packet buffer for a single ground station.

        Intended to be callable in parallel across multiple ground stations.
        For LookaheadGS instances this is largely a no-op inherited from the
        ReceiveGS base class, since ground stations only receive data.

        Args:
            gs (LookaheadGS): The ground station to load data for.
            timestep (float): The simulation timestep duration in seconds.
        """
        gs.load_data(timestep)
        gs.load_packet_buffer()

    @staticmethod
    def parallel_propogation(sat, time):
        """
        Propagate a satellite's orbital position to the given time.

        Updates the satellite's position using its TLE-based orbit model so that
        subsequent topology and routing computations use the correct location.

        Args:
            sat (LookaheadSatellite): The satellite whose orbit to propagate.
            time (Time): The time to propagate the orbit to.
        """
        sat.update_orbit(time)


    def run(self) -> None:
        """
        Execute the lookahead simulation from startTime to endTime.

        At each timestep the method:
        1. Propagates all satellite orbits to the current time.
        2. Loads data for each satellite (populates priority counts from query
           intersections) and converts counts into transmit-ready packets.
        3. Loads data for each ground station (no-op for receive-only stations).
        4. Builds a Topology of satellite-ground station links, computes routing
           via the Routing class, and simulates packet transmission.
        5. Advances the simulation clock by one timestep.

        After the loop completes, the transmission_log (accessible via
        self.transmission_log) contains predicted contact windows and priority
        breakdowns for each satellite. The log is also printed to stdout.
        """

        while self.time < self.endTime:
            s = time_now()
            print("Looking ahead to:", self.time.to_str())
            

            # for now, single threaded. easily parallelzable with threadpoolexecutor
            for sat in self.satList:
                LookaheadSimulator.parallel_propogation(sat, self.time)

            for sat in self.satList:
                LookaheadSimulator.parallel_sat_loads(sat, self.timeStep)

            for gs in self.gsList:
                LookaheadSimulator.parallel_gs_loads(gs, self.timeStep)

            topology = Topology(self.time, self.satList, self.gsList)
            routing = Routing(topology, self.timeStep, lookahead=True)
            Transmission(routing.bestDownLinks, topology, self.satList, self.gsList, self.timeStep, uplink=False)

            self.time.add_seconds(self.timeStep)
            print("Timestep took", time_now() - s)
            

        for k, v in self.transmission_log.items():
                print(k, v)

        log.close_logging_file()

class LookaheadSatellite(NodeDecorator):
    """
    A simplified satellite decorator for lookahead simulation.

    Instead of running actual ML inference to classify captured images, this
    class uses query formula probabilities to estimate the priority distribution
    of images a satellite would capture. It intersects the satellite's ground
    position with registered spatial queries, computes the probability of each
    query being satisfied using its filter formula, and accumulates the results
    in a priority_counts dictionary.

    The satellite operates with infinite power (currentMWs = infinity), removing
    power constraints from the lookahead prediction. Packets are created from
    the priority_counts and placed in a transmit queue ordered by descending
    priority.

    Attributes:
        node (Satellite): The underlying fresh Satellite node created for this
            lookahead instance (separate from the original satellite).
        transmitPacketQueue (PriorityQueueWrapper): Queue of PriorityPackets
            ready for downlink, ordered by priority.
        engine (SpatialQueryEngine): Spatial query engine for determining which
            queries intersect the satellite's current ground footprint.
        lookahead_time (Time): Reference to the simulation's current time object.
        image_size (int): The assumed size of each image in data units (default 1000).
        currentMWs (float): Available power set to infinity to disable power limits.
        priority_counts (Dict[int, float]): Maps priority levels (1-10) to the
            accumulated estimated data volume at that priority.
    """

    def __init__(
        self,
        original_sat,
        engine: SpatialQueryEngine,
        lookahead_time: Time,
        sim_id: int = 0
    ) -> None:
        """
        Initialize a LookaheadSatellite by wrapping an existing satellite.

        Creates a fresh Satellite node with a unique ID derived from the original
        satellite's ID and the simulation ID. Copies the original satellite's
        existing transmit queue into priority_counts so the lookahead accounts for
        data already queued for downlink.

        Args:
            original_sat (EarthsightSatellite): The real satellite being wrapped.
                Its TLE and existing packet queue are used to seed the lookahead.
            engine (SpatialQueryEngine): Spatial query engine for determining query
                intersections at the satellite's position.
            lookahead_time (Time): Reference to the shared simulation time object.
            sim_id (int, optional): Unique simulation identifier used to create a
                distinct node ID. Defaults to 0.
        """
        fresh_node =  Satellite(name="Lookahead " + original_sat.name, id=str(original_sat.get_id()) + "_" + str(sim_id), tle=original_sat.tle)
        super().__init__(fresh_node)
        Satellite.idToSatellite.pop(fresh_node.id, None)
        Satellite.nameToSatellite.pop(fresh_node.name, None)

        # attributes for lookahead satellite
        self.node = fresh_node
        self.transmitPacketQueue = PriorityQueueWrapper()
        self.engine: SpatialQueryEngine = engine
        self.lookahead_time = lookahead_time
        self.image_size = 1000
        self.currentMWs = float('inf') # no power limits for lookahead
        self.priority_counts = {i : 0 for i in range(1, 11)}

        for pkt in original_sat.transmitPacketQueue.queue:
            if 1 <= pkt.priority <= 10:
                self.priority_counts[pkt.priority] += self.image_size

    def populate_cache(self, time) -> None:
        """
        Estimate image priorities captured at the current position and accumulate them.

        Queries the spatial query engine for all queries that intersect the
        satellite's current ground coordinates. Groups queries by priority tier,
        then computes the probability that each priority tier's filter formula is
        satisfied using overall_confidence_dnf. The resulting probability is
        multiplied by image_size and added to the corresponding priority_counts
        entry. Any remaining probability (images that match no query) is added
        to priority 1 (lowest priority).

        Args:
            time (Time): The current simulation time (unused in computation but
                passed for interface consistency).
        """
        queries = self.engine.get_queries_at_coord(self.node.position.to_coords())
        # seaparate queries by priority into a dict
        query_dict = {i : [] for i in range(1, 11)}
        for q in queries:
            query_dict[q.priority_tier].append(q)
        
        all_fail_prob = 1
        for pri, quer in query_dict.items():
            formula = [[(f, True) for f in f_seq] for q in quer for f_seq in q.filter_categories]
            # filters_used = [f for f_seq in formula for f in f_seq]
            prob = overall_confidence_dnf(formula, [])
            self.priority_counts[pri] += prob * self.image_size
            all_fail_prob -= prob

        self.priority_counts[1] += all_fail_prob * self.image_size
 
    def load_packet_buffer(self) -> None:
        """
        Convert accumulated priority counts into transmit-ready packets.

        Iterates from the highest priority (10) down to the lowest (1), creating
        PriorityPacket objects from the priority_counts and prepending them to the
        transmitPacketQueue. Each packet carries up to image_size bytes of data.
        Stops when the transmit queue reaches 600 packets or all priority counts
        are exhausted. This ensures highest-priority data is transmitted first
        during downlink.
        """
        pri = 10
        while len(self.transmitPacketQueue) < 600 and any(self.priority_counts.values()):
            while self.priority_counts[pri] > 0 and pri > 0:
                self.transmitPacketQueue.appendleft(PriorityPacket(priority=pri, infoSize = min(self.image_size, self.priority_counts[pri]), relevantNode=self.node))
                self.priority_counts[pri] = max(0, self.priority_counts[pri] - self.image_size)
                if self.priority_counts[pri] < 0:
                    self.priority_counts[pri] = 0
            pri -=  1

    def get_cache_size(self) -> int:
        """
        Return the current cache size of the satellite.

        Returns:
            int: The number of items currently stored in the satellite's cache.
        """
        return self.cache_size

    def percent_of_memory_filled(self) -> float:
        """
        Return the fraction of the satellite's memory that is currently in use.

        Computes the ratio of the data queue length to the maximum capacity
        (10,000,000 entries).

        Returns:
            float: A value between 0.0 and 1.0 representing the percentage of
                memory filled.
        """
        return len(self.dataQueue) / 10000000

    def load_data(self, timeStep: float) -> None:
        """
        Simulate data acquisition for one timestep.

        Generates power for the timestep (though power is effectively unlimited),
        then divides the timestep into 45 sub-intervals. For each sub-interval,
        calls populate_cache to estimate what priority images the satellite would
        capture at its current position during that fraction of the timestep.

        Args:
            timeStep (float): The duration of the simulation timestep in seconds.
        """
        self.generate_power(timeStep)
        time_copy = self.lookahead_time.copy()
        img_r = timeStep / 45
        for _ in range(45):
            self.populate_cache(time_copy)
            time_copy.add_seconds(img_r)


class LookaheadGS(ReceiveGS):
    """
    A simplified ground station decorator for lookahead simulation.

    Instead of fully processing received images, this ground station logs the
    priority and size of each received packet into a shared transmission_log
    dictionary. Nearby transmissions (within 180 seconds of each other) are
    merged into a single contact window entry, accumulating data volumes by
    priority level.

    This allows the lookahead simulator to predict which priority tiers of data
    will be downlinked during each satellite-ground station contact window.

    Attributes:
        transmission_log (Dict[int, List]): Shared reference to the simulator's
            transmission log. Maps satellite IDs to a list of contact entries,
            where each entry is [Time, Dict[int, float]] representing the contact
            time and data volume received per priority level (1-10).
        time (Time): Reference to the shared simulation time object.
    """

    def __init__(self, node, transmission_log, time):
        """
        Initialize a LookaheadGS wrapping an existing ground station node.

        Args:
            node (Node): The original ground station node to wrap. Typically a
                Station object whose position and communication parameters are
                reused.
            transmission_log (Dict[int, List]): The shared transmission log
                dictionary from the LookaheadSimulator, mapping satellite IDs
                to their predicted contact entries.
            time (Time): Reference to the simulation's current time object,
                used to timestamp received transmissions.
        """
        super().__init__(node)
        self.transmission_log = transmission_log
        self.time = time

    def receive_packet(self, pck):
        """
        Log a received packet's priority and size into the transmission log.

        Ignores packets with priorities outside the 1-10 range. Extracts the
        originating satellite's ID from the packet's relevantNode. If the most
        recent log entry for that satellite is within 180 seconds of the current
        time, the packet's data is merged into that existing contact window entry.
        Otherwise, a new contact window entry is created.

        Args:
            pck (PriorityPacket): The received packet containing priority, infoSize,
                and relevantNode fields. The relevantNode's ID (before the underscore
                separator) identifies the originating satellite.
        """
        # transmission log:
        # dict[Node : List[time, dict[priority : count]]]

        if not 1 <= pck.priority <= 10:
            return
        relevant_id = int(pck.relevantNode.id.split("_")[0])
        node_log = self.transmission_log[relevant_id]
        if node_log and -180 <= Time.difference_in_seconds(self.time, node_log[-1][0]) <= 180:
            node_log[-1][0] = self.time.copy()
            node_log[-1][1][pck.priority] += pck.infoSize
            
        else:
            node_log.append([self.time, {i: 0 for i in range(1, 11)}])
            node_log[-1][1][pck.priority] += pck.infoSize 
        
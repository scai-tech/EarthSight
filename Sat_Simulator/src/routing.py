import itertools
from typing import List, Dict # type: ignore
from enum import Enum # type: ignore

import matplotlib.pyplot as plt # type: ignore
from time import time as time_now # type: ignore

import numpy as np # type: ignore
import networkx as nx # type: ignore

from src.satellite import Satellite
from src.station import Station
from src.links import Link
from src.topology import Topology
from src.log import Log

distanceBetweenGS = {}
lastTransmitted = {}

distanceBetweenGSlookahead = {}
lastTransmittedlookahead = {}
class Routing:
    """
    Class that creates the scheduling of the different satellites

    Attributes:
        topology (Topology) - Instance of Topology class created at this time
        bestLinks (Dict[Satellite][Station] = Link) - the links that were scheduled
    """
    def __init__(self, top: 'Topology', timeStep:float, lookahead = False) -> None:
        """
        Initialize the Routing instance by storing topology information and
        immediately computing the best uplink and downlink schedules.

        Args:
            top (Topology): The current network topology containing satellites,
                ground stations, and their possible links.
            timeStep (float): The duration of the current simulation timestep,
                used to define transmission windows.
            lookahead (bool): If True, use separate lookahead-specific caches
                for ground station distances and last-transmitted counters
                instead of the primary caches. Defaults to False.
        """
        self.timeStep = timeStep
        self.topology = top
        self.lookahead = lookahead
        self.bestUpLinks, self.bestDownLinks = self.schedule_best_links()
    
    def schedule_best_links(self) -> 'List[Dict[Satellite, Dict[Station, Link]]]':
        """
        Public method to schedule best links. If you want to change the routing mechanism, you can change the ROUTING_MECHANISM variable in const.py
        Returns:
            Dict[Satellite][Station] = Link - the links that were scheduled. If you try to schedule a link that is not possible, it should return a keyerror
        """
        possible_uplinks = self.schedule_uplink(self.topology.possibleUpLinks)
        possible_downlinks = self.schedule_downlink(self.topology.possibleDownLinks)
        return possible_uplinks, possible_downlinks
    
    def schedule_uplink(self, possibleLinks):
        """
        Schedule uplink transmissions from ground stations to satellites.

        Iterates over all ground stations that have pending data in their
        transmit queues. For each such station, peeks at the next packet to
        determine the target satellite, then searches the station's uplinks
        for the link to that target. When found, a transmission is assigned
        for the full duration of the current timestep.

        Args:
            possibleLinks: The collection of possible uplink connections
                from the current topology (typically
                ``self.topology.possibleUpLinks``).

        Returns:
            The same ``possibleLinks`` object, with transmissions assigned
            on the relevant links.
        """
        for gs in self.topology.groundList:
            if gs.has_data_to_transmit():
                pkt = gs.transmitPacketQueue.pop()
                target = pkt.relevantNode
                gs.transmitPacketQueue.append(pkt)
                for link in self.topology.nodeUpLinks[gs]:
                    if link.sat == target:
                        link.assign_transmission(0, self.timeStep, 0, gs)
                        print("Assigning transmission a for nodes: ", gs, " and ", target)
        return possibleLinks

    def schedule_downlink(self, possibleLinks):
        """
        Schedule downlink transmissions from satellites to ground stations
        using a graph-based maximum weight independent set approach.

        This method builds an interference/conflict graph where each node is a
        candidate downlink (satellite-to-ground-station link) weighted by its
        SNR, and edges connect links that would interfere with each other.
        Interference arises from two sources:

        1. **Same-satellite conflicts**: Two links originating from the same
           satellite whose destination ground stations are too close together
           (distance below a threshold).
        2. **Same-ground-station conflicts**: Two links from different
           satellites targeting the same ground station, or links that share
           a ground station through transitive satellite connections.

        The maximum weight independent set (MWIS) is then solved via a greedy
        heuristic on the conflict graph. The heuristic iteratively selects the
        node that minimises a cost ratio (weight vs. neighbour weight) and
        removes its neighbours, yielding an interference-free set of
        co-scheduled downlinks.

        For each satellite in the independent set, the co-scheduled links have
        their datarates updated to account for shared bandwidth
        (``Link.update_link_datarates``), and a transmission is assigned on the
        highest-SNR link for the full timestep duration.

        On the first invocation the method lazily initialises two global
        caches -- ground-station pairwise distances and per-satellite
        last-transmitted counters -- selecting the lookahead or primary cache
        based on ``self.lookahead``. The last-transmitted counters are
        incremented at the end of every call.

        Args:
            possibleLinks: A dict mapping each satellite to a dict of
                ground stations and their corresponding Link objects
                (typically ``self.topology.possibleDownLinks``).

        Returns:
            The same ``possibleLinks`` object, with transmissions assigned
            on the selected independent set of downlinks.
        """
        global distanceBetweenGS, lastTransmitted, distanceBetweenGSlookahead, lastTransmittedlookahead
        
        if self.lookahead:
            if len(distanceBetweenGSlookahead) == 0:
                distanceBetweenGSlookahead = {gs1: {gs2: gs1.position.get_distance(gs2.position) for gs2 in self.topology.groundList if gs1.receiveAble} for gs1 in self.topology.groundList if gs1.receiveAble}
            if len(lastTransmittedlookahead) == 0:
                lastTransmittedlookahead = {sat: 1 for sat in self.topology.satList}
        else:    
            if len(distanceBetweenGS) == 0:
                distanceBetweenGS = {gs1: {gs2: gs1.position.get_distance(gs2.position) for gs2 in self.topology.groundList if gs1.receiveAble} for gs1 in self.topology.groundList if gs1.receiveAble}
            if len(lastTransmitted) == 0:
                lastTransmitted = {sat: 1 for sat in self.topology.satList}
        print("Scheduling Downlink Ours")
        grph = nx.Graph()
        satLinks = {sat: [] for sat in self.topology.satList}
        validGs = {gs: [] for gs in self.topology.groundList if gs.receiveAble}
        for sat in self.topology.satList:
            for gs, link in possibleLinks[sat].items():
                if gs.receiveAble:
                    grph.add_node(link, weight=(link.snr + 10000))
                    satLinks[sat].append(link)
                    validGs[gs].append(link)

        for sat in satLinks.keys():
            for link1 in satLinks[sat]:
                for link2 in satLinks[sat]:
                    if link1 != link2 and link1.gs.receiveAble and link2.gs.receiveAble and \
                    ((not self.lookahead and distanceBetweenGS[link1.gs][link2.gs] < 0) or (self.lookahead and distanceBetweenGSlookahead[link1.gs][link2.gs] < 0)):
                        grph.add_edge(link1, link2)
                
        for gs in self.topology.groundList:
            if gs.receiveAble:
                for link in self.topology.nodeDownLinks[gs]:
                    for link2 in self.topology.nodeDownLinks[gs]:
                        if link != link2 and link.gs.receiveAble and link2.gs.receiveAble and link.sat != link2.sat:
                            grph.add_edge(link, link2)
                        for link3 in self.topology.nodeDownLinks[link2.sat]:
                            if link != link3 and link.gs.receiveAble and link3.gs.receiveAble and link.sat != link3.sat:
                                grph.add_edge(link, link3)    

        #plot the graph spread out so we can see it better
    
        if len(grph.edges)  != 0:
            #links = nx.maximal_independent_set(grph)
            #this algo is taken from https://stackoverflow.com/questions/30921996/heuristic-to-find-the-maximum-weight-independent-set-in-an-arbritary-graph
            #this solves for minimum weight independent set so we negate the weights
            
            adj_0 = nx.adjacency_matrix(grph).todense()
            a = -np.array([-grph.nodes[u]['weight'] for u in grph.nodes])
            IS = -np.ones(adj_0.shape[0])
            while np.any(IS==-1):
                rem_vector = IS == -1
                adj = adj_0.copy()
                adj = adj[rem_vector, :]
                adj = adj[:, rem_vector]

                u = np.argmin(a[rem_vector].dot(adj!=0)/a[rem_vector])
                n_IS = -np.ones(adj.shape[0])
                n_IS[u] = 1
                neighbors = np.argwhere(adj[u,:]!=0)
                if neighbors.shape[0]:
                    n_IS[neighbors] = 0
                IS[rem_vector] = n_IS
            
            goodInds = np.argwhere(IS == 1)
            indepedentLinks = [list(grph.nodes)[int(i[0])] for i in goodInds]
            scheduled = {}
            scheduledLinks = {}
            for link in indepedentLinks:
                sat = link.sat
                if sat in scheduledLinks:
                    scheduledLinks[sat].append(link)
                else:
                    scheduledLinks[sat] = [link]
                link.mark_gs_listening()
                if sat in scheduled:
                    if scheduled[sat].snr < link.snr:
                        scheduled[sat] = link
                else:
                    scheduled[sat] = link
                    
            for sat in scheduled.keys():
                Link.update_link_datarates(scheduledLinks[sat])
                scheduled[sat].assign_transmission(0, self.timeStep, 0, sat)
        
        # print("Done Scheduling")
        if self.lookahead:
            lastTransmittedlookahead = {sat: lastTransmittedlookahead[sat] + 1 for sat in self.topology.satList}
        else:
            lastTransmitted = {sat: lastTransmitted[sat] + 1 for sat in self.topology.satList}
        return possibleLinks
    
from typing import TYPE_CHECKING
from itertools import chain
import random # type: ignore
from time import time as timeNow
import numpy as np

from src.links import Link
from src.log import Log
from src.utils import Print
import src.const as const

if TYPE_CHECKING:
    from src.satellite import Satellite
    from src.station import Station
    from src.node import Node
    from src.packet import Packet
    from typing import List, Dict, Optional
    from src.topology import Topology

class CurrentTransmission:
    """Tracks the state of a single in-progress transmission from one sender.

    Holds the sending node, the set of intended receiving nodes, the channel
    being used, and the list of packets with their per-packet timing windows.
    Also stores per-receiver Packet Error Rate (PER) and Signal-to-Noise Ratio
    (SNR) values so that the delivery step can decide whether each receiver
    successfully decodes each packet.
    """

    def __init__(self, sending: 'Node', receivingNodes: 'List[Node]', channel: 'int') -> None:
        """Initialise a CurrentTransmission record.

        Args:
            sending: The node that is transmitting packets.
            receivingNodes: The list of nodes that are potential receivers of
                this transmission.
            channel: The integer channel identifier on which the transmission
                takes place.
        """
        self.sending = sending
        self.receivingNodes = receivingNodes
        self.receivingChannel = channel

        self.packets: 'List[Packet]' = []
        self.packetsTime: 'List[tuple[float]]' = [] #List of startTimes and endTimes for each packet relevative to the start of the timestep
        self.PER: 'Dict[Node, float]' = {} #the PER for each node. Should be set to 1 if the node isn't scheduled to receive the packet
        self.SNR: 'Dict[Node, float]' = {}
        
class Transmission:
    """Orchestrates packet transfer between nodes for a single simulation timestep.

    On construction the class immediately builds a list of
    ``CurrentTransmission`` objects via :meth:`get_new_transmissions` (which
    drains each sender's transmit queue according to the scheduled links and
    timing windows) and then delivers those packets to their receivers via
    :meth:`transmit` (which applies per-receiver PER-based random drops and
    power checks before handing packets to each receiving node).

    Supports both uplink (ground station to satellite) and downlink (satellite
    to ground station) directions, controlled by the *uplink* flag.
    """

    def __init__(self, links: 'Dict[Node, Dict[Node, Link]]', topology: 'Topology', satList: 'List[Satellite]', gsList: 'List[Station]', timeStep: 'int', uplink = False) -> None:
        """Create a Transmission and immediately execute it.

        After storing references to the link map, topology, node lists, and
        timing information, the constructor calls :meth:`get_new_transmissions`
        to build the set of ``CurrentTransmission`` objects for this timestep
        and then calls :meth:`transmit` to deliver the packets.

        Args:
            links: Nested dictionary mapping each sending node to a dictionary
                of its receiving nodes and their corresponding ``Link``
                objects.
            topology: The current network ``Topology``, used to look up
                per-node uplink/downlink link lists when beamforming is not
                active.
            satList: List of all ``Satellite`` nodes in the simulation.
            gsList: List of all ground ``Station`` nodes in the simulation.
            timeStep: The duration of the current simulation timestep (used to
                clamp transmission end times).
            uplink: If ``True``, treat the transmission direction as uplink
                (ground to satellite). Defaults to ``False`` (downlink).
        """
        self.links = links ##links should be a case of Dict[Satellite][Station] = links
        self.linkList = list( chain( *(list(d.values()) for d in self.links.values()) ))
        self.nodes = [i for i in chain(satList, gsList)]
        self.topology = topology
        self.timeStep = timeStep
        self.satList = satList
        self.gsList = gsList
        self.uplink = uplink

        transmissions = self.get_new_transmissions()
        self.transmit(transmissions)
        
    def transmit(self, transmissions: 'List[CurrentTransmission]'):
        """Deliver packets from all current transmissions to their receivers.

        For every ``CurrentTransmission``, each packet is mapped to each
        potential receiving node on the appropriate channel. The method then
        iterates over receivers and channels and, for each packet:

        1. Draws a uniform random number and compares it to the receiver's
           Packet Error Rate (PER). If the draw is at or below the PER the
           packet is silently dropped.
        2. Checks whether the receiver has sufficient power budget to receive
           for the packet's duration. If so, the power is consumed and the
           packet is handed to the receiver via ``receive_packet``.

        Args:
            transmissions: List of ``CurrentTransmission`` objects produced by
                :meth:`get_new_transmissions`.
        """
        #so here's how this works
        #we have each device which has been scheduled from x time to y time
        #so let's do this, for each reception device's channel, we store a list of each packet's (startTime, endTime)
        #we then find any collisions in the startTime and endTime

        #receiving is a dict[node][channel] = List[ (packet, (startTime, endTime), PER, SNR) ]

        receiving = {}
        #s = timeNow()
        for transmission in transmissions:
            for node in transmission.receivingNodes:
                for i in range(len(transmission.packets)):
                    lst = receiving.setdefault(node, {})
                    chanList = lst.setdefault(transmission.receivingChannel, [])
                    chanList.append((transmission.packets[i], transmission.packetsTime[i], transmission.PER[node], transmission.SNR[node], str(transmission.sending), str(transmission.packets[i])))
                    #receiving[node][transmission.receivingChannel].append((transmission.packets[i], transmission.packetsTime[i], transmission.PER[node], str(transmission.sending), str(transmission.packets[i])))
        
        #print("Time to create receiving dict", timeNow() - s)
        
        #print receiving but call the repr function for each object
        #now let's go through each receiving and find any overlapping times 
        #t = timeNow()
        for receiver in receiving.keys():
            for channel, blocks in receiving[receiver].items():
                if len(blocks) == 0:
                    continue

                for block in blocks:
                    packet = block[0]
                    PER = block[2]
                    
                    #let's check if this packet gets dropped by PER
                    
                    if random.random() <= PER:
                        #print("Packet dropped", packet, receiver)
                        pass
                        #Log("Packet dropped", packet)
                    else:
                        #print("Packet received", packet, receiver)
                        time = block[1][1] - block[1][0]
                        if receiver.has_power_to_receive(time):
                            receiver.use_receive_power(time)
                            receiver.receive_packet(packet)

    def get_new_transmissions(self) -> 'Dict[int, List]':
        """Build the list of ``CurrentTransmission`` objects for this timestep.

        Iterates over every scheduled ``Link``, and for each link's scheduled
        transmission window:

        * Validates that the sending node is not already transmitting on a
          different link (raises ``Exception`` if it is).
        * Determines the set of receiving nodes. When the sender uses
          beamforming, only the single link partner is targeted. Otherwise all
          nodes connected via the topology's uplink or downlink list are
          considered, and any receiver whose link data rate is below the
          current link's rate or whose link is not listening has its PER set
          to 1 (guaranteed drop).
        * Drains the sender's ``transmitPacketQueue`` one packet at a time,
          each consuming a fixed 0.1-second slot, until the transmission
          window is exhausted, the queue is empty, or the sender lacks
          transmit power. In uplink mode only a single packet is sent per
          window to avoid routing packets intended for different satellites.

        Returns:
            A list of ``CurrentTransmission`` objects, each fully populated
            with packets, timing tuples, and per-receiver PER / SNR values.
        """
        devicesTransmitting = {}
        currentTransmissions = []
        
        for link in self.linkList:
            #print("Link has {} start times".format(len(link.startTimes)), *link.nodeSending)
            for idx in range(len(link.startTimes)):
                sending = link.nodeSending[idx]
                startTime = link.startTimes[idx]
                channel = link.channels[idx]
                endTime = min(link.endTimes[idx], self.timeStep)
                
                #let's do this to avoid duplicate sending - maybe think of a better way to handle this??
                if sending in devicesTransmitting:
                    #check if this is from the same  or another, if its in another - raise an exception
                    if link is devicesTransmitting[sending]:
                        pass
                    else:
                        raise Exception("{} is transmitting on two links at the same time".format(sending))
                devicesTransmitting[sending] = link
                
                receiving = []
                datarate = 0
                if sending.beamForming:
                    receiving = [link.get_other_object(sending)]
                    per = {receiving[0]: link.PER}
                    snr = {receiving[0]: link.snr}
                    datarate = link.get_relevant_datarate(sending)
                else:
                    listOfLinks = self.topology.nodeUpLinks[sending] if self.uplink else self.topology.nodeDownLinks[sending]
                    receiving = [i.get_other_object(sending) for i in listOfLinks]
                    per = {i.get_other_object(sending): i.PER for i in listOfLinks}
                    snr = {i.get_other_object(sending): i.snr for i in listOfLinks}
                    receiving = [i for i in receiving if i.receiveAble]
                    
                    #lst = [i.get_relevant_datarate(sending) for i in listOfLinks if i.get_other_object(sending).receiveAble]
                    #datarate = min(lst)
                    datarate = link.get_relevant_datarate(sending)
                    for i in listOfLinks:
                        if i.get_relevant_datarate(sending) < datarate or not i.is_listening():
                            per[i.get_other_object(sending)] = 1
                    #Log("Sending", sending, "receiving", *receiving, "channel", channel, "datarate", datarate, "PER", per, "SNR", snr, "totalPackets")
                
                trns = CurrentTransmission(sending, receiving, channel)
                # print(sending)
                #now let's assign the packets within this transmission
                currentTime = startTime
                # print(currentTime, endTime, len(sending.transmitPacketQueue))
                tp = 0
                while currentTime < endTime and len(sending.transmitPacketQueue) > 0:
                    timeForNext = 0.1
                    if currentTime + timeForNext <= endTime and sending.has_power_to_transmit(timeForNext):

                        tp += 1
                        sending.use_transmit_power(timeForNext)
                        pck = sending.send_data()
                        trns.packets.append(pck)
                        trns.packetsTime.append((currentTime, currentTime + timeForNext))
                        currentTime = currentTime + timeForNext

                        if self.uplink:
                            break # only transmit one packet because it has one destination satellite - don't send packets meant for sat1 and sat2 to sat1.
                    else:
                        if currentTime + timeForNext > endTime:
                            pass
                            # print("Time for next packet exceeds end time, breaking")
                        else:
                            print("Not enough power to transmit, breaking")
                        break
                # if tp > 0:
                #     print("transmitted", tp)
                # Log("Sending", sending, "receiving", *receiving, "channel", channel, "datarate", datarate, "PER", per, "SNR", snr, "totalPackets", len(trns.packets))
                assert len(trns.packets) == len(trns.packetsTime)
                trns.PER = per
                trns.SNR = snr
                currentTransmissions.append(trns)
                        
        return currentTransmissions
        
        

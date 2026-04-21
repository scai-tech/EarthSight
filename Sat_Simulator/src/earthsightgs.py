from src.image import Image
from src.receiveGS import ReceiveGS
from src.packet import Packet
from src.scheduler import EarthSightScheduler
from src.log import get_logging_time
from src.utils import Time
from src.node import Node
from . import log


class EarthSightGroundStation(ReceiveGS):
    """Ground station node decorator for EarthSight-specific behaviour.

    Extends ReceiveGS to handle two core responsibilities:

    1. **Schedule management** -- when a satellite sends a "schedule request"
       packet, the ground station invokes the configured scheduler and
       transmits the resulting schedule back to the satellite.
    2. **Image reception and delay tracking** -- when a satellite downlinks
       an "image" packet, the ground station records delivery-delay metrics
       (time elapsed between image capture and reception) broken down by
       priority level and by satellite.

    Class Attributes:
        rcv_data (dict): Per-satellite reception statistics.  Keyed by
            satellite identifier, each value is a dict mapping priority
            levels (int) to cumulative delay, along with corresponding
            ``count_<priority>`` and ``unavoidable_delay_<priority>``
            entries.
    """

    mode = 0 # 0 for serval, 1 for earthsight stl, 2 for earthsight mtl. For metrics.
    rcv_data = {}
    
    def __init__(self, node: Node, scheduler: EarthSightScheduler, mode: int) -> None:
        """Initialise the EarthSight ground station decorator.

        Wraps a base ``Node`` (typically a ground-station node) with
        EarthSight-specific receive and scheduling behaviour.  The
        station can receive packets from satellites and, upon a schedule
        request, use the provided *scheduler* to compute and return an
        observation schedule.

        Args:
            node (Node): The underlying ground-station node to decorate.
            scheduler: Scheduler instance whose ``schedule`` method will
                be called to produce observation schedules for satellites.
        """
        super().__init__(node)
        self.upload_bandwidth = 1000000
        self.scheduler = scheduler
        EarthSightGroundStation.mode = mode

    def get_upload_bandwidth(self) -> int:
        """
        Returns the upload bandwidth
        """
        return self.upload_bandwidth

    def has_data_to_transmit(self) -> bool:
        """Check whether the ground station has packets queued for uplink.

        Returns:
            bool: True if the transmit packet queue is non-empty,
            False otherwise.
        """
        return len(self.transmitPacketQueue) > 0
    
    def make_schedule(self, satellite, time_start, length) -> None:
        """
        Builds a schedule and sends it to the satellite
        """
        return self.scheduler.schedule(satellite, time_start, length)

    def receive_packet(self, pck: 'Packet') -> None:
        """Process an incoming packet from a satellite.

        Handles two recognised packet types:

        * **"schedule request"** -- Invokes the scheduler to build an
          observation schedule for the requesting satellite and enqueues
          the resulting schedule packet for uplink transmission.
        * **"image"** -- Extracts the downlinked image, computes the
          delivery delay (difference between the current simulation time
          and the image's capture time) as well as the unavoidable delay
          (difference between the current simulation time and the image's
          earliest possible transmit time), and records both metrics in
          the class-level ``rcv_data`` and ``delays`` dictionaries,
          keyed by satellite and priority level.

        Any packet whose descriptor is not a string raises a
        ``ValueError``.  Packets with an unrecognised string descriptor
        are forwarded to the parent ``ReceiveGS.receive_packet``
        implementation.

        Args:
            pck (Packet): The packet received from a satellite.  Its
                ``descriptor`` attribute determines the handling path.

        Raises:
            ValueError: If ``pck.descriptor`` is not a string.
        """
        if type(pck.descriptor) != str:
            print(pck, "Unknown packet type")
            print(pck.descriptor)
            log.Log("Unknown packet type", pck, "Unknown packet type")
            raise ValueError("Unknown packet type")
        elif pck.descriptor.startswith("schedule request"):
            schedule = self.make_schedule(satellite=pck.relevantNode, time_start=pck.relevantData[0], length=60*60*6)
            pkt = Packet(relevantData=schedule, relevantNode=pck.relevantNode, descriptor="schedule") # relevant node is the target node
            self.transmitPacketQueue.appendleft(pkt)
        elif pck.descriptor == "image":
            sat = pck.relevantNode
            if sat not in EarthSightGroundStation.rcv_data:
                EarthSightGroundStation.rcv_data[sat] = {
                    **{i: 0 for i in range(-1, 11)},
                    **{'count_{}'.format(i): 0 for i in range(-1, 11)},
                    **{'total_delay_{}'.format(i): 0 for i in range(-1, 11)}
                }

            if type(pck.relevantData) != Image:
                image = pck.relevantData[0]
            else:
                image : Image = pck.relevantData

            collection_time = image.time
            current_time = get_logging_time()
            total_delay = Time.difference_in_seconds(current_time, collection_time)
            unavoidable_transmit_delay = Time.difference_in_seconds(current_time, image.earliest_possible_transmit_time)
            avoidable_delay = total_delay - unavoidable_transmit_delay
            ground_truth = image.descriptor            
            EarthSightGroundStation.rcv_data[sat][ground_truth] += avoidable_delay
            EarthSightGroundStation.rcv_data[sat]['total_delay_' + str(ground_truth)] += total_delay
            EarthSightGroundStation.rcv_data[sat]['count_' + str(ground_truth)] += 1
        else:
            print(pck, "Unknown packet type")
            super().receive_packet(pck)

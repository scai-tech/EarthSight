import json
import os
import statistics
import numpy as np
from typing import TYPE_CHECKING
from src.station import Station
from src.utils import Location
from src.packet import Packet
from src.log import Log
from src.nodeDecorator import NodeDecorator
import src.const as const
import src.utils as utils

if TYPE_CHECKING:
    from src.node import Node

class ReceiveGS(NodeDecorator):
    """
    Receive-only ground station decorator.

    Wraps a Node (typically a Station) and configures it so that
    ``transmitAble`` is False and ``receiveAble`` is True.  Incoming
    packets are logged and, when ``INCLUDE_UNIVERSAL_DATA_CENTER`` is
    enabled, queued for conversion into data objects.  ACK packets are
    handled via the underlying node's ``receive_ack`` method.
    """
    mode = 0
    def __init__(self, node: 'Node') -> None:
        """Initialize the receive-only ground station decorator.

        Wraps the given node and sets transmission/reception flags so
        that the station can only receive satellite data and never
        transmit on the space link.

        Args:
            node: The Node (usually a Station) to decorate as a
                receive-only ground station.
        """
        super().__init__(node)
        ##self._node is a station object, so set transmit
        self.transmitAble = False
        self.receiveAble = True
        self.waitForAck = False
        self.sendAcks = False
        self.groundTransmitAble = True
        self.groundReceiveAble = True

    #@profile
    def receive_packet(self, pck: 'Packet') -> None:
        """Process an incoming packet at this receive-only ground station.

        If the packet is an ACK, it is forwarded to the underlying
        node's ``receive_ack`` handler.  If ``sendAcks`` is enabled, an
        ACK is generated for the sender.  The packet is always logged;
        when ``INCLUDE_UNIVERSAL_DATA_CENTER`` is enabled the packet is
        also placed in the receive queue for later conversion.

        Args:
            pck: The Packet received by this ground station.
        """
        if "ack" in pck.descriptor:
            self.receive_ack(pck)
        elif self.sendAcks:
            ##this assumes that once this get's here, the packet is done
            #self.receivePacketQueue.appendleft(pck)
            self.generate_ack(pck)
        Log("Iot Received packet:", pck, self)
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.receivePacketQueue.appendleft(pck)
    def load_data(self, timeStep: float) -> None:
        """Process received data for the current time step.

        Since this is a receive-only station it does not generate new
        data.  When ``INCLUDE_UNIVERSAL_DATA_CENTER`` is enabled, any
        packets in the receive buffer are converted into data objects.

        Args:
            timeStep: The current simulation time step in seconds.
        """
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.convert_receive_buffer_to_data_objects()
        ##Process data objects

    def load_packet_buffer(self) -> None:
        """Convert pending data objects into transmit-buffer packets.

        Normally a no-op for a receive-only station.  When
        ``INCLUDE_UNIVERSAL_DATA_CENTER`` is enabled, data objects are
        converted into the transmit buffer for relay to the data center.
        """
        if const.INCLUDE_UNIVERSAL_DATA_CENTER:
            self.convert_data_objects_to_transmit_buffer()

    @classmethod
    def get_transmission_overhead(cls) -> int:
        """Return the transmission delay for this ground station. Varies based on 
        processing mode and learning type for the i'th ground station to account
        for overhead. This is *added* to the observed delay to get the total delay 
        for a packet received at this ground station.
        """
        code = cls.mode  * 1.5 if cls.mode == 1 else cls.mode
        return 19*(code + 1)
        # return 22 * code + 19 * 2 #2Ghz 19ms...

    def get_upload_bandwidth(self) -> None:
        """Return the upload bandwidth for this ground station.

        Returns:
            int: A fixed upload bandwidth value of 5000 (bits per second).
        """
        return 5000

def assess_gs_logs(items, logs_dir: str, config: dict) -> None:
    """Assess the logs of receive-only ground stations to evaluate performance.

    This function iterates through the log files in the specified directory,
    extracts relevant performance metrics (e.g., packet reception rates, delays),
    and prints a summary of the results.

    Args:
        logs_dir: The directory containing the log files for receive-only ground stations.
    """
    delay_by_priority = {i: [] for i in range(-1, 11)}
    count_by_priority = {i: 0 for i in range(-1, 11)}

    for _, data in items:
        sat_hi_delay = 0
        sat_hi_count = 0
        sat_lo_delay = 0
        sat_lo_count = 0

        for i in range(-1, 11):
            count = data.get(f'count_{i}', 0)
            avoidable = data.get(i, 0)
            if count > 0:
                delay_by_priority[i].append(utils.correct_and_format(avoidable / count, config, dt=60)) # handle unit conversions, negative values, etc. for corrected delay
                count_by_priority[i] += count
                if i > 5:
                    sat_hi_delay += avoidable
                    sat_hi_count += count
                else:
                    sat_lo_delay += avoidable
                    sat_lo_count += count

    print("=====================================")
    print("\nAvoidable Delay by Priority Level [Minutes]:")
    for i in range(0, 11):
        delays = delay_by_priority[i]
        if delays:
            mean_d = statistics.mean(delays)
            std_d = statistics.stdev(delays) if len(delays) > 1 else 0.0
            p90 = np.percentile(delays, 90)
            p99 = np.percentile(delays, 99)
            print(f"  Priority {i}: Mean = {mean_d:.2f}, Std Dev = {std_d:.2f}, 90th = {p90:.2f}, 99th = {p99:.2f}")

    # Weighted average P90 across priorities > 5
    total_weighted_p90 = 0
    total_weighted_p10 = 0
    total_count = 0
    for i in range(6, 11):
        p90 = np.percentile(delay_by_priority[i], 90) if delay_by_priority[i] else 0
        p50 = np.percentile(delay_by_priority[i], 50) if delay_by_priority[i] else 0
        count = count_by_priority[i]
        total_weighted_p90 += (p90 * count)
        total_weighted_p10 += (p50 * count)
        total_count += count
    weighted_avg_p90 = total_weighted_p90 / total_count if total_count > 0 else 0
    print(f"\nWeighted Average P90 Avoidable Delay (Priority > 5) [Minutes]: {weighted_avg_p90:.2f}")
    # weighted average corrected p90
    
    # Save latency summary to JSON
    summary = {
        "config": {
            "mode": config["mode"],
            "scenario": config["scenario"],
            "learning": config["learning"],
            "hardware": config["hardware"],
        },
        "avoidable_delay_by_priority": {
            str(i): {
                "mean": statistics.mean(delay_by_priority[i]),
                "std": statistics.stdev(delay_by_priority[i]) if len(delay_by_priority[i]) > 1 else 0.0,
                "p50": float(np.percentile(delay_by_priority[i], 50)),
                "p90": float(np.percentile(delay_by_priority[i], 90)),
                "p99": float(np.percentile(delay_by_priority[i], 99)),
                "count": count_by_priority[i],
            } for i in range(0, 11) if delay_by_priority[i]
        },
        "weighted_avg_p90_priority_gt_5": weighted_avg_p90

    }
    # Make sure no one else is editing this file
    with open(os.path.join(logs_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

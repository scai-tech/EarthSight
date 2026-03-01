"""Simulation constants and configuration parameters.

This module centralizes every tunable knob for the EarthSight satellite
simulation, including timing resolution, packet sizes, link-budget
parameters, SNR mechanisms, alpha-tuning thresholds, data-collection
frequency, pruning limits, and logging/output paths.  Other modules
import ``src.const`` and reference these values at runtime; changing a
value here affects the entire simulation.
"""

from enum import Enum
import math

##Overall settings:
DEBUG = False #Whether or not to print debug statements
INCLUDE_POWER_CALCULATIONS = True #Whether to include power calculations in the simulation
INCLUDE_UNIVERSAL_DATA_CENTER = False #Whether to use a universal data center or not
##TRANSMISSION DETAILS:
SEND_ACKS = False #Whether to store info until ack or just send and delete the info

##Packet info:
PACKET_SIZE = 2400000000 #bits #Size of a packet excluding preamble
ACK_SIZE = 0 #bits #Size of an ACK packet
PREAMBLE_SIZE = 0 #bits #Size of the preamble
DATA_SIZE = 2400000000 #bits #Size of each data object
ONLY_CONVERT_ONE_DATA_OBJECT = True #Whether to only convert one data object or not

##Availability map settings:
MINIMUM_VISIBLE_ANGLE = 15 # minimum angle in degrees above horizon for gs to see sat

##For debugging:
FIXED_SATELLITE_POSITION = False #Whether to fix the satellite position or not

##Details for link calculations
INCLUDE_WEATHER_CALCULATIONS = False #Whether to include weather in the link quality calculations
SNR_SCALING = 53 #The hardware offset of the ground station's antenna, in dB
FREQUENCY = 8e9 #The frequency of the Sat, in Hz
BANDWIDTH = 76.79e6 #The bandwidth of the Sat, in Hz
ALLOWED_BITS_WRONG = 2 #The number of bits that can be wrong in a packet before it is considered corrupted

class SNRMechanism(Enum):
    """Enumeration of supported signal-to-noise ratio calculation methods.

    Members:
        lora: LoRa-style SNR calculation.
        greater_than17: Accept link if SNR exceeds 17 dB.
        bill: Custom SNR model (named after its author).
        none: Bypass SNR filtering entirely.
    """
    lora = 1
    greater_than17 = 2
    bill = 3
    none=4


# Logging details:
# whether to include uplink calculations in the log
INCLUDE_UPLINK_CALCULATIONS = True
# This will be reset at the beginning of the simulation and then continually updated
LOGGING_FILE = "logs/base_expect_empty_log"

# Filter configs:
# This is the file that contains the filter config
SNR_UPLINK_MECHANISM = SNRMechanism.none
SNR_DOWNLINK_MECHANISM = SNRMechanism.bill

p1 = .2
p2 = 0
ALPHA = 1

#If alpha is being tuned in our algorithm, these are the values it will be tuned between:
INITIAL_ALPHA = 1
ALPHA_HIGHER_THRESHOLD = .26
ALPHA_LOWER_THRESHOLD = .3
ALPHA_DOWN_THRESHOLD = .3
ALPHA_INCREASE = .005

#let's do 10 a day - 86400/10 = 8640
#DATA_COLLECTION_FREQUENCY = 8640
DATA_COLLECTION_FREQUENCY = 1
#DATA_COLLECTION_FREQUENCY = 60*60*3 #once every 3 hours
#DATA_COLLECTION_FREQUENCY = 8640 ##in seconds, how often a data object should be created. (This is actually random, so its when random num is less than this)
TIME_SINCE_LAST_CONTACT = 720 #actually determined by the simulation, this is just because we need to initialize it to something - (in seconds)
PROBABILITY_OF_DATA = 1 - math.exp(-1/DATA_COLLECTION_FREQUENCY*TIME_SINCE_LAST_CONTACT) ##probability of data object being created

##For pruning:
MIN_DISTANCE = 3000 #meters
MINIMUM_BER = 1e-4
JETSON_POWER_DRAW = 30 * 1000
CORAL_POWER_DRAW = 1 * 1000
CONTEXT_SWITCH_OVERHEAD = 20 # ms of overhead per filter execution for scheduling/context switching overhead, Same as STL mode=0

##Logging details:
INCLUDE_UPLINK_CALCULATIONS = True #Whether to include uplink calculations in the log
MAPS_PATH = ""

# At maximum how much bandwidth can sending priority images take up
MAX_PRIORITY_BANDWIDTH = 1

# Scale down the downlink bandwidth potentially for dgs only
DOWNLINK_BANDWIDTH_SCALING = 1
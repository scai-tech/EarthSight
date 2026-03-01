from typing import TYPE_CHECKING, List
from src.log import Log

if TYPE_CHECKING:
    from typing import Tuple
    from src.data import Data

import src.const as const

class Packet:
    """
    Packet type

    Attributes:
        size (int) - total size in bits
        infoSize (int) - size of the information in bits
        preambleSize (int) - size of the preamble in bits
        id (int) - number to keep track of which packet object is which
        descriptor (str) - string to describe what the data is
    Static Attributes:
        idCount (int) - keeps track of how many packets have been created
        packetIdToData (dict[Packet] = List[Data]]) - dictionary to keep track of which packet object is which
    """

    idCount = 0
    #@profile
    def __init__(self, relevantData, infoSize: int = const.PACKET_SIZE, preambleSize: int = const.PREAMBLE_SIZE, descriptor: str = "", relevantNode = None, generationTime = None) -> None:
        self.size = infoSize + preambleSize
        self.infoSize = infoSize
        self.preambleSize = preambleSize
        self.descriptor = descriptor
        self.image = None

        self.id = Packet.idCount
        Packet.idCount += 1

        if not isinstance(relevantData, List):
            relevantData = [relevantData]

        self.relevantData = relevantData
        self.generationTime = generationTime #already a string
        self.relevantNode = relevantNode
        
    def __str__(self) -> str:
        return "{{packetId: {}, packetSize: {}, packetDescriptor: {}, relevantNode: {}, generationTime: {}, relevantData: {}}}".format(self.id, self.size, self.descriptor, self.relevantNode, self.generationTime, self.relevantData)

    
class PriorityPacket(Packet):
    """
    Packet type for priority packets. This is used when the packet needs to be sent to a specific node.
    
    Attributes:
        priority (int) - the priority of the packet (1-10)
    """
    def __init__(self, relevantData=None, infoSize: int = const.PACKET_SIZE, preambleSize: int = const.PREAMBLE_SIZE, descriptor: str = "", relevantNode = None, generationTime = None, priority: int = 1) -> None:
        super().__init__(relevantData, infoSize, preambleSize, descriptor, relevantNode, generationTime)
        self.priority = priority

    def __lt__(self, other: 'PriorityPacket') -> bool:
        """
        Compares two packets based on their priority
        """
        if not isinstance(other, PriorityPacket):
            return NotImplemented
        return self.priority > other.priority # higher priority is better than lower priority
    
    def __str__(self) -> str:
        return "{{packetId: {}, packetSize: {}, packetDescriptor: {}, relevantNode: {}, generationTime: {}, relevantData: {}, priority = {} }}".format(self.id, self.size, self.descriptor, self.relevantNode, self.generationTime, self.relevantData, self.priority)
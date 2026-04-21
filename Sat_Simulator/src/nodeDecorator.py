from typing import TYPE_CHECKING
from src.node import Node

if TYPE_CHECKING:
    from src.packet import Packet

class NodeDecorator (Node):
    """
    Decorator-pattern base class for Node.

    Wraps an existing Node instance (stored as ``_node``) and delegates
    attribute access to it via custom ``__getattr__`` and ``__setattr__``
    implementations.  Subclasses such as ``EarthsightSatellite`` and
    ``ReceiveGS`` extend this class to layer domain-specific behavior
    (e.g., imaging, receive-only ground-station logic) on top of the
    underlying Node without modifying it directly.
    """
    def __init__(self, node: 'Node'):
        """Initialize the decorator by storing the wrapped Node.

        Args:
            node: The Node instance to decorate. All attribute accesses
                that are not overridden by the decorator subclass will
                be forwarded to this object.
        """
        self._node = node
        
    def get_node(self) -> 'Node':
        """Return the underlying wrapped Node instance."""
        return self._node

    def load_data(self, timeStep: 'float') -> None:
        """Load or generate data for the current time step.

        Default implementation is a no-op; subclasses override this to
        produce data (e.g., captured images) at each simulation step.

        Args:
            timeStep: The current simulation time step in seconds.
        """
        pass

    def load_packet_buffer(self, packet:'Packet' = None ) -> None:
        """Convert pending data objects into packets in the transmit buffer.

        Default implementation is a no-op; subclasses override this to
        packetize data for transmission.

        Args:
            packet: An optional pre-built Packet to enqueue directly.
        """
        pass

    def receive_packet(self, pck: 'Packet') -> None:
        """Handle an incoming packet.

        Default implementation is a no-op; subclasses override this to
        process received packets (e.g., store data, send ACKs).

        Args:
            pck: The Packet received by this node.
        """
        pass

    def __getattr__(self, name):
        """Delegate attribute look-ups to the wrapped ``_node``.

        Called only when normal attribute resolution fails.  If
        ``_node`` has already been set, the look-up is forwarded to it;
        otherwise ``None`` is implicitly returned so that ``__init__``
        can complete without recursion.

        Args:
            name: The attribute name being accessed.

        Returns:
            The attribute value from the wrapped node, or None if
            ``_node`` has not been set yet.
        """
        if "_node" in self.__dict__.keys():
            return getattr(self._node, name)
        else:
            pass

    def __setattr__(self, name, value):
        """Delegate attribute assignment to the wrapped ``_node``.

        If ``_node`` has already been stored in the instance dictionary,
        all subsequent attribute assignments are forwarded to the wrapped
        node so that its state is mutated directly.  Before ``_node`` is
        set (i.e., during ``__init__``), the value is stored on the
        decorator itself.

        Args:
            name: The attribute name being set.
            value: The value to assign.
        """
        if "_node" in self.__dict__.keys():
            setattr(self._node, name, value)
        else:
            self.__dict__[name] = value

    def __str__(self):
        """Return the string representation of the wrapped node."""
        return str(self._node)

    def __setstate__(self, d):
        """Restore instance state during unpickling.

        Directly updates ``__dict__`` so that the decorator's own
        attributes (including ``_node``) are restored without triggering
        the custom ``__setattr__`` delegation.

        Args:
            d: A dictionary representing the object's pickled state.
        """
        self.__dict__ = d

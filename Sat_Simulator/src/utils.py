from datetime import datetime, timedelta, timezone
from queue import PriorityQueue
from typing import Deque, Iterable, Tuple, List
from astropy.coordinates import EarthLocation, ITRS, AltAz, CIRS  # type: ignore
from astropy import units as astropyUnit
import numpy.linalg as la  # type: ignore
import numpy as np
from collections import deque
import src.const as const
import src.log as log
from src.filter import get_processing_coefficient

class TeeStream:
    """Write to both a file and the original stdout simultaneously."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush() # if this is on, output is streamed; otherwise, it will populate on crash or close, which may be desirable for performance reasons
    def flush(self):
        self.stream.flush()
        self.file.flush()
        
class FusedPriorityQueue():
    """Multi-tier transmit queue used by EarthsightSatellite.

    Items are dequeued in strict tier order: schedule requests (highest priority)
    are always returned first, followed by priority-scored images, then compute
    queue items, and finally low-priority images.

    The queue also tracks ``earliest_possible_transmit_time`` for delay analytics
    by annotating images that have not yet been timestamped at the moment they
    first become eligible for transmission.

    Attributes:
        schedule_request: Single-element list holding the current schedule
            request packet, or ``None`` if no schedule request is pending.
        priority_queue (PriorityQueue): Heap-ordered queue of high-priority
            image packets ranked by their priority score.
        compute_queue (Deque): FIFO queue of packets awaiting or resulting
            from on-board compute processing.
        low_priority_queue (PriorityQueue): Heap-ordered queue of low-priority
            image packets.
        untimed_images (set): Set of images whose
            ``earliest_possible_transmit_time`` has not yet been recorded.
    """

    def __init__(self, schedule_request, priority_queue: PriorityQueue, compute_queue : Deque, low_priority_queue : PriorityQueue, target):
        """Initialize the fused priority queue with its four internal tiers.

        Args:
            schedule_request: Single-element list whose first element is the
                current schedule request packet or ``None``.
            priority_queue (PriorityQueue): Queue for high-priority scored
                image packets.
            compute_queue (Deque): FIFO queue for compute-related packets.
            low_priority_queue (PriorityQueue): Queue for low-priority image
                packets.
        """
        self.schedule_request = schedule_request
        self.priority_queue = priority_queue
        self.compute_queue = compute_queue
        self.low_priority_queue = low_priority_queue
        self.untimed_images = set()
        self.target = target # Avoid circular imports without complex OOP

    def empty(self):
        """Return ``True`` if all four internal queues are empty."""
        return not self.schedule_request[0] and self.priority_queue.empty() and not self.compute_queue and self.low_priority_queue.empty()
    
    def pop(self):
        """Remove and return the highest-priority item across all tiers.

        Tier order (highest to lowest):
            1. Schedule request
            2. Priority queue (heap-ordered by score)
            3. Compute queue (FIFO)
            4. Low-priority queue (heap-ordered by score)

        Before returning items from tiers 2-4, any images in
        ``untimed_images`` are annotated with the current logging time as
        their ``earliest_possible_transmit_time``.

        Returns:
            The next packet to transmit, or ``None`` if all queues are empty.
        """
        if self.schedule_request[0]:
            pck = self.schedule_request[0]
            self.schedule_request[0] = None
            return pck
        
        # annotate the images
        best_case_transmit_time = log.get_logging_time().add(-self.target.get_transmission_overhead(), unit=60)
        for image in self.untimed_images:
            image.earliest_possible_transmit_time = best_case_transmit_time
        self.untimed_images.clear()

        if not self.priority_queue.empty():
            return self.priority_queue.get()
        elif self.compute_queue:
            return self.compute_queue.popleft()
        elif not self.low_priority_queue.empty():
            return self.low_priority_queue.get()
        else:
            return None
        
    def put_schedule(self, item):
        """Place a schedule request packet into the highest-priority slot.

        Args:
            item: The schedule request packet to enqueue. Overwrites any
                previously stored schedule request.
        """
        self.schedule_request[0] = item

    def put_priority(self, item):
        """Enqueue an item into the high-priority scored image queue.

        If the image carried by *item* has no ``earliest_possible_transmit_time``
        yet, it is added to ``untimed_images`` so it can be timestamped when
        the next pop occurs.

        Args:
            item: A packet whose ``relevantData[0]`` is the associated image.
        """
        if item.relevantData[0].earliest_possible_transmit_time is None:
            self.untimed_images.add(item.relevantData[0])
        self.priority_queue.put(item)

    def put_compute(self, item):
        """Enqueue an item into the compute (FIFO) queue.

        If the image carried by *item* has no ``earliest_possible_transmit_time``
        yet, it is added to ``untimed_images`` for later annotation.

        Args:
            item: A packet whose ``relevantData[0]`` is the associated image.
        """
        if item.relevantData[0].earliest_possible_transmit_time is None:
            self.untimed_images.add(item.relevantData[0])
        self.compute_queue.append(item)

    def put_low_priority(self, item):
        """Enqueue an item into the low-priority image queue.

        If the image carried by *item* has no ``earliest_possible_transmit_time``
        yet, it is added to ``untimed_images`` for later annotation.

        Args:
            item: A packet whose ``relevantData[0]`` is the associated image.
        """
        if item.relevantData[0].earliest_possible_transmit_time is None:
            self.untimed_images.add(item.relevantData[0])
        self.low_priority_queue.put(item)

    def has_schedule_request(self):
        """Return ``True`` if a schedule request packet is currently queued."""
        return self.schedule_request[0] is not None

    def __len__(self):
        """Return the total number of items across all four internal queues."""
        return (1 if self.schedule_request[0] else 0) + self.priority_queue.qsize() + len(self.compute_queue) + self.low_priority_queue.qsize()
        

class PriorityQueueWrapper(PriorityQueue):
    """Thin wrapper around Python's ``PriorityQueue`` providing deque-compatible methods.

    Adds ``appendleft``, ``popleft``, ``append``, ``pop``, ``peek``, and
    ``__len__`` so that a ``PriorityQueueWrapper`` can be used interchangeably
    with a ``collections.deque`` in code that relies on those interfaces while
    still maintaining heap ordering internally.
    """

    def peek(self):
        """Return the smallest item without removing it, or ``None`` if empty."""
        return self.queue[0] if not self.empty() else None

    def appendleft(self, item):
        """Insert *item* into the priority queue (deque-compatible alias for ``put``)."""
        self.put(item)

    def popleft(self):
        """Remove and return the smallest item (deque-compatible alias for ``get``)."""
        return self.get()

    def append(self, item):
        """Insert *item* into the priority queue (deque-compatible alias for ``put``)."""
        self.put(item)

    def pop(self):
        """Remove and return the smallest item (deque-compatible alias for ``get``)."""
        return self.get()

    def __len__(self):
        """Return the number of items currently in the queue."""
        return self.qsize()
    
class FusedQueue(Deque):
    """Multi-deque fusion with bandwidth allocation tracking.

    Wraps a list of underlying deques and presents a single deque-like
    interface. Items added via ``appendleft`` / ``extendleft`` enter the
    *first* (priority) sub-queue, while items added via ``append`` / ``extend``
    enter the *last* (non-priority) sub-queue.

    On ``pop``, the queue iterates through the sub-queues from first to last.
    If the fraction of bytes already sent from priority queues exceeds
    ``MAX_PRIORITY_BANDWIDTH`` (from ``const``), priority queues are skipped
    when the non-priority queue still has data, ensuring fair bandwidth
    allocation.

    An optional *callback* is invoked on every enqueue and dequeue operation.

    Attributes:
        queue_list (List[Deque]): Ordered list of underlying sub-queues.
            All queues except the last are considered priority queues.
        priority_bw_allocation (float): Maximum fraction of total sent bytes
            that may come from priority queues before throttling kicks in.
        sent_size (int): Cumulative size of all items popped so far.
        priority_sent_size (int): Cumulative size of items popped from
            priority (non-last) sub-queues.
        callback: Optional callable invoked as ``callback(item, operation)``
            on each enqueue or dequeue event.
    """

    def __init__(self, queue_list: List[Deque], priority_bw_allocation=None, callback=None):
        """Initialize the fused queue.

        Args:
            queue_list (List[Deque]): Ordered list of sub-queues. All queues
                except the last are treated as priority queues for bandwidth
                allocation purposes.
            priority_bw_allocation (float or None): Maximum fraction
                (0.0 -- 1.0) of total transmitted bytes that may come from
                priority queues. Defaults to ``const.MAX_PRIORITY_BANDWIDTH``
                when ``None``.
            callback: Optional callable invoked as ``callback(item, operation)``
                whenever an item is enqueued or dequeued.
        """
        self.queue_list = queue_list
        super().__init__()
        self.priority_bw_allocation = priority_bw_allocation if priority_bw_allocation is not None else const.MAX_PRIORITY_BANDWIDTH
        self.sent_size = 0
        self.priority_sent_size = 0
        self.callback = callback
    
    def empty(self):
        """Return ``True`` if the total length across all sub-queues is zero."""
        return len(self) == 0

    def pop(self):
        """Remove and return the next item, respecting bandwidth allocation.

        Iterates through sub-queues from first (highest priority) to last. If
        the ratio ``priority_sent_size / sent_size`` exceeds
        ``priority_bw_allocation`` and the last (non-priority) sub-queue has
        items, priority sub-queues are skipped to enforce fair bandwidth.

        The item's ``.size`` attribute is used to update cumulative byte
        counters. If a *callback* is registered it is called with the result
        and the string ``'pop'``.

        Returns:
            The next item according to priority and bandwidth rules, or
            ``None`` if all sub-queues are empty.
        """
        result = None
        for queue in self.queue_list:
            if self.priority_sent_size > 0 and self.priority_sent_size/self.sent_size > self.priority_bw_allocation and queue is not self.queue_list[-1] and len(self.queue_list[-1]) > 0:
                continue
            if len(queue) > 0:
                result = queue.pop()
                self.sent_size += result.size
                if queue is not self.queue_list[-1]:
                    self.priority_sent_size += result.size
                break
        if self.callback is not None:
            self.callback(result, 'pop')
        return result

    def appendleft(self, item):
        """Insert *item* at the left of the first (priority) sub-queue.

        Args:
            item: The item to enqueue as high priority.
        """
        if self.callback is not None:
            self.callback(item, 'appendleft')
        self.queue_list[0].appendleft(item)

    def extendleft(self, __iterable: Iterable) -> None:
        """Extend the first (priority) sub-queue from the left with *__iterable*.

        Args:
            __iterable (Iterable): Items to enqueue as high priority.
        """
        if self.callback is not None:
            self.callback(__iterable, 'extendleft')
        self.queue_list[0].extendleft(__iterable)

    def append(self, item):
        """Append *item* to the right of the last (non-priority) sub-queue.

        Args:
            item: The item to enqueue as non-priority.
        """
        if self.callback is not None:
            self.callback(item, 'append')
        self.queue_list[-1].append(item)

    def extend(self, __iterable: Iterable) -> None:
        """Extend the last (non-priority) sub-queue from the right with *__iterable*.

        Args:
            __iterable (Iterable): Items to enqueue as non-priority.
        """
        if self.callback is not None:
            self.callback(__iterable, 'extend')
        self.queue_list[-1].extend(__iterable)

    def __len__(self) -> int:
        """Return the total number of items across all sub-queues."""
        return sum([len(queue) for queue in self.queue_list])

    def __getitem__(self, __index: 'int'):
        """Return the item at the given logical index across all sub-queues.

        The index is resolved by walking through sub-queues in order,
        subtracting each queue's length until the target queue is found.

        Args:
            __index (int): Zero-based index into the virtual concatenation
                of all sub-queues.

        Returns:
            The item at the specified index.
        """
        i = 0
        while __index >= len(self.queue_list[i]):
            __index -= len(self.queue_list[i])
            i += 1
        return self.queue_list[i][__index]

    def __setitem__(self, __i: 'int', __x):
        """Set the item at the given logical index across all sub-queues.

        Index resolution works identically to ``__getitem__``.

        Args:
            __i (int): Zero-based index into the virtual concatenation of
                all sub-queues.
            __x: The value to store at that position.
        """
        i = 0
        while __i >= len(self.queue_list[i]):
            __i -= len(self.queue_list[i])
            i += 1
        self.queue_list[i][__i] = __x


class MyQueue(Deque, PriorityQueue):
    """Combined Deque and PriorityQueue interface with an optional callback.

    Inherits from both ``Deque`` and ``PriorityQueue`` and exposes the
    deque-style API (``appendleft``, ``extendleft``, ``append``, ``extend``,
    ``pop``) while using ``PriorityQueue``'s internal heap for ordering.

    An optional *callback* is invoked on every enqueue and dequeue operation,
    receiving the item (or iterable) and a string naming the operation.

    Random access via ``__getitem__`` and ``__setitem__`` is explicitly
    unsupported and will raise ``NotImplementedError``.

    Attributes:
        callback: Optional callable invoked as ``callback(item, operation)``
            on each enqueue or dequeue event.
    """

    def __init__(self, callback=None):
        """Initialize the queue.

        Args:
            callback: Optional callable invoked as ``callback(item, operation)``
                on every enqueue or dequeue event, where *operation* is one of
                ``'appendleft'``, ``'extendleft'``, ``'append'``, ``'extend'``,
                or ``'pop'``.
        """
        super(PriorityQueue, self).__init__()
        self.callback = callback

    def appendleft(self, item):
        """Insert *item* into the priority queue (deque-compatible alias).

        Args:
            item: The item to enqueue. Must be comparable for heap ordering.
        """
        if self.callback is not None:
            self.callback(item, 'appendleft')
        self.put(item)

    def extendleft(self, __iterable: Iterable) -> None:
        """Enqueue every element of *__iterable* via ``appendleft``.

        Args:
            __iterable (Iterable): Items to enqueue.
        """
        if self.callback is not None:
            self.callback(__iterable, 'extendleft')
        for item in __iterable:
            self.appendleft(item)

    def append(self, item):
        """Insert *item* into the priority queue (deque-compatible alias).

        Args:
            item: The item to enqueue. Must be comparable for heap ordering.
        """
        if self.callback is not None:
            self.callback(item, 'append')
        self.put(item)

    def extend(self, __iterable: Iterable) -> None:
        """Enqueue every element of *__iterable* via ``append``.

        Args:
            __iterable (Iterable): Items to enqueue.
        """
        if self.callback is not None:
            self.callback(__iterable, 'extend')
        for item in __iterable:
            self.append(item)

    def pop(self):
        """Remove and return the smallest item from the heap.

        Returns:
            The smallest item currently in the queue.
        """
        item = self.get()
        if self.callback is not None:
            self.callback(item, 'pop')
        return item

    def __len__(self) -> int:
        """Return the number of items currently in the queue."""
        return super(PriorityQueue, self).qsize()

    def __str__(self) -> str:
        """Return the string representation of the underlying deque."""
        return super(PriorityQueue, self).__str__()

    def __getitem__(self, __index):
        """Raise ``NotImplementedError``; random access is not supported."""
        raise NotImplementedError("This queue does not support random access")

    def __setitem__(self, __i, __x):
        """Raise ``NotImplementedError``; random access is not supported."""
        raise NotImplementedError("This queue does not support random access")

    def __repr__(self) -> str:
        """Return the repr of the underlying deque."""
        return super(PriorityQueue, self).__repr__()

    def empty(self) -> bool:
        """Return ``True`` if the queue contains no items."""
        return len(self) == 0


def Print(*args, logLevel: str = "debug") -> None:
    """Debug printing utility that respects the ``const.DEBUG`` flag.

    Concatenates all positional arguments into a single space-separated string
    and prints it according to the requested log level.

    Note:
        This function currently returns immediately (no-op) due to an early
        ``return`` statement. The logging logic below is inactive.

    Args:
        *args: Values to print. Each is converted to ``str`` and joined by
            spaces.
        logLevel (str): Controls output behaviour:
            - ``"debug"``: prints only when ``const.DEBUG`` is ``True``.
            - ``"always"``: prints unconditionally.
            - ``"error"``: prints in red using ANSI escape codes.

    Returns:
        None
    """
    return
    if (logLevel == "debug" and not const.DEBUG):
        return
    outStr = ""
    for arg in args:
        outStr += str(arg) + " "

    if (logLevel == "debug"):
        if (const.DEBUG):
            print(outStr)
    if (logLevel == "always"):
        print(outStr)
    if (logLevel == "error"):
        RED = '\033[31m'
        ENDC = '\033[0m'
        print(RED, outStr, ENDC)


class Time:
    """
    Wrapper from datetime class cause python datetime can be annoying at times.

    Attributes:
        time (datetime) - All times here are UTC!
    """

    def __init__(self) -> None:
        """Initialize a Time instance with a default date of 1900-01-01 00:00:00 UTC."""
        self.time = datetime(1900, 1, 1, 0, 0, 0)

    def copy(self) -> 'Time':
        """
        Returns another time object with same date
        """
        return Time().from_str(self.to_str())

    def from_str(self, time: str, format: str = "%Y-%m-%d %H:%M:%S") -> 'Time':
        """
        Gets time from specified format

        Arguments:
            time (str) - time in format specified by second input
            format (str) - format string, by default YYYY-MM-DD HH:MM:SS
        """
        self.time = datetime.strptime(time, format)
        self.time = self.time.replace(tzinfo=timezone.utc)
        return self

    def to_str(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Outputs time in format YYYY-MM-DD HH:MM:SS by default

        Arguments:
            format (str) - optional format string to change default
        """
        return self.time.strftime(format)

    def from_datetime(self, time: datetime) -> 'Time':
        """Set this Time from a Python ``datetime`` object.

        The timezone is forced to UTC regardless of the original tzinfo.

        Args:
            time (datetime): A Python datetime object.

        Returns:
            Time: This instance (*self*), for method chaining.
        """
        self.time = time
        self.time = self.time.replace(tzinfo=timezone.utc)
        return self
    
    def __repr__(self) -> str:
        """Return the string representation in ``YYYY-MM-DD HH:MM:SS`` format."""
        return self.to_str()

    def __str__(self) -> str:
        """Return the human-readable string in ``YYYY-MM-DD HH:MM:SS`` format."""
        return self.to_str()

    @staticmethod
    def difference_in_seconds(time1: 'Time', time2: 'Time') -> float:
        """
        Finds the difference between two time objects. Finds time1 - time2

        Arguments:
            time1 (Time) - time object
            time2 (Time) - time object
        """
        return (time1.time - time2.time).total_seconds()

    def to_datetime(self) -> datetime:
        """Convert to a standard Python ``datetime`` with UTC timezone.

        Returns:
            datetime: The underlying datetime object with ``tzinfo`` set to UTC.
        """
        self.time = self.time.replace(tzinfo=timezone.utc)
        return self.time

    def add_seconds(self, second: float) -> None:
        """
        Updates self by this number of seconds

        Arguments:
            second (float)
        """
        self.time = self.time + timedelta(seconds=second)
    
    def add(self, quantity: float, unit: int) -> None:
        """
        Updates self by this number of seconds

        Arguments:
            quantity (float) - quantity to add
            unit (int) - unit in seconds (e.g., 60 for minutes, 3600 for hours)

        """
        self.time = self.time + timedelta(seconds=quantity*unit)
        return self

    # Operators:
    def __lt__(self, other):
        """Return ``True`` if this time is strictly earlier than *other*."""
        return (self.time < other.time)

    def __le__(self, other):
        """Return ``True`` if this time is earlier than or equal to *other*."""
        return(self.time <= other.time)

    def __gt__(self, other):
        """Return ``True`` if this time is strictly later than *other*."""
        return(self.time > other.time)

    def __ge__(self, other):
        """Return ``True`` if this time is later than or equal to *other*."""
        return(self.time >= other.time)

    def __eq__(self, other):
        """Return ``True`` if this time is exactly equal to *other*."""
        return (self.time == other.time)

    def __ne__(self, other):
        """Return ``True`` if this time is not equal to *other*."""
        return not(self.__eq__(self, other))

    def __str__(self) -> str:
        """Return the human-readable string in ``YYYY-MM-DD HH:MM:SS`` format."""
        return self.to_str()  # + " (" + repr(self) + ")"

def get_mode_int(config):
    """
    Maps the scenario and learning configuration to an integer for use in model registry indexing.
    """

    if (config["mode"] == "earthsight" and config["learning"] == "mtl") or config["scenario"].startswith("n"):
        return 2 # EarthSightMTL or ND (Treat same for registry purposes since ND uses one backbone)
    if config["mode"] == "earthsight" and config["learning"] == "stl":
        return 1 # EarthSightSTL (Construct STL-based registry)
    else:
        return 0 # Serval (No need to construct registry)

def correct_and_format(time_s, config, dt = 60, altitude = 600) -> float:
    """
    Corrects time dilation for a given altitude and time.

    Arguments:
        dt (float) - time difference in seconds
        altitude (float) - altitude in km
        time_s - time in seconds, representing avoidable satellite component of delay.
    """

    time = time_s / dt # correct the units
    if config['hours'] < 24 or time < 10: return max(0, time) # no correction for the first 24 hours or small values to avoid instability
    a,c,b,d = get_processing_coefficient(config, altitude)
    return Location.sigmoid_error_correct(a, b, c, d, time)

class Location:
    """ITRF (International Terrestrial Reference Frame) Cartesian coordinate.

    Provides methods for converting to and from WGS84 latitude/longitude,
    computing Euclidean distances between locations, calculating altitude angles
    from a ground point to a satellite, and determining altitude/azimuth
    relative to an observer on the ground. Coordinate transforms rely on
    ``astropy.coordinates``.

    Attributes:
        x (float): X coordinate in meters (ITRF).
        y (float): Y coordinate in meters (ITRF).
        z (float): Z coordinate in meters (ITRF).
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Initialize a Location with ITRF Cartesian coordinates.

        Args:
            x (float): X coordinate in meters. Defaults to 0.
            y (float): Y coordinate in meters. Defaults to 0.
            z (float): Z coordinate in meters. Defaults to 0.
        """
        self.x = x
        self.y = y
        self.z = z

    def from_lat_long(self, lat: float, lon: float, elev: float = 0) -> 'Location':
        """
        Converts location from WGS84 lat, long, height to x, y, z in ITRF

        Arguments:
            lat (float) - latitude in degrees
            lon (float) - longitude in degrees
            elev (float)- elevation in meters relative to WGS84's ground.
        Returns:
            Location at point (self)
        """
        earthLoc = EarthLocation.from_geodetic(lon=lon, lat=lat,  height=elev, ellipsoid='WGS84').get_itrs(
        )  # Idk why they have this order, but it takes lon, lat. Also elev is distance above WGS reference, so like 0 is sea level

        self.x = float(earthLoc.x.value)
        self.y = float(earthLoc.y.value)
        self.z = float(earthLoc.z.value)
        return self

    def to_lat_long(self) -> 'Tuple[float, float, float]':
        """
        Returns lat, long, and elevation (WGS 84 output)

        Returns:
            Tuple (float, float, float) - lat, long, elevation in (deg, deg, m)

        """
        geoCentric = EarthLocation.from_geocentric(
            x=self.x, y=self.y, z=self.z, unit=astropyUnit.m)

        # round all of these to four decimal places
        lat = round(geoCentric.lat.value, 4)
        lon = round(geoCentric.lon.value, 4)
        elev = round(geoCentric.height.value, 4)
        return (lat, lon, elev)
    
    @staticmethod
    def sigmoid_error_correct(a, b, c, d, x):
        return d + (a / (1 + np.exp(c - b*x)))
    
    def to_coords(self):
        """Return WGS84 latitude and longitude as a ``(lat, lon)`` tuple in degrees.

        This is a convenience wrapper around ``to_lat_long`` that discards the
        elevation component.

        Returns:
            Tuple[float, float]: ``(latitude, longitude)`` in degrees.
        """
        lat, lon, lev = self.to_lat_long()
        return (lat, lon)

    @staticmethod
    def batch_to_coords(locations):
        """Convert a list of Location objects to (lat, lon) tuples in a single vectorized call.

        Args:
            locations: Iterable of Location objects with x, y, z in metres.

        Returns:
            List of (latitude, longitude) tuples in degrees, rounded to 4 decimals.
        """
        xs = np.array([loc.x for loc in locations])
        ys = np.array([loc.y for loc in locations])
        zs = np.array([loc.z for loc in locations])
        geo = EarthLocation.from_geocentric(x=xs, y=ys, z=zs, unit=astropyUnit.m)
        lats = np.round(geo.lat.value, 4)
        lons = np.round(geo.lon.value, 4)
        return list(zip(lats.tolist(), lons.tolist()))

    def to_alt_az(self, groundPoint: 'Location', time: 'Time') -> 'Tuple[float, float, float]':
        """
        Converts this location (self) to get the alt, az, and elevation relative to this point

        Arguments:
            groundPoint (Location) - location of ground point
            time (Time) - time when calculation needed
        Returns:
            tuple (float, float, float) - (alt, az, distance) in (degrees, degrees, and meters)
        Raise:
            ValueError - if input location and self are the same
        """
        if self == groundPoint:
            raise ValueError("Location of object and ground are the same")

        # based on https://docs.astropy.org/en/stable/coordinates/common_errors.html

        t = time.to_datetime()
        sat = EarthLocation.from_geocentric(
            x=self.x, y=self.y, z=self.z, unit=astropyUnit.m)
        ground = EarthLocation.from_geocentric(
            x=groundPoint.x, y=groundPoint.y, z=groundPoint.z, unit=astropyUnit.m)
        itrs_vec = sat.get_itrs().cartesian - ground.get_itrs().cartesian
        cirs_vec = ITRS(itrs_vec, obstime=t).transform_to(
            CIRS(obstime=t)).cartesian
        cirs_topo = CIRS(cirs_vec, obstime=t, location=ground)
        altAz = cirs_topo.transform_to(AltAz(obstime=t, location=ground))

        return (altAz.alt.value, altAz.az.value, altAz.distance.value)

    def calculate_altitude_angle(self, groundPoint: 'Location') -> float:
        """
        Calculates the altitude angle for self at the groundPoint

        Arguments:
            self (Location) - location of satellite
            groundPoint (Location) - point where you want the altitude at
        Returns:
            float - angle in degrees
        """
        # eqn 1 in https://arxiv.org/pdf/1611.02402.pdf
        rSat = np.array(self.to_tuple())
        rGround = np.array(groundPoint.to_tuple())
        delR = rSat - rGround
        r0Ground = rGround/np.linalg.norm(rGround, ord=2)
        val = np.dot(delR, r0Ground)/np.linalg.norm(delR, ord=2)
        return np.arcsin(val)*180/np.pi

    def get_radius(self) -> float:
        """
        Gets the height above Earth's center of mass in m
        """
        return float(la.norm(self.to_tuple(), ord=2))  # numpy norm

    def to_tuple(self) -> 'Tuple[float, float, float]':
        """Return the ITRF coordinates as a ``(x, y, z)`` tuple in meters."""
        return (self.x, self.y, self.z)

    def to_str(self) -> str:
        """Return a human-readable ``(x, y, z)`` string of the ITRF coordinates."""
        return "(" + str(self.x) + "," + str(self.y) + ", " + str(self.z) + ")"

    def get_distance(self, other: 'Location') -> float:
        """
        Return distance in m from this point to another

        Arguments:
            other (Location) - other object
        Returns:
            float - (distance in m)
        """
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2))

    @staticmethod
    def multiple_to_lat_long(locs: 'List[Location]') -> 'Tuple[List[float], List[float], List[float]]':
        """
        Returns lat, long, and elevation (WGS 84 output) of all of the locations. Faster than each one seperately
        Arguments:
            List[Location]
        Returns:
            Tuple (List[float], List[float], List[float]) - lat, long, elevation in (deg, deg, m)

        """
        xLst, yLst, zLst = zip(*[(pos.x, pos.y, pos.z) for pos in locs])
        geoCentric = EarthLocation.from_geocentric(
            x=xLst, y=yLst, z=zLst, unit=astropyUnit.m)

        lat = np.round(geoCentric.lat.value, 4).tolist()
        lon = np.round(geoCentric.lon.value, 4).tolist()
        elev = np.round(geoCentric.height.value, 4).tolist()

        return (lat, lon, elev)

    @staticmethod
    def multiple_from_lat_long(latLst: 'List[float]', lonLst: 'List[float]', elevLst: 'List[float]') -> 'List[Location]':
        """
        Returns a list of locations from lat, long, and elevation (WGS 84 input). Take a look in from_lat_long for more info

        Arguments:
            List[float] - latitudes (deg)
            List[float] - longitudes (deg)
            List[float] - elevations (m)
        Returns:
            List[Location] - locations
        """
        earthLoc = EarthLocation.from_geodetic(lon=lonLst, lat=latLst,  height=elevLst, ellipsoid='WGS84').get_itrs(
        )  # Idk why they have this order, but it takes lon, lat.Also elev is distance above WGS reference, so like 0 is sea level

        xLst = np.round(earthLoc.x.value, 4).tolist()
        yLst = np.round(earthLoc.y.value, 4).tolist()
        zLst = np.round(earthLoc.z.value, 4).tolist()

        return [Location(x, y, z) for x, y, z in zip(xLst, yLst, zLst)]

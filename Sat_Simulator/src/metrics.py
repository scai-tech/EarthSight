from src.image import Image
metr = None

class Metrics(object):
    """
    Singleton class that tracks simulation-wide performance statistics.

    Use the module-level ``metr()`` function (or the static ``Metrics.metr()``
    method) to obtain the single shared instance.  Tracked statistics
    include the number of images captured, high-priority images
    captured/computed/sent, and cumulative computation and transmission
    delays.
    """
    metr = None

    def metr():
        """Return the singleton Metrics instance, creating it if necessary.

        Returns:
            Metrics: The single shared Metrics instance for the simulation.
        """
        global metr
        if not metr:
            metr = Metrics()
        return metr

    def __init__(self) -> None:
        """Initialize all metric counters and delay accumulators to zero.

        Attributes:
            images_captured (int): Total number of images captured.
            pri_captured (int): Number of priority images captured.
            hipri_captured (int): Number of potentially high-priority images captured.
            hipri_computed (int): Number of high-priority images that have been
                computed/identified by on-board processing.
            hipri_sent (int): Number of high-priority images successfully sent.
            cmpt_delay (list): Two-element list ``[total_delay, count]`` for
                tracking average computation delay.
            transmit_delay (list): Two-element list ``[total_delay, count]`` for
                tracking average transmission delay.
        """
        self.images_captured = 0
        self.pri_captured = 0
        self.hipri_captured = 0
        self.hipri_computed = 0
        self.hipri_sent = 0
        self.cmpt_delay = [0,1E-32]
        self.transmit_delay = [0,1E-32]


    def print(self) -> None:
        """Print a human-readable summary of key simulation metrics to stdout.

        Displays counts for images captured, potentially high-priority
        images captured, actual high-priority images identified, and
        high-priority images sent.
        """
        print("Images Captured: ", self.images_captured)
        print("Potentially high-priority images Captured: ", self.hipri_captured)
        print("Actual high-priority images identified: ", self.hipri_computed)
        print("High Priority Images Sent: ", self.hipri_sent)



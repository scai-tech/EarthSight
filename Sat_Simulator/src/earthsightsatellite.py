from datetime import datetime
from src.image import Image, evaluate_image
from . import node
from . import log
from .utils import FusedPriorityQueue, Time
from src.metrics import Metrics
from src.packet import PriorityPacket, Packet
from src.schedule import Schedule
from collections import deque
from queue import PriorityQueue
from src.nodeDecorator import NodeDecorator
from src.earthsightgs import EarthSightGroundStation

class EarthsightSatellite(NodeDecorator):
    """Decorator around a satellite Node that adds EarthSight on-board processing.

    This class wraps a base satellite :class:`Node` via the Decorator pattern
    (:class:`NodeDecorator`) and layers EarthSight-specific behaviour on top:

    * **Image capture** -- captures images at a configurable rate
      (default 0.75 img/sec) and evaluates each against the currently
      scheduled filter formulas.
    * **Power management** -- tracks milliwatt-hour budgets for
      computation, camera, transmit, and receive sub-systems and
      throttles computation when stored energy is low.
    * **Transmit ordering** -- uses a :class:`FusedPriorityQueue` that
      drains queues in the following precedence:
      schedule requests > high-priority images > compute queue >
      low-priority images.
    * **Adaptive filtering** -- dynamically adjusts a filtering
      threshold (*alpha*) based on power headroom and image rejection
      rates.
    * **Schedule management** -- proactively requests observation
      schedules from ground stations with a 4-hour lookahead horizon,
      and avoids duplicate or premature re-requests.

    Class-level Attributes:
        power_consumptions (float): Cumulative power consumed (in milliwatt-
            hours) by **all** satellites combined.
        power_generation (float): Cumulative power generated (in milliwatt-
            hours) by **all** satellites combined.
    """

    power_consumptions = 0
    power_generation = 0
    total_compute_power = 0
    compute_s = 0

    def __init__(
        self,
        node: "node.Node",
        start_time: datetime = datetime(2025, 1, 1),
        mode = "earthsight",
        mtl_registry = None
    ) -> None:
        """Initialise an EarthSight-decorated satellite node.

        Wraps *node* with EarthSight on-board processing capabilities,
        setting up image capture parameters, power budgets, transmit
        queues, and adaptive-filtering state.

        Args:
            node: The underlying satellite :class:`~src.node.Node` to
                decorate.  All attributes not overridden here are
                transparently delegated to this object.
            start_time: Simulation start time used as the reference epoch
                for schedule management.  Defaults to 2025-01-01 00:00 UTC.
            mode: Processing mode string forwarded to
                :func:`~src.image.evaluate_image`.  Defaults to
                ``"earthsight"``.
            mtl_registry: Optional registry of multi-task-learning models
                used during image evaluation.  ``None`` disables
                registry-based evaluation.
        """

        super().__init__(node)
        self.mode = mode
        self.node = self.get_node()
        self.cache_size = 0
        self.computation_schedule = deque()
        self.computation_time_cache = 0

        self.mtl_registry = mtl_registry
        self.schedule_request = [None]
        self.computationQueue = deque()
        self.prioritizedQueue = PriorityQueue()
        self.deprioritizedQueue = PriorityQueue()

        self.transmitPacketQueue : FusedPriorityQueue = FusedPriorityQueue(
            schedule_request=self.schedule_request,
            priority_queue=self.prioritizedQueue,
            compute_queue=self.computationQueue,
            low_priority_queue=self.deprioritizedQueue,
            target=EarthSightGroundStation
        )

        self.normalPowerConsumption = 2.13 * 1000 # (ADACS) (units are in milliwatt hours)
        self.currentMWs = 100 * 1000
        self.target_power = 70 * 1000
        self.compute_power = 10 * 1000 # 10 W
        self.receivePowerConsumption = 1 * 1000 # 1 W
        self.transmitPowerConsumption = 5 * 1000 # 5 W
        self.camera_power = 4.5 * 1000 # 4.5 W
        self.powerGeneration = 4 * 1000 # 4.0 W
        self.maxMWs = 4000000
        
        self.start_time = Time().from_datetime(start_time)
        self.image_rate = .75 # images per second
        self.coords = self.node.position.to_coords()
        self.scheduled_until = None
        self.schedule_req_time = None
        self.alpha = 1.0
        self.r_rej = 0.6
        self.r_dep_num = 1.0
        self.r_dep_denom = 1.0 # avoid 0 division


    def dynamic_threshold_update(self, lambda_1 = 0.0, lambda_2 = 0.0): # tune the parameters here or elsewhere idc
        """Adjust the adaptive filtering threshold *alpha* in place.

        Alpha is nudged each time-step by two competing pressures:

        1. **Power headroom** (``lambda_1``): when stored energy exceeds
           the target, alpha increases (more permissive filtering); when
           energy is below target, alpha decreases (stricter filtering).
        2. **Deprioritisation rate** (``lambda_2``): when the fraction of
           deprioritised images exceeds the target rejection rate
           ``r_rej``, alpha decreases to reduce unnecessary work.

        The resulting alpha is clamped to [0, 1].

        Args:
            lambda_1: Gain for the power-headroom term.  Higher values
                make alpha more sensitive to energy reserves.
            lambda_2: Gain for the rejection-rate term.  Higher values
                make alpha more sensitive to the deprioritisation ratio.
        """
        r_power = self.currentMWs / self.target_power
        alpha = self.alpha + lambda_1 * (r_power - 1) - lambda_2 * (self.r_dep_num / self.r_dep_denom - self.r_rej)
        self.alpha = max(0, min(1, alpha))

    def populate_cache(self, timeStep: float) -> int:
        """Capture images for this time-step and route them to the appropriate transmit queues.

        For each captured image the method:

        1. Deducts camera power from the current energy budget.
        2. Updates the adaptive filtering threshold via
           :meth:`dynamic_threshold_update`.
        3. If a scheduled filter formula is available, evaluates the
           image against it (via :func:`~src.image.evaluate_image`) and
           records the computation time.
        4. Routes the image into the :attr:`transmitPacketQueue`:
           - Images that required computation go to the **compute queue**.
           - All other images go to the **low-priority queue** (with a
             descriptor distinguishing unscheduled from scheduled-but-
             not-computed images).
        5. Tracks deprioritisation counts for the adaptive-threshold
           feedback loop.

        Args:
            timeStep: Duration of the current simulation step in seconds.

        Returns:
            The number of images captured during this time-step.
        """
        coords = self.node.position.to_coords()
        images_captured = int(timeStep * self.image_rate)
        self.currentMWs -= self.camera_power * timeStep
        EarthsightSatellite.power_consumptions += self.camera_power * timeStep
        Metrics.metr().images_captured += images_captured
        collection_time = log.get_logging_time()

        self.dynamic_threshold_update()

        log.Log("node", self, {"computation schedule length": len(self.computation_schedule), "images_captured": images_captured, "time": collection_time})

        for _ in range(images_captured):
            image = Image(10, time=collection_time, coord=coords, name="Target Image")

            if self.computation_schedule:
                Metrics.metr().hipri_captured += 1
                tasks = self.computation_schedule.popleft()
                image.score, image.compute_time, image.descriptor = evaluate_image(formula=tasks.items[0], mode=self.mode, registry=self.mtl_registry)
        
            if image.descriptor > 1:
                Metrics.metr().hipri_sent += 1
            else:
                self.r_dep_num += 1
            
            self.r_dep_denom += 1

            self.cache_size += image.size
            if image.compute_time > 0:
                Metrics.metr().hipri_computed += 1
                self.transmitPacketQueue.put_compute(Packet(relevantData=image, relevantNode=self.node, descriptor="image"))
                log.Log("node", self, {"image": image.id, "score": image.score, "compute_time": image.compute_time})
            else:
                image.descriptor = 0 if self.computation_schedule else -1
                self.transmitPacketQueue.put_low_priority(PriorityPacket(priority=image.descriptor + 2, relevantData=image, descriptor="image", relevantNode=self.node))
        
        return images_captured
            
    def get_cache_size(self) -> int:
        """Return the cumulative size (in arbitrary units) of all images captured so far.

        Returns:
            The total size of images currently held in the on-board cache.
        """
        return self.cache_size

    def do_computation(self) -> None:
        """Drain the compute queue, scoring images until the computation-time budget is exhausted.

        Each image's pre-recorded ``compute_time`` is subtracted from
        :attr:`computation_time_cache`.  Processing stops when the cache
        is depleted or the queue is empty.

        After scoring, images are routed into the transmit pipeline:

        * **Positive score** -- placed in the high-priority transmit
          queue (ordered by score).
        * **Non-positive score** -- demoted to the low-priority transmit
          queue.
        """
        while (len(self.computationQueue) > 0):
            if self.computation_time_cache <= 0:
                break
            
            pkt : Packet = self.computationQueue.popleft()
            image : Image = pkt.relevantData[0]
            self.computation_time_cache -= image.compute_time

            self.images_processed_per_timestep += 1
            
            if image.score > 0:
                self.transmitPacketQueue.put_priority(PriorityPacket(priority=image.score, relevantData=image, descriptor="image", relevantNode=self.node))
            else:
                self.transmitPacketQueue.put_low_priority(PriorityPacket(priority=5, relevantData=image, descriptor="image", relevantNode=self.node))

    def percent_of_memory_filled(self) -> float:
        """Return the fraction of on-board memory currently occupied.

        The denominator is a fixed capacity of 10,000,000 queue slots.

        Returns:
            A float in [0, 1+] representing the proportion of memory
            used.  Values above 1.0 indicate the queue has exceeded
            nominal capacity.
        """
        return len(self.dataQueue) / 10000000

    def receive_packet(self, pck: Packet) -> None:
        """Process an incoming packet containing a new observation schedule.

        The schedule's task entries are appended to the on-board
        :attr:`computation_schedule` deque, the ``scheduled_until``
        horizon is updated, and the target rejection rate ``r_rej`` is
        recalculated from the schedule's proportion of tasks that
        require on-board computation.

        Args:
            pck: A :class:`~src.packet.Packet` whose ``relevantData``
                contains a :class:`~src.schedule.Schedule` instance as
                its first element.
        """
        schedule: Schedule = pck.relevantData[0]
        self.scheduled_until : Time = schedule.end
        self.computation_schedule.extend(schedule.toQueue())
        log.Log("Node {}: Scheduled with length {}".format(self.id, len(self.computation_schedule)))
        self.r_rej = 1 - schedule.percentage_requiring_compute()

    def should_request_schedule(self, timestep: float) -> bool:
        """Determine whether this satellite should request a new schedule from a ground station.

        A request is suppressed when any of the following conditions hold:

        1. The current schedule already covers at least 4 hours into the
           future.
        2. A schedule request packet is already queued in the transmit
           pipeline (its timestamp is refreshed to stay current).
        3. Fewer than 10 minutes have elapsed since the last request was
           sent, giving the ground station time to respond.

        Args:
            timestep: Duration of the current simulation step in seconds
                (currently unused but accepted for interface consistency).

        Returns:
            ``True`` if a new schedule request should be enqueued;
            ``False`` otherwise.
        """
        schedule_horizon = log.get_logging_time().copy()
        schedule_horizon.add_seconds(60*60*4) # 6 hours
        if self.scheduled_until and self.scheduled_until >= schedule_horizon:
            # schedule is valid for 6 hours already
            return False
        
        if self.transmitPacketQueue.has_schedule_request():
            # already requested a schedule
            self.transmitPacketQueue.schedule_request[0].relevantData = [log.get_logging_time()]
            return False
            
        
        if self.schedule_req_time and Time.difference_in_seconds(log.get_logging_time(), self.schedule_req_time) <= 60*10:
            # give gs 10 minutes to respond before resending
            return False

        return True
        

    def load_data(self, timeStep: float) -> None:
        """Execute the satellite's main per-time-step processing loop.

        This is the top-level entry point called by the simulation each
        tick.  It orchestrates the following actions in order:

        1. **Schedule request** -- if needed, enqueues a schedule
           request packet via :meth:`should_request_schedule`.
        2. **Image capture** -- calls :meth:`populate_cache` to capture
           and initially classify images.
        3. **Power generation** -- credits the satellite's solar panel
           output for this time-step.
        4. **Computation budget** -- if the computation-time cache is
           low and sufficient energy remains, allocates additional
           compute time in 15-second increments (up to the full
           time-step), deducting the corresponding power.
        5. **Compute queue processing** -- calls :meth:`do_computation`
           to score queued images.
        6. **Analytics snapshot** -- appends a diagnostic tuple for this
           time-step to :attr:`analytics`.

        Args:
            timeStep: Duration of the current simulation step in seconds.
        """
        if self.should_request_schedule(timeStep):
            time = log.get_logging_time() if self.scheduled_until is None else self.scheduled_until
            self.transmitPacketQueue.put_schedule(PriorityPacket(priority=11, relevantData=time, descriptor="schedule request", relevantNode=self))
            self.schedule_req_time = log.get_logging_time()

        self.images_processed_per_timestep = 0
        self.time = log.loggingCurrentTime
        ims_captured = self.populate_cache(timeStep)
        EarthsightSatellite.power_generation += self.powerGeneration * timeStep
        self.currentMWs += self.powerGeneration * timeStep
        # Do computation
        if self.computation_time_cache < timeStep:
            if self.currentMWs > self.compute_power * timeStep / 4: # load in 15 second increments
                # timestep groups afforded by the power
                steps = min(int(self.currentMWs / (self.compute_power * timeStep)), 4)
                self.computation_time_cache += timeStep * steps / 4
                self.currentMWs -= self.compute_power * timeStep * steps / 4
                EarthsightSatellite.power_consumptions += self.compute_power * timeStep * steps / 4
                EarthsightSatellite.total_compute_power += self.compute_power * timeStep * steps / 4
                EarthsightSatellite.compute_s += timeStep * steps / 4
                # print("Computing: ", prevmw, self.computation_time_cache, self.currentMWs, timeStep, self.compute_power)
            else:
                pass
                # print("Not enough power to compute: ", self.currentMWs, timeStep, self.compute_power)
        log.Log(
            "Computation time cache", self, {
                "time_cache": self.computation_time_cache}
        )
        self.do_computation()

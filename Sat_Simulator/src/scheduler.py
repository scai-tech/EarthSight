import pickle
from src.satellite import Satellite
from src.schedule import Schedule, ScheduleItem
from src.utils import Time, Location
from src.lookaheadsimulation import LookaheadSimulator
from src.query import SpatialQueryEngine, Query
from src.workload import get_padding_query, get_padding_probability
import random
import os


class LookaheadRuntime(object):
    """Manages a lookahead simulation that predicts future satellite-ground station contacts.

    Runs a lightweight forward simulation (up to 6 hours ahead) to estimate which
    priority-tier images each satellite will be able to downlink during upcoming
    ground station passes. The predicted transmission log is used by the scheduler
    to set minimum priority thresholds so that satellites only capture images they
    are likely to be able to transmit.

    Attributes:
        satellites: List of Satellite objects participating in the simulation.
        groundstations: List of ground station objects used for contact prediction.
        qe: A SpatialQueryEngine used by the inner LookaheadSimulator to resolve
            spatial queries during the forward simulation.
        current_time: The Time at which the lookahead window begins.
        lookahead_results: Dictionary mapping satellite IDs to lists of
            (time, schedule) tuples produced by the forward simulation, or None
            if no simulation has been run yet.
        sim: The underlying LookaheadSimulator instance, or None before the first run.
    """

    def __init__(self, satellites, groundstations, query_engine, current_time, cache_dir="cache") -> None:
        """Initialize the LookaheadRuntime.

        Args:
            satellites: List of Satellite objects to include in the lookahead
                simulation.
            groundstations: List of ground station objects for predicting
                downlink contacts.
            query_engine: A SpatialQueryEngine that the inner simulator uses to
                resolve spatial queries during the forward simulation.
            current_time: A Time object representing the start of the lookahead
                window.
            cache_dir: Directory path for storing lookahead result caches.
                Defaults to "cache".
        """
        self.satellites = satellites
        self.groundstations = groundstations
        self.qe = query_engine
        self.current_time = current_time
        self.lookahead_results = None
        self.sim : LookaheadSimulator = None
        self.cache_dir = cache_dir
    
    def refresh_results(self):
        """Run (or re-run) the full 6-hour lookahead simulation from scratch.

        Creates a new LookaheadSimulator spanning from ``current_time`` to
        ``current_time + 6 hours`` with a 60-second time step, executes it,
        and stores the resulting transmission log in ``lookahead_results``.

        If a previous set of results existed before the refresh, the new results
        are also serialized to ``log/results.pkl`` for persistence across runs.
        On the very first call (no prior results), the pickle step is skipped.
        """
        temp = self.lookahead_results is not None
        self.lookahead_results = None
        refresh_until : Time = self.current_time.copy()
        refresh_until.add_seconds(60 * 60 * 6)
        sim = LookaheadSimulator(60, self.current_time, refresh_until, self.satellites, self.groundstations, engine=self.qe)
        sim.run()
        self.lookahead_results = sim.transmission_log
        self.sim = sim

        # save the results to a file
        f = os.path.join(self.cache_dir, "results.pkl")
        os.makedirs(self.cache_dir, exist_ok=True)

        if temp:
            try:
                with open(f, "wb") as pk:
                    pickle.dump(self.lookahead_results, pk)
            except Exception:
                pass

    def extend_results(self, new_time):
        """Extend the lookahead simulation window to cover a later end time.

        If no results exist yet, a full refresh is performed first. Otherwise,
        the existing simulator's end time is advanced to ``new_time`` and the
        simulation is continued from where it left off. The updated transmission
        log replaces the stored ``lookahead_results``.

        Args:
            new_time: A Time object representing the desired new end of the
                lookahead window. If ``new_time`` is not later than
                ``current_time``, no additional simulation is performed.
        """
        if self.lookahead_results is None:
            self.refresh_results()

        # check if the new time is greater than the current time
        if new_time > self.current_time:
            self.sim.endTime = new_time
            self.sim.run()
            self.lookahead_results = self.sim.transmission_log

    def prune_past_results(self, cutoff_time, satellites=None):
        """Remove lookahead entries whose timestamps fall before a cutoff time.

        Iterates over each satellite's result list and removes entries whose
        associated time is earlier than ``cutoff_time``. Because result lists
        are ordered chronologically, the method scans from the front until it
        finds the first entry at or after the cutoff and removes everything
        before it.

        Args:
            cutoff_time: A Time object. All entries with a time strictly before
                this value are discarded.
            satellites: Optional list of Satellite objects whose results should
                be pruned. If None or empty, all satellites tracked by this
                runtime are pruned.
        """
        if not satellites:
            satellites = self.satellites

        for sat in satellites:
            bad_idx = []
            for i in range(len(self.lookahead_results[sat.id])):
                if self.lookahead_results[sat.id][i][1] < cutoff_time:
                    bad_idx.append(i)
                else:
                    break

            for i in sorted(bad_idx,reverse=True):
                self.lookahead_results[sat.id].pop(i)
            
    def get_results(self, sat, time):
        """Retrieve the next predicted downlink schedule for a satellite after a given time.

        Attempts to load cached results from ``log/results.pkl`` first. If no
        results are available in memory or on disk, a full ``refresh_results``
        call is triggered. The method then searches for the first entry whose
        time exceeds ``time`` by at least a 20-minute buffer.

        If the existing simulation window does not contain a suitable entry for
        the requested satellite, the window is extended in 1-hour increments
        (up to 6 hours past ``time``) until a matching entry is found.

        Args:
            sat: The Satellite object to look up results for.
            time: A Time object representing the current moment. Results are
                returned only if they occur more than 20 minutes after this
                time.

        Returns:
            A tuple ``(contact_time, schedule_dict)`` where ``contact_time``
            is the predicted Time of the next ground station contact and
            ``schedule_dict`` maps priority tiers to counts of images expected
            to be transmitted.

        Raises:
            Exception: If no suitable schedule entry is found for the satellite
                within the maximum lookahead window.
        """
        f = os.path.join(self.cache_dir, "results.pkl")

        try:
            with open(f, "rb") as pk:
                self.lookahead_results = pickle.load(pk)
        except Exception:
            pass


        if self.lookahead_results is None:
            self.refresh_results()

        # pickle the results
        

        buffer_time = time.copy()
        buffer_time.add_seconds(60 * 20) # 20 mins

        max_extend_time = time.copy()
        max_extend_time.add_seconds(60 * 60 * 6) # 6 hours
        while ((sat.id not in self.lookahead_results  \
                or len(self.lookahead_results[sat.id]) == 0 \
                or self.lookahead_results[sat.id][-1][0] <= buffer_time) 
                and self.sim.endTime < max_extend_time):
            
            self.sim.endTime.add_seconds(60 * 60)
            self.extend_results(self.sim.endTime)

            print("Extending results for satellite {} at time {}".format(sat.node.id, time))

        for t, sched in self.lookahead_results[sat.id]:
            if t > buffer_time:
                return t, sched

        print(self.lookahead_results)
        raise Exception("No schedule found for satellite at time {}".format(time))
            

class EarthSightScheduler(object):
    """Main scheduler that generates per-satellite image-capture schedules.

    For each scheduling window the scheduler:

    1. Computes the satellite's future ground track by propagating its orbit
       forward in 1.25-second steps.
    2. At each step, queries the SpatialQueryEngine to find areas of interest
       that intersect the satellite's footprint.
    3. Builds a DNF (Disjunctive Normal Form) formula for each image-capture
       slot from the filter categories of matching queries.
    4. Optionally uses LookaheadRuntime results to determine a minimum priority
       threshold so only images the satellite can realistically downlink are
       captured.
    5. Caches completed schedules to disk as pickle files to avoid redundant
       recomputation.

    Two operating modes are supported:

    * **EarthSight mode** (default): Uses the lookahead simulation to
      dynamically set the minimum capture priority.
    * **Serval mode** (``serval_mode=True``): Bypasses the lookahead and fixes
      the minimum priority at tier 2.

    Attributes:
        queries: List of Query objects defining areas of interest.
        satellites: List of Satellite objects in the constellation.
        stations: List of ground station objects for downlink planning.
        qe: A SpatialQueryEngine loaded with the provided queries.
        sim_time: The overall simulation Time reference.
        runtime: A LookaheadRuntime instance (created lazily on first use),
            or None in serval mode.
        serval_mode: Boolean flag selecting Serval-style fixed-priority
            scheduling when True.
    """

    def __init__(self, queries, satellites, stations, sim_time, limit_priority=True, cache_dir = "cache") -> None:
        """Initialize the EarthSightScheduler.

        Args:
            queries: List of Query objects that define geographic areas of
                interest and their associated priority tiers and filter
                categories.
            satellites: List of Satellite objects in the constellation to be
                scheduled.
            stations: List of ground station objects used for downlink contact
                prediction by the lookahead runtime.
            sim_time: A Time object representing the overall simulation clock
                reference.
            limit_priority: If True, use a fixed minimum priority of 2
            cache_dir: Directory path for storing schedule and lookahead
                caches. Defaults to "cache".
            workload_intensity: 0 or 1. When 1, empty schedule slots are
                padded with a combined monitoring query at the given
                padding_probability. When 0, no padding occurs.
            padding_probability: Probability of injecting a padding query
                into an empty schedule slot.
        """
        self.queries = queries
        self.satellites = satellites
        self.stations = stations
        self.qe = SpatialQueryEngine()
        self.qe.load_queries(queries)
        self.sim_time = sim_time
        self.runtime = None
        self.limit_priority = limit_priority
        self.cache_dir = cache_dir
        self.workload_intensity = get_padding_probability()
        self.padding_query = get_padding_query()
        

    def schedule(self, sat : Satellite, start, length) -> Schedule:
        """Generate an image-capture schedule for a single satellite over a time window.

        Propagates the satellite's orbit forward in 1.25-second steps, querying
        the SpatialQueryEngine at each position to find intersecting areas of
        interest. For every capture slot, a DNF formula is constructed from the
        matching queries' filter categories and priority tiers and wrapped in a
        ScheduleItem.

        In EarthSight mode, the lookahead runtime is consulted to determine the
        lowest priority tier that was successfully downlinked in the predicted
        future, and only queries at or above that tier are included. In Serval
        mode the minimum priority is fixed at 2. With a 20% probability, if no
        queries match a given position, a combined monitoring query from the
        reference scenarios is injected.

        Completed schedules are cached to disk under
        ``schedule_cache_serval_combined/`` keyed by satellite ID and start time.
        On subsequent calls with the same key the cached schedule is loaded
        directly, bypassing recomputation.

        Args:
            sat: The Satellite whose orbit is propagated and for which the
                schedule is built.
            start: A Time object marking the beginning of the scheduling window.
            length: Duration of the scheduling window in seconds.

        Returns:
            A Schedule object whose ``tasklist`` contains one ScheduleItem per
            capture slot (approximately 45 items per minute of window length).
        """

        endTime : Time = start.copy()
        endTime.add_seconds(length)
        schedule = Schedule(tasklist=[], startTime=start, endTime=endTime) # length in seconds
        length_in_minutes = length // 60
        image_count = 45*length_in_minutes
        current_time : Time = start

        if self.limit_priority:
            pri = 2
        else:
            if not self.runtime:
                self.runtime = LookaheadRuntime(self.satellites, self.stations, self.qe, current_time, cache_dir=self.cache_dir)

            time, sched = self.runtime.get_results(sat, current_time)
            pri = 1 # pri is the lowest priority such that an image of priority pri was received
            while sched[pri] == 0 and pri <= 10:
                pri += 1

        fname = os.path.join(self.cache_dir, "{}_{}_schedule.pkl".format(sat.node.id, str(start).replace(":", "-")))

        try:
            with open(fname, "rb") as pk:
                schedule.tasklist = pickle.load(pk)
            return schedule
        except Exception:
            pass  

        # Batch-compute all orbit positions and convert to lat/lon in one vectorized pass
        positions = sat.calculate_orbit_at_multiple_times(start.copy(), endTime, 1.25)
        locations = list(positions.values())
        all_coords = Location.batch_to_coords(locations)

        # Cache spatial query results by rounded coordinate to avoid redundant lookups
        query_cache = {}
        for coord in all_coords:
            cache_key = (round(coord[0], 1), round(coord[1], 1))
            if cache_key in query_cache:
                queries = query_cache[cache_key]
            else:
                queries = self.qe.get_queries_at_coord(coord, min_pri=pri, max_pri=10)
                query_cache[cache_key] = queries

            if not queries and random.random() < self.workload_intensity:
                queries = {self.padding_query}

            formula = [([(f, True) for f in f_seq], q.priority_tier) for q in queries for f_seq in q.filter_categories]

            schedule_item = ScheduleItem(items=[formula])
            schedule.add_task(schedule_item)

        # save schedule to file
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "wb") as f:
            pickle.dump(schedule.tasklist, f)
        return schedule

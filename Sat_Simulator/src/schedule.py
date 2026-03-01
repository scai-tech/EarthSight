from enum import Enum
import json
from collections import deque

class ScheduleItem:
    """
    A single entry in a satellite's computation schedule.

    Each ScheduleItem wraps a list of items (typically a DNF formula) that the
    satellite should evaluate against the next captured image at the corresponding
    time slot.

    Attributes:
        items (list): List of task payloads. For EarthSight, items[0] is a DNF formula
            as [(term, priority), ...] where term is [(filter_id, polarity), ...].
    """
    def __init__(self, items: list = []) -> None:
        self.items = items

class Schedule:
    """
    An ordered sequence of ScheduleItems covering a time window, sent from a
    ground station to a satellite to direct on-board image processing.

    The schedule maps 1:1 with captured images — each ScheduleItem corresponds to
    one image capture slot (at the satellite's image_rate). Items with non-empty
    formulas trigger on-board ML inference; empty items mean the image is low-priority.

    Attributes:
        tasklist (list[ScheduleItem]): Ordered list of schedule entries.
        start (Time): Start time of the schedule window.
        end (Time): End time of the schedule window.
    """
    def __init__(self, tasklist: list[ScheduleItem] = [], startTime=None, endTime=None) -> None:
        self.tasklist = tasklist
        self.start = startTime
        self.end = endTime

    def add_task(self, task: ScheduleItem):
        """Append a single ScheduleItem to the end of this schedule."""
        self.tasklist.append(task)

    def add_tasks(self, tasks: list[ScheduleItem]):
        """Append multiple ScheduleItems to this schedule."""
        self.tasklist.extend(tasks)

    def get_task(self, index: int) -> ScheduleItem:
        """Return the ScheduleItem at the given index."""
        return self.tasklist[index]

    def get_tasks(self) -> list[ScheduleItem]:
        """Return the full list of ScheduleItems."""
        return self.tasklist

    def naive_serialize(self) -> str:
        """Serialize the tasklist to a JSON string."""
        return str(json.dumps(self.tasklist))

    def contains_anything(self):
        """Return True if any ScheduleItem has a non-empty items list."""
        for task in self.tasklist:
            if len(task.items) > 0:
                return True

    def percentage_requiring_compute(self):
        """
        Return the fraction of schedule slots that require on-board computation
        (i.e., have a non-empty DNF formula).
        """
        total_tasks = len(self.tasklist)
        if total_tasks == 0:
            return 0
        compute_tasks = sum(len(task.items[0]) > 0 for task in self.tasklist)
        return compute_tasks / total_tasks

    def toQueue(self):
        """Convert the tasklist to a deque for sequential consumption by the satellite."""
        return deque(self.tasklist)

    @classmethod
    def naive_deserialize(cls, data: str) -> 'Schedule':
        """Deserialize a JSON string back into a Schedule object."""
        return Schedule(json.loads(data))
    


    

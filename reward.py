"""Shared evaluation metric only; no learning or simulator dependencies."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardParts:
    driving: float
    speed: float
    destination: float

    @property
    def total(self):
        return self.driving + self.speed + self.destination


def evaluation_reward(previous_progress_m, progress_m, speed_kmh, goal=False):
    """Exact supplied formula; speed is a magnitude in km/h, progress in meters."""
    values = (previous_progress_m, progress_m, speed_kmh)
    if not all(math.isfinite(v) for v in values) or speed_kmh < 0:
        raise ValueError('Expected finite progress and a nonnegative speed magnitude in km/h')
    return RewardParts(float(progress_m - previous_progress_m),
                       .1 * float(speed_kmh) / 80., 20. if goal else 0.)


class EpisodeReward:
    """Instantiate at each reset, after obtaining the actual initial progress."""

    def __init__(self, initial_progress_m):
        if not math.isfinite(initial_progress_m):
            raise ValueError('Initial progress must be finite')
        self.progress_m = initial_progress_m
        self.parts = RewardParts(0., 0., 0.)
        self.steps = 0
        self.closed = False

    def add(self, progress_m, speed_kmh, terminal_reason=''):
        if self.closed:
            raise ValueError('Episode ended; create a new accumulator after reset')
        if terminal_reason not in ('', 'goal', 'collision', 'offroad', 'stall_timeout', 'time_limit'):
            raise ValueError('Unknown terminal reason')
        parts = evaluation_reward(self.progress_m, progress_m, speed_kmh,
                                  terminal_reason == 'goal')
        self.parts = RewardParts(self.parts.driving + parts.driving,
                                 self.parts.speed + parts.speed,
                                 self.parts.destination + parts.destination)
        self.progress_m = progress_m
        self.steps += 1
        self.closed = bool(terminal_reason)
        return parts

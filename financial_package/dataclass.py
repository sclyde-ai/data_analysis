from dataclasses import dataclass
from typing import List

@dataclass
class ListAndStr():
    elem: List[str]
    def __post_init__(self):
        if isinstance(self.items, list):
            return
        if isinstance(self.elem, str):
            self.elem=[self.elem]
        else:
            raise ValueError("error")

@dataclass
class HourInDay():
    hour: int
    def __post_init__(self):
        if not isinstance(self.hour, int):
            raise ValueError("type is incorrect")
        if self.hour >= 24:
            raise ValueError("hour is below 24")

@dataclass        
class MinuteInHour():
    minute: int
    def __post_init__(self):
        if not isinstance(self.minute, int):
            raise ValueError("type is incorrect")
        if self.minute >= 60:
            raise ValueError("hour is below 24")
@dataclass
class CashTime():
    cache_days: int
    cache_hours: HourInDay
    cache_minutes: MinuteInHour
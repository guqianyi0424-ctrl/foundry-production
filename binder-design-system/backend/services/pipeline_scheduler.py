from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PipelineTaskScheduler:
    mpnn_slots: int = 1
    rf3_slots: int = 1

    def __post_init__(self):
        if self.mpnn_slots < 1:
            raise ValueError("mpnn_slots must be at least 1")
        if self.rf3_slots < 1:
            raise ValueError("rf3_slots must be at least 1")

    def run_mpnn_tasks(self, tasks: list[T], worker: Callable[[T], R]) -> list[R]:
        return self._run_limited(tasks, worker, self.mpnn_slots)

    def run_rf3_tasks(self, tasks: list[T], worker: Callable[[T], R]) -> list[R]:
        return self._run_limited(tasks, worker, self.rf3_slots)

    def _run_limited(self, tasks: list[T], worker: Callable[[T], R], slots: int) -> list[R]:
        if not tasks:
            return []
        results: list[R | None] = [None] * len(tasks)
        max_workers = min(slots, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]

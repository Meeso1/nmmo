import time


class TaskTimer:
    _instance: 'TaskTimer' = None
    quiet: bool = False

    def __init__(self) -> None:
        self._start_times = {}

    @classmethod
    def _get(cls) -> 'TaskTimer':
        if cls._instance is None:
            cls._instance = TaskTimer()
        return cls._instance

    @staticmethod
    def start_task(task: str, show: bool = True) -> None:
        instance = TaskTimer._get()
        instance._start_times[task] = time.time()
        if show and not TaskTimer.quiet:
            print(f"Starting task '{task}'")

    @staticmethod
    def end_task(task: str, show: bool = True) -> float | None:
        instance = TaskTimer._get()
        if task not in instance._start_times:
            if show and not TaskTimer.quiet:
                print(f"Task '{task}' was not started")
            return None

        elapsed_time = time.time() - instance._start_times[task]
        if show and not TaskTimer.quiet:
            print(f"'{task}' finished: {elapsed_time:.2f}s")

        del instance._start_times[task]
        return elapsed_time

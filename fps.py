from datetime import datetime

class FPS:
    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()

        return self

    def increment(self):
        self._num_occurrences += 1

    def counts_per_sec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()

        return self._num_occurrences / elapsed_time

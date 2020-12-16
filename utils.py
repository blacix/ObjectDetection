import time


class OpsMeter:
    def __init__(self):
        self.prev_time = time.time()
        self.current_time = time.time()
        self.ops = 0

    def loop(self):
        self.current_time = time.time()
        self.ops = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time
        self.ops = int(self.ops)
        return self.ops

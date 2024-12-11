from collections import deque
class VideoBuffer:
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)
    def add_Frame(self, frame):
        self.buffer.append(frame)
    def get_Buffer(self):
        if self.buffer:
            return self.buffer
        return None  
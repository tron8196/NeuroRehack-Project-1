import cv2 as cv
from threading import Thread
import time

class VideoStream:
    """
    Class that continuously gets frames from a video file
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.read_queue = []

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                if self.grabbed:
                    self.read_queue.append(self.frame)

    def stop(self):
        self.stopped = True

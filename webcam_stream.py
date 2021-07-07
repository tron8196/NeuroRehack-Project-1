import cv2 as cv
from threading import Thread

class WebcamStream:
    """
    Class that continuously gets frames from a webcam
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

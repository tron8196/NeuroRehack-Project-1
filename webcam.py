import cv2 as cv
from fps import FPS
from video_get import VideoGet


if __name__ == '__main__':

    vid_getter = VideoGet(0).start()
    fps = FPS().start()

    font = cv.FONT_HERSHEY_SIMPLEX

    while True:
        if cv.waitKey(1) == ord('q') or vid_getter.stopped:
            vid_getter.stop()
            break

        frame = vid_getter.frame
        frame = cv.flip(frame, 1)
        cv.putText(frame, "{:.0f} FPS".format(fps.counts_per_sec()),
            (10, 450), font, 1.0, (0, 0, 255))

        print('FPS of video: %d' % fps.counts_per_sec(), end="\r", flush=True)
        cv.imshow('webcam', frame)
        fps.increment()

print('FPS of video: %d' % fps.counts_per_sec())

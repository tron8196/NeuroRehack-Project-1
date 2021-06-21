from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
from skeleton_sequence import SkeletonSequence
import pyttsx3
import threading
import numpy as np

try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

ROOT_PATH = './'
json_recordings_dir = os.path.join(ROOT_PATH, 'json_recordings')
video_recordings_dir = os.path.join(ROOT_PATH, 'video_recordings')

class Compare:
    def __init__(self, args):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)

        self.params = dict()
        self.params['model_pose'] = 'BODY_25'
        self.params['model_folder'] = "../openpose/models/"
        self.params['net_resolution'] = "-656x368"

        self.skeleton_seq = SkeletonSequence()
        self.skeleton_seq_comp = SkeletonSequence()

        exercise_json_dir = os.path.join(json_recordings_dir, args.folder)
        file_name = next(os.walk(exercise_json_dir))[2][0]
        self.skeleton_seq_comp.load_from_json(os.path.join(exercise_json_dir, file_name))

        self.user_in_position = False

        exercise_vid_dir = os.path.join(video_recordings_dir, args.folder)
        self.template_video = os.path.join(exercise_vid_dir, next(os.walk(exercise_vid_dir))[2][0])
        print(self.template_video)


    def countdown_text(self):
        self.engine.say('Please get to a position so that the camera can see your whole body.')
        # self.engine.say('10'), self.engine.say('9'), self.engine.say('8')
        for i in range(3, -1, -1):
            self.engine.say(i)

        self.engine.runAndWait()

        self.user_in_position = True


    def webcam_loop(self):
        try:
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(self.params)
            opWrapper.start()

            stream = cv.VideoCapture(0)
            template_stream = cv.VideoCapture(self.template_video)

            video_codex = cv.VideoWriter_fourcc('F','M','P','4')
            writer = cv.VideoWriter('./output.mp4', video_codex, 30.0, (640, 480))

            font = cv.FONT_HERSHEY_SIMPLEX

            startTime = datetime.now()
            while True:
                datum = op.Datum()
                has_frame, image = stream.read()
                if not has_frame:
                    break

                if self.user_in_position:
                    datum.cvInputData = image
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                    body_keypoints = datum.poseKeypoints
                    if (body_keypoints.shape[0]) > 1:
                        self.engine.say('This program does not support more than 1 person in frame!')
                        self.engine.runAndWait()
                        break

                    self.skeleton_seq.add_keypoints(body_keypoints)
                    # Display Image
                    output_image = datum.cvOutputData
                    has_template_frame, template_image = template_stream.read()
                    if not has_template_frame:
                        break

                    writer.write(image)

                    cv.putText(output_image, "Press 'q' to quit or 's' to save", (20, 30),
                    font, 1, (0, 0, 0), 1, cv.LINE_AA)

                    cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                    cv.resizeWindow('image', 1280, 720)
                    horizontal = np.concatenate((cv.resize(cv.flip(template_image, 1), (300, 300)),
                                                 cv.resize(cv.flip(output_image, 1), (300, 300))), axis=1)
                    cv.imshow('Openpose result', horizontal)

                    key = cv.waitKey(1)
                    if key == ord('q'):
                        print("Time taken:", str(datetime.now() - startTime))
                        break
                else:
                    cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                    cv.resizeWindow('image', 1280, 720)
                    cv.imshow('Openpose result', cv.flip(image, 1))
                    cv.waitKey(1)

            stream.release()
            template_stream.release()
            writer.release()
            cv.destroyAllWindows()
        except Exception as e:
            print(e)
            sys.exit(-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Record an action or compare with an existing one')

    parser.add_argument('--folder', help='Name of an action folder inside of recordings directory.')

    args = parser.parse_args()

    assert args.folder, "Argument --folder is required to find json recordings of an action."

    comp = Compare(args)

    webcam_p = threading.Thread(target=comp.webcam_loop)
    countdown_p = threading.Thread(target=comp.countdown_text)

    webcam_p.start()
    countdown_p.start()

    countdown_p.join()
    webcam_p.join()

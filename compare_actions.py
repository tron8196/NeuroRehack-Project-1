from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
from skeleton_sequence import SkeletonSequence
import pyttsx3
import threading

try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

ROOT_PATH = './'
json_recordings_dir = os.path.join(ROOT_PATH, 'json_recordings')

class Compare:
    def __init__(self, args):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)

        self.params = dict()
        self.params['model_pose'] = 'BODY_25'
        self.params['model_folder'] = "../openpose/models/"

        self.skeleton_seq = SkeletonSequence()
        self.skeleton_seq_comp = SkeletonSequence()

        exercise_dir = os.path.join(json_recordings_dir, args.folder)
        file_name = next(os.walk(exercise_dir))[2][0]
        self.skeleton_seq_comp.load_from_json(os.path.join(exercise_dir, file_name))

        self.user_in_position = False


    def countdown_text(self):
        self.engine.say('Please get to a position so that the camera can see your whole body.')
        # self.engine.say('10'), self.engine.say('9'), self.engine.say('8')
        for i in range(7, -1, -1):
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
            font = cv.FONT_HERSHEY_SIMPLEX

            datum = op.Datum()

            startTime = datetime.now()
            while True:
                has_frame, image = stream.read()
                if not has_frame:
                    break

                if self.user_in_position:
                    datum.cvInputData = cv.resize(image, (128, 128))
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                    body_keypoints = datum.poseKeypoints
                    if (body_keypoints.shape[0]) > 1:
                        self.engine.say('This program does not support more than 1 person in frame!')
                        self.engine.runAndWait()
                        break

                    self.skeleton_seq.add_keypoints(body_keypoints)
                    # Display Image
                    output_image = datum.cvOutputData

                    cv.putText(output_image, "Press 'q' to quit or 's' to save", (20, 30),
                    font, 1, (0, 0, 0), 1, cv.LINE_AA)

                    cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                    cv.resizeWindow('image', 1366, 784)
                    cv.imshow('Openpose result', cv.flip(output_image, 1))

                    key = cv.waitKey(1)
                    if key == ord('q'):
                        print("Time taken:", str(datetime.now() - startTime))
                        print('Skeletons collected :: ', len(skeleton_seq.skeletons))
                        break
                    elif key == ord('s'):
                        if args.command == 'record':
                            assert args.folder, "Argument --folder is required to save this recording as a json file."
                            skeleton_seq.save_as_json(args.folder)
                        break
                else:
                    cv.namedWindow('Get into position', cv.WINDOW_NORMAL)
                    cv.resizeWindow('image', 1366, 784)
                    cv.imshow('Get into position', cv.flip(image, 1))
                    cv.waitKey(1)

            stream.release()
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

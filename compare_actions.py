from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
from skeleton_sequence import SkeletonSequence
import threading
import pyttsx3

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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

        self.skeleton_seq_comp = SkeletonSequence()

        exercise_json_dir = os.path.join(json_recordings_dir, args.folder)
        file_name = next(os.walk(exercise_json_dir))[2][0]
        self.skeleton_seq_comp.load_from_json(os.path.join(exercise_json_dir, file_name))

        self.user_in_position = False

        exercise_vid_dir = os.path.join(video_recordings_dir, args.folder)
        self.template_video = os.path.join(exercise_vid_dir, next(os.walk(exercise_vid_dir))[2][0])

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()


    def countdown_text(self):
        self.engine.say('Please get to a position so that the camera can see your whole body.')
        # self.engine.say('10'), self.engine.say('9'), self.engine.say('8')
        for i in range(7, -1, -1):
            self.engine.say(i)

        self.engine.runAndWait()

        self.user_in_position = True


    def passthrough_openpose(self, image):
        try:
            datum = op.Datum()
            datum.cvInputData = image
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            body_keypoints = datum.poseKeypoints
            if (body_keypoints.shape[0]) > 1:
                self.engine.say('This program does not support more than 1 person in frame!')
                self.engine.runAndWait()
                return None, None
            # Display Image
            output_image = datum.cvOutputData

            return body_keypoints, output_image
        except Exception as e:
            print(e)
            sys.exit(-1)


    def webcam_loop(self):
        stream = cv.VideoCapture(0)
        template_stream = cv.VideoCapture(self.template_video)

        video_codex = cv.VideoWriter_fourcc('F','M','P','4')
        writer = cv.VideoWriter('./output.mp4', video_codex, 30.0, (640, 480))

        font = cv.FONT_HERSHEY_SIMPLEX

        while True:
            has_frame, image = stream.read()
            if not has_frame:
                break

            if self.user_in_position:
                body_keypoints, output_image = self.passthrough_openpose(image)
                if output_image is None:
                    break

                # Display Image
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


    def process_output_video(self):
        output_vid_stream = cv.VideoCapture('./output.mp4')
        skeleton_seq = SkeletonSequence()

        no_of_frames = 0
        while True:
            has_frame, image = output_vid_stream.read()
            if not has_frame:
                break

            body_keypoints, _ = self.passthrough_openpose(image)
            if body_keypoints is None:
                break

            skeleton_seq.add_keypoints(body_keypoints)
            no_of_frames += 1
            print('No of frames processed: %d' % no_of_frames, end="\r", flush=True)

        print('Total no of frames processed: %d' % no_of_frames)
        skeleton_seq.smoothen()
        skeleton_seq.save_as_json(output=True)
        output_vid_stream.release()

        return skeleton_seq


    def calc_dtw_score(self, skeleton_seq):
        no_of_joints = len(skeleton_seq.sequence_data['joint_angles'].keys())
        aggregate_distance_score = 0

        for key in skeleton_seq.sequence_data['joint_angles'].keys():
            distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles'][key],
                            self.skeleton_seq_comp.sequence_data['joint_angles'][key], dist=euclidean)

            aggregate_distance_score += distance
            print('DTW score for ' + key + ': ', distance)

        print('Aggregate distance score: ', aggregate_distance_score)
        print('Avg DTW distance score: ', (aggregate_distance_score/no_of_joints))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Record an action or compare with an existing one')

    parser.add_argument('--folder', help='Name of an action folder inside of recordings directory.')

    parser.add_argument('--direct_compare', action='store_true')

    args = parser.parse_args()

    assert args.folder, "Argument --folder is required to find json recordings of an action."

    comp = Compare(args)

    if not args.direct_compare:
        webcam_p = threading.Thread(target=comp.webcam_loop)
        countdown_p = threading.Thread(target=comp.countdown_text)

        webcam_p.start()
        countdown_p.start()

        countdown_p.join()
        webcam_p.join()

    skeleton_seq = comp.process_output_video()
    comp.calc_dtw_score(skeleton_seq)

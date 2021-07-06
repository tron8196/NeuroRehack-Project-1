from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
import threading
import pyttsx3

from skeleton_sequence import SkeletonSequence
from skeleton import Skeleton
from video_get import VideoGet
from fps import FPS

import math
from colour import Color
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

import json
cf = open('./config.json')
json_config = json.load(cf)

NECK_INDEX = json_config['neck_index']
SHOULDER_LEFT_INDEX = json_config['shoulder_left_index']
SHOULDER_RIGHT_INDEX = json_config['shoulder_right_index']

green = Color("green")
colors = list(green.range_to(Color("red"), 9))

class Compare:
    def __init__(self, args):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)

        self.params = dict()
        self.params['model_pose'] = json_config['model_pose']
        self.params['model_folder'] = json_config['model_folder']
        self.params['net_resolution'] = json_config['net_resolution']

        self.skeleton_seq_comp = SkeletonSequence()

        self.folder_name = args.folder
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

        self.pose_pairs = [
            [1, 0], [1, 2], [1, 5], [1, 8],
            [3, 2], [3, 4],
            [6, 5], [6, 7],
            [8, 9], [8, 12],
            [10, 9], [10, 11],
            [13, 12], [13, 14]
        ]


    def countdown_text(self):
        exercise_name = ' '.join(self.folder_name.split('_'))
        self.engine.say('You are about to perform.')
        self.engine.say(exercise_name)
        self.engine.say('Please get to a position so that the camera can see your whole body.')
        # self.engine.say('10'), self.engine.say('9'), self.engine.say('8')
        for i in range(json_config['countdown_duration'], -1, -1):
            self.engine.say(i)

        self.engine.runAndWait()

        self.user_in_position = True


    def passthrough_openpose(self, image):
        try:
            datum = op.Datum()
            datum.cvInputData = image
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            pose_kp = datum.poseKeypoints
            skeleton = None
            output_image = None
            if (pose_kp.shape[0]) > 1:
                self.engine.say('This program does not support more than 1 person in frame!')
                self.engine.runAndWait()

                return skeleton, output_image

            skeleton = Skeleton(pose_kp)
            # Display Image
            output_image = datum.cvOutputData

            return skeleton, output_image
        except Exception as e:
            print(e)
            sys.exit(-1)


    def webcam_loop(self):
        fps = FPS().start()
        stream = VideoGet(0).start()
        template_stream = cv.VideoCapture(self.template_video)

        video_codex = cv.VideoWriter_fourcc(*'XVID')
        writer = cv.VideoWriter('./output.avi', video_codex, 30.0, (640, 480))

        font = cv.FONT_HERSHEY_SIMPLEX

        total_template_frames = len(self.skeleton_seq_comp.sequence_data['normalized_keypoints'])
        template_frame_index = 0

        while True:
            image = stream.frame

            if cv.waitKey(1) == ord('q') or stream.stopped:
                stream.stop()
                break

            if self.user_in_position:
                skeleton, output_image = self.passthrough_openpose(image)
                if output_image is None:
                    stream.stop()
                    break

                # Display Image
                has_frame, template_image = template_stream.read()
                if not has_frame:
                    stream.stop()
                    break

                writer.write(image)

                if template_frame_index < total_template_frames:
                    keypoints = skeleton.keypoints
                    # draw skeleton
                    for pair in self.pose_pairs:
                        point1_idx = pair[0]
                        point2_idx = pair[1]

                        point1_2D = (int(keypoints[point1_idx][0]), int(keypoints[point1_idx][1]))
                        point2_2D = (int(keypoints[point2_idx][0]), int(keypoints[point2_idx][1]))

                        if point1_2D[0] and point2_2D[0] and point1_2D[1] and point2_2D[1]:
                            # distance score
                            point1_compare = np.array(self.skeleton_seq_comp.sequence_data['normalized_keypoints'][template_frame_index][point1_idx])
                            point1 = np.array(skeleton.normalized_keypoints[point1_idx])
                            point1_dist = np.linalg.norm(point1 - point1_compare)

                            point2_compare = np.array(self.skeleton_seq_comp.sequence_data['normalized_keypoints'][template_frame_index][point2_idx])
                            point2 = np.array(skeleton.normalized_keypoints[point2_idx])
                            point2_dist = np.linalg.norm(point2 - point2_compare)

                            dist_score = (point1_dist + point2_dist) / 2
                            dist_score = int(dist_score*10)
                            if dist_score > 8:
                                dist_score = 8

                            print('DIST SCORE :: ', dist_score)

                            cv.line(image, point1_2D, point2_2D, tuple(255*t for t in colors[dist_score].rgb[::-1]), 5)

                    template_frame_index+=1

                cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                cv.resizeWindow('image', 1280, 720)
                horizontal = np.concatenate((cv.resize(cv.flip(template_image, 1), (300, 300)),
                                             cv.resize(cv.flip(image, 1), (300, 300))), axis=1)
                cv.imshow('Openpose result', horizontal)
            else:
                cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                cv.resizeWindow('image', 1280, 720)
                cv.imshow('Openpose result', cv.flip(image, 1))

            print('FPS of video: %d' % fps.counts_per_sec(), end="\r", flush=True)
            fps.increment()

        template_stream.release()
        writer.release()
        cv.destroyAllWindows()
        print('FPS of video: %d' % fps.counts_per_sec())


    def process_output_video(self):
        output_vid_stream = cv.VideoCapture('./output.mp4')
        skeleton_seq = SkeletonSequence()

        no_of_frames = 0
        while True:
            has_frame, image = output_vid_stream.read()
            if not has_frame:
                break

            skeleton, _ = self.passthrough_openpose(image)

            if skeleton is None:
                break

            skeleton_seq.add_keypoints(skeleton.keypoints)
            no_of_frames += 1
            print('No of frames processed: %d' % no_of_frames, end="\r", flush=True)

        print('Total no of frames processed: %d' % no_of_frames)
        skeleton_seq.smoothen()
        skeleton_seq.save_as_json(output=True)
        output_vid_stream.release()

        return skeleton_seq


    def calc_dtw_score(self, skeleton_seq):
        upper_joint_angles = ['LNeckJoint', 'RNeckJoint', 'LArmpitJoint', 'RArmpitJoint',
        'LElbowJoint', 'RElbowJoint','LHipJoint','RHipJoint']

        agg_score = 0
        no_of_joints = len(upper_joint_angles)
        for key in upper_joint_angles:
            distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles'][key],
                            self.skeleton_seq_comp.sequence_data['joint_angles'][key], dist=euclidean)

            agg_score += distance
            print('DTW score for ' + key + ': ', distance)

        print('Upper body avg DTW distance score: ', (agg_score/no_of_joints))


        lower_joint_angles = ['LThighJoint', 'RThighJoint', 'LKneeJoint', 'RKneeJoint']

        agg_score = 0
        no_of_joints = len(lower_joint_angles)
        for key in lower_joint_angles:
            distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles'][key],
                            self.skeleton_seq_comp.sequence_data['joint_angles'][key], dist=euclidean)

            agg_score += distance
            print('DTW score for ' + key + ': ', distance)

        print('Lower body avg DTW distance score: ', (agg_score/no_of_joints))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Record an action or compare with an existing one')

    parser.add_argument('--folder', help='Name of an action folder inside of recordings directory.')

    parser.add_argument('--no_webcam', action='store_true')

    args = parser.parse_args()

    assert args.folder, "Argument --folder is required to find json recordings of an action."

    comp = Compare(args)

    if not args.no_webcam:
        webcam_p = threading.Thread(target=comp.webcam_loop)
        countdown_p = threading.Thread(target=comp.countdown_text)

        webcam_p.start()
        countdown_p.start()

        countdown_p.join()
        webcam_p.join()

    skeleton_seq = comp.process_output_video()
    comp.calc_dtw_score(skeleton_seq)

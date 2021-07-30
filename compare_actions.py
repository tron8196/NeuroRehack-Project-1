from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
import threading
import pyttsx3
import time

from skeleton_sequence import SkeletonSequence
from skeleton import Skeleton
from webcam_stream import WebcamStream
from video_stream import VideoStream
from fps import FPS

import math
from colour import Color
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os

# Change these variables to point to the correct folder (Release/x64 etc.)
if os.name == 'nt':
    dir_path = r'C:/Users/hp/openpose'
    sys.path.append(dir_path + '/build/python/openpose/Release');
    print(dir_path + '/build/python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/build/x64/Release;' +  dir_path + '/build/bin;'
    print(os.environ['PATH'] + ';' + dir_path + '/build/x64/Release;' +  dir_path + '/build/bin;')


try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

import json
cf = open('./config.json')
json_config = json.load(cf)

ROOT_PATH = './'
json_recordings_dir = os.path.join(ROOT_PATH, json_config['template_vids_json'])
video_recordings_dir = os.path.join(ROOT_PATH, json_config['template_vids'])
webcam_json_recordings_dir = os.path.join(ROOT_PATH, json_config['webcam_vids_json'])
webcam_video_recordings_dir = os.path.join(ROOT_PATH, json_config['webcam_vids'])

NECK_INDEX = json_config['neck_index']
SHOULDER_LEFT_INDEX = json_config['shoulder_left_index']
SHOULDER_RIGHT_INDEX = json_config['shoulder_right_index']

green = Color("green")
colors = list(green.range_to(Color("red"), 9))

body_keypoint_dict = {
    0: 'Nose',
    1: 'Neck',
    2: 'RShoulder',
    3: 'RElbow',
    4: 'RWrist',
    5: 'LShoulder',
    6: 'LElbow',
    7: 'LWrist',
    8: 'Midhip',
    9: 'RHip',
    10: 'RKnee',
    11: 'RAnkle',
    12: 'LHip',
    13: 'LKnee',
    14: 'LAnkle'
}

class Compare:
    def __init__(self, args):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 140)

        self.params = dict()
        self.params['model_pose'] = json_config['model_pose']
        self.params['model_folder'] = json_config['model_folder_windows'] if os.name == 'nt' else json_config['model_folder']
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
        webcam_stream = WebcamStream().start()
        # TODO: do not read all frames at once
        template_video_stream = VideoStream(self.template_video).start()

        webcam_dir = os.path.join(webcam_video_recordings_dir, self.folder_name)
        webcam_file = self.folder_name + ".avi"
        video_codex = cv.VideoWriter_fourcc(*'XVID')
        writer = cv.VideoWriter(os.path.join(webcam_dir, webcam_file), video_codex, 30.0, (640, 480))

        font = cv.FONT_HERSHEY_SIMPLEX

        total_template_frames = len(self.skeleton_seq_comp.sequence_data['normalized_keypoints'])
        template_frame_index = 0
        ms_per_frame = 1000 / json_config['template_vid_fps']
        while template_frame_index < total_template_frames-1:
            image = webcam_stream.frame
            if cv.waitKey(1) == ord('q') or webcam_stream.stopped:
                webcam_stream.stop()
                template_video_stream.stop()
                break

            if self.user_in_position:
                start_time = datetime.now()

                skeleton, output_image = self.passthrough_openpose(image)
                if output_image is None:
                    webcam_stream.stop()
                    template_video_stream.stop()
                    break

                # Display Image
                template_image = template_video_stream.read_queue[template_frame_index]

                writer.write(image)

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
                        point2_compare = np.array(self.skeleton_seq_comp.sequence_data['normalized_keypoints'][template_frame_index][point2_idx])

                        if not(point1_compare[0] == -1 and point1_compare[1] == -1 and point1_compare[2] == -1) and not(point2_compare[0] == -1 and point2_compare[1] == -1 and point2_compare[2] == -1):
                            point1 = np.array(skeleton.normalized_keypoints[point1_idx])
                            point1_dist = np.linalg.norm(point1 - point1_compare)

                            point2 = np.array(skeleton.normalized_keypoints[point2_idx])
                            point2_dist = np.linalg.norm(point2 - point2_compare)

                            dist_score = (point1_dist + point2_dist) / 2
                            dist_score = int(dist_score*10)
                            if dist_score > 8:
                                dist_score = 8


                            color_tuple = tuple(255*t for t in colors[dist_score].rgb[::-1])
                            cv.line(image, point1_2D, point2_2D, color_tuple, 5)
                            cv.circle(image, point1_2D, 6, color_tuple, thickness=2, lineType=cv.FILLED)
                            cv.circle(image, point2_2D, 6, color_tuple, thickness=2, lineType=cv.FILLED)


                # print(template_frame_index)
                # print('no of frames in queue: ', len(template_video_stream.read_queue))
                # print('time taken for openpose processing : ', (datetime.now() - start_time))
                processing_time = ((datetime.now()-start_time).microseconds)/1000
                no_of_frames_to_skip = math.ceil(processing_time / ms_per_frame)
                template_frame_index += no_of_frames_to_skip
                cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                cv.resizeWindow('image', 1280, 720)
                horizontal = np.concatenate((cv.resize(cv.flip(template_image, 1), (300, 300)),
                                             cv.resize(cv.flip(image, 1), (300, 300))), axis=1)
                cv.imshow('Openpose result', horizontal)
            else:
                cv.namedWindow('Openpose result', cv.WINDOW_NORMAL)
                cv.resizeWindow('image', 1280, 720)
                cv.imshow('Openpose result', cv.flip(image, 1))

            # print('FPS of video: %d' % fps.counts_per_sec(), end="\r", flush=True)
            fps.increment()

        webcam_stream.stop()
        template_video_stream.stop()
        writer.release()
        cv.destroyAllWindows()
        # print('FPS of video: %d' % fps.counts_per_sec())


    def process_output_video(self):
        webcam_dir = os.path.join(webcam_video_recordings_dir, self.folder_name)
        file_names = next(os.walk(webcam_dir))[2]

        for file_name in file_names:
            output_vid_stream = cv.VideoCapture(os.path.join(webcam_dir, file_name))
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
            skeleton_seq.create_sequence_data()
            skeleton_seq.save_as_json(self.folder_name, webcam=True)

        output_vid_stream.release()
        return skeleton_seq


    def calc_dtw_score(self, skeleton_seq):
        skeleton_seq.smoothen()            # webcam vid
        self.skeleton_seq_comp.smoothen() # template vid

        seq_comp_keypoints = {}
        for keypoints in self.skeleton_seq_comp.sequence_data['normalized_keypoints']:
            for index, point in enumerate(keypoints):
                kp_list = seq_comp_keypoints.get(index, [])
                point = [point[0], point[1]]
                kp_list.append(point)
                seq_comp_keypoints[index] = kp_list

        seq_keypoints = {}
        for keypoints in skeleton_seq.sequence_data['normalized_keypoints']:
            for index, point in enumerate(keypoints):
                kp_list = seq_keypoints.get(index, [])
                point = [point[0], point[1]]
                kp_list.append(point)
                seq_keypoints[index] = kp_list

        agg_score = 0
        no_of_kp_selected = 0
        relevant_key_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        for key in relevant_key_points:
            if not(seq_comp_keypoints[key][0][0] == -1) and not(seq_comp_keypoints[key][0][1] == -1):
                distance, _ = fastdtw(np.array(seq_keypoints[key]),
                                     np.array(seq_comp_keypoints[key]), dist=euclidean)

                print('DTW score for body point ' + body_keypoint_dict[key] + ' is: ', distance)
                no_of_kp_selected+=1
                agg_score += distance

        print('No of relevant points : ', no_of_kp_selected)
        print('Avg DTW score for normalized skeletons : ', (agg_score/no_of_kp_selected))
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        upper_joint_angles = ['LNeckJoint', 'RNeckJoint', 'LArmpitJoint', 'RArmpitJoint',
        'LElbowJoint', 'RElbowJoint','LHipJoint','RHipJoint']

        agg_score = 0
        no_of_joints = len(upper_joint_angles)
        no_of_joints_selected = 0
        for key in upper_joint_angles:
            variance = np.var(self.skeleton_seq_comp.sequence_data['joint_angles'][key])
            variance = round(variance, 2)

            if variance > 0.01:
                no_of_joints_selected+=1
                distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles'][key],
                                self.skeleton_seq_comp.sequence_data['joint_angles'][key], dist=euclidean)

                agg_score += distance
                print('DTW score for ' + key + ' angle: ', distance)

        if agg_score:
            print('Upper body avg DTW distance score: ', (agg_score/no_of_joints_selected))
            print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")


        lower_joint_angles = ['LThighJoint', 'RThighJoint', 'LKneeJoint', 'RKneeJoint']
        agg_score = 0
        no_of_joints = len(lower_joint_angles)
        no_of_joints_selected = 0
        for key in lower_joint_angles:
            variance = np.var(self.skeleton_seq_comp.sequence_data['joint_angles'][key])
            variance = round(variance, 2)

            if variance > 0.01:
                no_of_joints_selected+=1
                distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles'][key],
                                self.skeleton_seq_comp.sequence_data['joint_angles'][key], dist=euclidean)

                agg_score += distance
                print('DTW score for ' + key + ' angle: ', distance)

        if agg_score:
            print('Lower body avg DTW distance score: ', (agg_score/no_of_joints_selected))


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

    else:
        # process from already available json
        webcam_json_dir = os.path.join(webcam_json_recordings_dir, args.folder)
        file_name = next(os.walk(webcam_json_dir))[2][0]
        skeleton_seq = SkeletonSequence()
        skeleton_seq.load_from_json(os.path.join(webcam_json_dir, file_name))
        comp.calc_dtw_score(skeleton_seq)

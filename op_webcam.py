from datetime import datetime
import cv2 as cv
import os
import sys
import argparse
import json
from skeleton import Skeleton
import argparse

# use if necessary
# sys.path.append('/usr/local/python')

ROOT_PATH = './'
recordings_dir = os.path.join(ROOT_PATH, 'recordings')

def save_as_json(skeletons, folder_name=None):
    action_dir = os.path.join(recordings_dir, folder_name)
    file_name = "recording_{:%Y%m%dT%H%M%S}.json".format(datetime.now())
    joint_angles = []
    for s in skeletons:
        joint_angles.append(s.joint_angles)

    data = {'patient_name': '', 'joint_angles': joint_angles}

    with open(os.path.join(action_dir, file_name), 'w', encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# current directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

def set_params():
    params = dict()
    params['model_pose'] = 'BODY_25'
    params['model_folder'] = "../openpose/models/"

    return params

if __name__ == '__main__':

    params = set_params()
    skeletons = []

    parser = argparse.ArgumentParser(description='Record an action or compare with an existing one')

    parser.add_argument('command', help="'record' or 'compare'")

    parser.add_argument('--folder', default='action1', help='Name of an action folder inside of recordings directory.')

    args = parser.parse_args()

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        stream = cv.VideoCapture(0)
        font = cv.FONT_HERSHEY_SIMPLEX

        startTime = datetime.now()
        while True:
            # Process Image
            datum = op.Datum()
            _, image = stream.read()
            datum.cvInputData = image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            body_keypoints = datum.poseKeypoints
            if (body_keypoints.shape[0]) > 1:
                print('This program does not support more than 1 person in the frame!')
                break
            skeletons.append(Skeleton(body_keypoints))

            # Display Image
            output_image = datum.cvOutputData
            cv.putText(output_image, "Press 'q' to quit", (20, 30),
                                font, 1, (0, 0, 0), 1, cv.LINE_AA)
            cv.imshow("Openpose result", output_image)

            key = cv.waitKey(1)
            if key == ord('q'):
                print("Time taken:", str(datetime.now() - startTime))
                print('Skeletons collected :: ', len(skeletons))

                if args.command == 'record':
                    save_as_json(skeletons, args.folder)

                break

    except Exception as e:
        print(e)
        sys.exit(-1)

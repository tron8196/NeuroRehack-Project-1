from datetime import datetime
import os
import cv2 as cv
import sys
import argparse
from skeleton_sequence import SkeletonSequence

try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

ROOT_PATH = './'
video_recordings_dir = os.path.join(ROOT_PATH, 'video_recordings')

def set_params():
    params = dict()
    params['model_pose'] = 'BODY_25'
    params['model_folder'] = "../openpose/models/"
    params['net_resolution'] = "-1x128"

    return params

if __name__ == '__main__':

    params = set_params()

    parser = argparse.ArgumentParser(description='Record an action or compare with an existing one')
    parser.add_argument('--folder', help='Name of an action folder inside of recordings directory.')
    args = parser.parse_args()

    assert args.folder, "Argument --folder is required to save this recording as a json file."

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        exercise_dir = os.path.join(video_recordings_dir, args.folder)
        print(exercise_dir)
        file_names = next(os.walk(exercise_dir))[2]
        print(file_names)
        for file_name in file_names:
            print(os.path.join(exercise_dir, file_name))
            stream = cv.VideoCapture(os.path.join(exercise_dir, file_name))
            skeleton_seq = SkeletonSequence()

            startTime = datetime.now()
            no_of_frames = 0
            while True:
                # Process Image
                datum = op.Datum()
                has_frame, image = stream.read()
                if not has_frame:
                    break

                datum.cvInputData = image
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                body_keypoints = datum.poseKeypoints
                if (body_keypoints.shape[0]) > 1:
                    print('This program does not support more than 1 person in the frame!')
                    break

                skeleton_seq.add_keypoints(body_keypoints)
                no_of_frames += 1
                print('No of frames processed: %d' % no_of_frames, end="\r", flush=True)

            skeleton_seq.save_as_json(args.folder)

    except Exception as e:
        print(e)
        sys.exit(-1)

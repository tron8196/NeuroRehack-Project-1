
import cv2 as cv
import os
import sys
import argparse

# use if necessary 
# sys.path.append('/usr/local/python')

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

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        stream = cv.VideoCapture(0)
        font = cv.FONT_HERSHEY_SIMPLEX

        while True:
            # Process Image
            datum = op.Datum()
            _, image = stream.read()
            datum.cvInputData = image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display Image
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            output_image = datum.cvOutputData
            cv.putText(output_image, "Press 'q' to quit", (20, 30),
                                font, 1, (0, 0, 0), 1, cv.LINE_AA)
            cv.imshow("Openpose result", output_image)

            key = cv.waitKey(1)

            if key == ord('q'):
                break

    except Exception as e:
        print(e)
        sys.exit(-1)

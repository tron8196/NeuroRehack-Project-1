from datetime import datetime
import cv2 as cv
import numpy as np
import os
import sys
import argparse

# use if necessary
# sys.path.append('/usr/local/python')

class Skeleton:
    def __init__(self, body_keypoints):
        bk = body_keypoints.reshape(25, 3)
        self.body_kp = {
            'Nose': bk[0],
            'Neck': bk[1],
            'RShoulder': bk[2],
            'RElbow': bk[3],
            'RWrist': bk[4],
            'LShoulder': bk[5],
            'LElbow': bk[6],
            'LWrist': bk[7],
            'Midhip': bk[8],
            'RHip': bk[9],
            'RKnee': bk[10],
            'RAnkle': bk[11],
            'LHip': bk[12],
            'LKnee': bk[13],
            'LAnkle': bk[14],
            'REye': bk[15],
            'LEye': bk[16],
            'REar': bk[17],
            'LEar': bk[18],
            'LBigToe': bk[19],
            'LSmallToe': bk[20],
            'LHeel': bk[21],
            'RBigToe': bk[22],
            'RSmallToe': bk[23],
            'RHeel': bk[24]
        }
        self.calculate_joint_angles()

        print('Joint Angles :: ')
        print(self.joint_angles)


    def calc_joint_angle(self, a, b, c):
        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / ( np.linalg.norm(ba) * np.linalg.norm(bc) )

        return np.arccos(cosine)


    def calculate_joint_angles(self):
        self.joint_angles = {
            'LArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['LShoulder'], self.body_kp['LElbow']),
            'RArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['RShoulder'], self.body_kp['RElbow']),
            'LElbowJoint': self.calc_joint_angle(self.body_kp['LShoulder'], self.body_kp['LElbow'], self.body_kp['LWrist']),
            'RElbowJoint': self.calc_joint_angle(self.body_kp['RShoulder'], self.body_kp['RElbow'], self.body_kp['RWrist']),
            'LHipJoint': self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['LHip'], self.body_kp['LKnee']),
            'RHipJoint': self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['RHip'], self.body_kp['RKnee']),
            'LKneeJoint': self.calc_joint_angle(self.body_kp['LHip'], self.body_kp['LHip'], self.body_kp['LAnkle']),
            'RKneeJoint': self.calc_joint_angle(self.body_kp['RHip'], self.body_kp['RHip'], self.body_kp['RAnkle'])
        }


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
                break

    except Exception as e:
        print(e)
        sys.exit(-1)

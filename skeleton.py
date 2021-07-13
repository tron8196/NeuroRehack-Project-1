import numpy as np

import json
cf = open('./config.json')
json_config = json.load(cf)

NECK_INDEX = json_config['neck_index']
SHOULDER_LEFT_INDEX = json_config['shoulder_left_index']
SHOULDER_RIGHT_INDEX = json_config['shoulder_right_index']

class Skeleton:
    def __init__(self, data, load_from_json=False):
        if load_from_json:
            self.joint_angles = data
        else:
            self.keypoints = data.reshape(25, 3)
            self.body_kp = {
                'Nose': self.keypoints[0],
                'Neck': self.keypoints[1],
                'RShoulder': self.keypoints[2],
                'RElbow': self.keypoints[3],
                'RWrist': self.keypoints[4],
                'LShoulder': self.keypoints[5],
                'LElbow': self.keypoints[6],
                'LWrist': self.keypoints[7],
                'Midhip': self.keypoints[8],
                'RHip': self.keypoints[9],
                'RKnee': self.keypoints[10],
                'RAnkle': self.keypoints[11],
                'LHip': self.keypoints[12],
                'LKnee': self.keypoints[13],
                'LAnkle': self.keypoints[14],
                'REye': self.keypoints[15],
                'LEye': self.keypoints[16],
                'REar': self.keypoints[17],
                'LEar': self.keypoints[18],
                'LBigToe': self.keypoints[19],
                'LSmallToe': self.keypoints[20],
                'LHeel': self.keypoints[21],
                'RBigToe': self.keypoints[22],
                'RSmallToe': self.keypoints[23],
                'RHeel': self.keypoints[24]
            }
            self.calculate_joint_angles()

            shoulder_vector = abs(np.subtract(self.keypoints[SHOULDER_LEFT_INDEX], self.keypoints[SHOULDER_RIGHT_INDEX]))
            self.shoulder_dist = np.linalg.norm(shoulder_vector)
            self.normalize_body_points()


    def normalize_body_points(self):
        self.normalized_keypoints = []

        for point in self.keypoints:
            if point[0] or point[1] or point[2]:
                normalized_point = np.divide(np.subtract(point, self.keypoints[NECK_INDEX]), self.shoulder_dist)
                self.normalized_keypoints.append(list(np.float64(normalized_point)))
            else:
                self.normalized_keypoints.append([-1, -1, -1])


    def calc_joint_angle(self, a, b, c):
        ba = a - b
        bc = c - b

        dot_product = np.dot(ba, bc)
        magnitude =  ( np.linalg.norm(ba) * np.linalg.norm(bc) )

        if not magnitude:
            return 0.0

        cosine = dot_product / magnitude

        return np.float64(np.arccos(cosine))


    def calculate_joint_angles(self):
        self.joint_angles = {
            'LNeckJoint': self.calc_joint_angle(self.body_kp['Nose'], self.body_kp['Neck'], self.body_kp['LShoulder']),
            'RNeckJoint': self.calc_joint_angle(self.body_kp['Nose'], self.body_kp['Neck'], self.body_kp['RShoulder']),
            'LArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['LShoulder'], self.body_kp['LElbow']),
            'RArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['RShoulder'], self.body_kp['RElbow']),
            'LElbowJoint':  self.calc_joint_angle(self.body_kp['LShoulder'], self.body_kp['LElbow'], self.body_kp['LWrist']),
            'RElbowJoint':  self.calc_joint_angle(self.body_kp['RShoulder'], self.body_kp['RElbow'], self.body_kp['RWrist']),
            'LHipJoint':    self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['Midhip'], self.body_kp['LHip']),
            'RHipJoint':    self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['Midhip'], self.body_kp['RHip']),
            'LThighJoint': self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['LHip'], self.body_kp['LKnee']),
            'RThighJoint': self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['RHip'], self.body_kp['RKnee']),
            'LKneeJoint':   self.calc_joint_angle(self.body_kp['LHip'], self.body_kp['LHip'], self.body_kp['LAnkle']),
            'RKneeJoint':   self.calc_joint_angle(self.body_kp['RHip'], self.body_kp['RHip'], self.body_kp['RAnkle'])
        }

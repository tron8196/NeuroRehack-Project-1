import numpy as np

NECK_INDEX = 1
SHOULDER_LEFT_INDEX = 2
SHOULDER_RIGHT_INDEX = 5

class Skeleton:
    def __init__(self, data, load_from_json=False):
        if load_from_json:
            self.joint_angles = data
        else:
            keypoints = data.reshape(25, 3)
            self.body_kp = {
                'Nose': keypoints[0],
                'Neck': keypoints[1],
                'RShoulder': keypoints[2],
                'RElbow': keypoints[3],
                'RWrist': keypoints[4],
                'LShoulder': keypoints[5],
                'LElbow': keypoints[6],
                'LWrist': keypoints[7],
                'Midhip': keypoints[8],
                'RHip': keypoints[9],
                'RKnee': keypoints[10],
                'RAnkle': keypoints[11],
                'LHip': keypoints[12],
                'LKnee': keypoints[13],
                'LAnkle': keypoints[14],
                'REye': keypoints[15],
                'LEye': keypoints[16],
                'REar': keypoints[17],
                'LEar': keypoints[18],
                'LBigToe': keypoints[19],
                'LSmallToe': keypoints[20],
                'LHeel': keypoints[21],
                'RBigToe': keypoints[22],
                'RSmallToe': keypoints[23],
                'RHeel': keypoints[24]
            }
            self.calculate_joint_angles()

            shoulder_vector = abs(np.subtract(keypoints[SHOULDER_LEFT_INDEX], keypoints[SHOULDER_RIGHT_INDEX]))
            self.shoulder_dist = np.linalg.norm(shoulder_vector)
            self.normalize_body_points(keypoints)


    def normalize_body_points(self, keypoints):
        self.normalized_bk = []

        for point in keypoints:
            normalized_point = np.divide(np.subtract(point, keypoints[NECK_INDEX]), self.shoulder_dist)
            self.normalized_bk.append(list(np.float64(normalized_point)))


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
            'LArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['LShoulder'], self.body_kp['LElbow']),
            'RArmpitJoint': self.calc_joint_angle(self.body_kp['Neck'], self.body_kp['RShoulder'], self.body_kp['RElbow']),
            'LElbowJoint':  self.calc_joint_angle(self.body_kp['LShoulder'], self.body_kp['LElbow'], self.body_kp['LWrist']),
            'RElbowJoint':  self.calc_joint_angle(self.body_kp['RShoulder'], self.body_kp['RElbow'], self.body_kp['RWrist']),
            'LHipJoint':    self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['LHip'], self.body_kp['LKnee']),
            'RHipJoint':    self.calc_joint_angle(self.body_kp['Midhip'], self.body_kp['RHip'], self.body_kp['RKnee']),
            'LKneeJoint':   self.calc_joint_angle(self.body_kp['LHip'], self.body_kp['LHip'], self.body_kp['LAnkle']),
            'RKneeJoint':   self.calc_joint_angle(self.body_kp['RHip'], self.body_kp['RHip'], self.body_kp['RAnkle'])
        }

import numpy as np

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

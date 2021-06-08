from skeleton import Skeleton
import os
from datetime import datetime
import json

ROOT_PATH = './'
recordings_dir = os.path.join(ROOT_PATH, 'recordings')

class SkeletonSequence():
    def __init__(self):
        # patient data, including joint angle sequence
        self.sequence_data = {}
        self.sequence_data['joint_angles'] = {
            'LArmpitJoint': [],
            'RArmpitJoint': [],
            'LElbowJoint':  [],
            'RElbowJoint':  [],
            'LHipJoint':    [],
            'RHipJoint':    [],
            'LKneeJoint':   [],
            'RKneeJoint':   []
        }

        # joint angle sequences
        self.skeletons = []

    def add_keypoints(self, body_keypoints):
        self.skeletons.append(Skeleton(body_keypoints))

    def load_from_json(self, folder_name=None):
        sf = open(folder_name)
        self.sequence_data = json.load(sf)

    def save_as_json(self, folder_name=None):
        action_dir = os.path.join(recordings_dir, folder_name)
        file_name = "recording_{:%Y%m%dT%H%M%S}.json".format(datetime.now())

        for skeleton in self.skeletons:
            for key, value in skeleton.joint_angles.items():
                self.sequence_data['joint_angles'][key].append(value)

        with open(os.path.join(action_dir, file_name), 'w', encoding='utf-8') as write_file:
            json.dump(self.sequence_data, write_file, ensure_ascii=False, indent=4)

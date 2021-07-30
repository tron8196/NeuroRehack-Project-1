from skeleton import Skeleton
import os
from datetime import datetime
import json

from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

import json
cf = open('./config.json')
json_config = json.load(cf)

ROOT_PATH = './'
json_recordings_dir = os.path.join(ROOT_PATH, json_config['template_vids_json'])
webcam_recordings_dir = os.path.join(ROOT_PATH, json_config['webcam_vids_json'])

class SkeletonSequence():
    def __init__(self):
        # patient data, including joint angle sequence
        self.sequence_data = {}
        self.sequence_data['joint_angles'] = {
            'LNeckJoint': [],
            'RNeckJoint': [],
            'LArmpitJoint': [],
            'RArmpitJoint': [],
            'LElbowJoint':  [],
            'RElbowJoint':  [],
            'LHipJoint':    [],
            'RHipJoint':    [],
            'LThighJoint':  [],
            'RThighJoint':  [],
            'LKneeJoint':   [],
            'RKneeJoint':   []
        }
        self.sequence_data['normalized_keypoints'] = []

        # joint angle sequences
        self.skeletons = []


    def add_keypoints(self, body_keypoints):
        self.skeletons.append(Skeleton(body_keypoints))


    def load_from_json(self, folder_name=None):
        sf = open(folder_name)
        self.sequence_data = json.load(sf)


    def create_sequence_data(self):
        for skeleton in self.skeletons:
            for key, values in skeleton.joint_angles.items():
                self.sequence_data['joint_angles'][key].append(values)

            self.sequence_data['normalized_keypoints'].append(skeleton.normalized_keypoints)


    def smoothen(self):

        for key, values in self.sequence_data['joint_angles'].items():

            values = medfilt(volume=values, kernel_size=json_config['kernel_size'])
            values = gaussian_filter(input=values, sigma=json_config['sigma'])

            self.sequence_data['joint_angles'][key] = list(values)


    def save_as_json(self, folder_name='no_action', webcam=False):
        if not webcam:
            action_dir = os.path.join(json_recordings_dir, folder_name)
        else:
            action_dir = os.path.join(webcam_recordings_dir, folder_name)

        file_name = folder_name +".json"

        with open(os.path.join(action_dir, file_name), 'w', encoding='utf-8') as write_file:
            json.dump(self.sequence_data, write_file, ensure_ascii=False, indent=4)

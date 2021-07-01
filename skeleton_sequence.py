from skeleton import Skeleton
import os
from datetime import datetime
import json

from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

ROOT_PATH = './'
json_recordings_dir = os.path.join(ROOT_PATH, 'json_recordings')

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

    def smoothen(self, kernel_size=3, sigma=1):

        for key, values in self.sequence_data['joint_angles'].items():

            values = medfilt(volume=values, kernel_size=kernel_size)
            values = gaussian_filter(input=values, sigma=sigma)

            self.sequence_data['joint_angles'][key] = list(values)


    def save_as_json(self, folder_name='no_action', sigma=1, filt_size=3, output=False):
        action_dir = os.path.join(json_recordings_dir, folder_name)
        file_name = "recording_"+ folder_name +"_{:%Y%m%dT%H%M%S}.json".format(datetime.now())

        if output:
            action_dir=ROOT_PATH
            file_name = 'output.json'

        for skeleton in self.skeletons:
            for key, values in skeleton.joint_angles.items():
                self.sequence_data['joint_angles'][key].append(values)


        with open(os.path.join(action_dir, file_name), 'w', encoding='utf-8') as write_file:
            json.dump(self.sequence_data, write_file, ensure_ascii=False, indent=4)

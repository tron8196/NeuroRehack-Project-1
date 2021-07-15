import os
import argparse

from skeleton_sequence import SkeletonSequence

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


skeleton_seq = SkeletonSequence()

parser = argparse.ArgumentParser(description='Visualize skeleton sequence data.')
parser.add_argument('--data' , help='Path to the json recording of an action.')
parser.add_argument('--sigma' , help='sigma value of the gaussian filter used for smoothening the signal')
parser.add_argument('--filt_size', help='Median Filter size')

args = parser.parse_args()
print(args)
assert args.data, "Argument --data is required for loading a recording from a json file."
skeleton_seq.load_from_json(args.data)


if args.filt_size is None:
    filt_size = 13
else:
    filt_size = int(args.filt_size)

if args.sigma is None:
    sigma = 0.1
else:
    sigma = float(args.sigma)


def smoothing(values):
    values = medfilt(volume=values, kernel_size=filt_size)
    values = gaussian_filter(input=values, sigma=sigma)

    return values


sns.set()

for key, values in skeleton_seq.sequence_data['joint_angles'].items():

    values = smoothing(values)
    # skeleton_seq.sequence_data['joint_angles'][key] = list(values)

    plt.plot(range(len(values)),
                        values)


upper_joint_angles = ['LNeckJoint', 'RNeckJoint', 'LArmpitJoint', 'RArmpitJoint',
'LElbowJoint', 'RElbowJoint','LHipJoint','RHipJoint']
no_of_joints = len(upper_joint_angles)
for key in upper_joint_angles:
    print(key + '::')
    print('Mean: ', np.average(skeleton_seq.sequence_data['joint_angles'][key]))
    print('Variance: ', round(np.var(skeleton_seq.sequence_data['joint_angles'][key]), 2))
    print("=====================================")


lower_joint_angles = ['LThighJoint', 'RThighJoint', 'LKneeJoint', 'RKneeJoint']
no_of_joints = len(lower_joint_angles)
for key in lower_joint_angles:
    print(key + '::')
    print('Mean: ', np.average(skeleton_seq.sequence_data['joint_angles'][key]))
    print('Variance: ', round(np.var(skeleton_seq.sequence_data['joint_angles'][key]), 2))
    print("=====================================")

plt.legend(skeleton_seq.sequence_data['joint_angles'].keys(), ncol=2, loc='upper right');
plt.show()
# plt.savefig

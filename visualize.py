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
    filt_size = 3
else:
    filt_size = int(args.filt_size)

if args.sigma is None:
    sigma = 1
else:
    sigma = float(args.sigma)


sns.set()

for key, values in skeleton_seq.sequence_data['joint_angles'].items():
    # removing NaNs
    values = [v for v in values if v==v]

    values = medfilt(volume=values, kernel_size=filt_size)
    values = gaussian_filter(input=values, sigma=sigma)

    skeleton_seq.sequence_data['joint_angles'][key] = values

    plt.plot(range(len(values)), values)


distance, _ = fastdtw(skeleton_seq.sequence_data['joint_angles']['LArmpitJoint'],
                        skeleton_seq.sequence_data['joint_angles']['RArmpitJoint'], dist=euclidean)

print('DTW distance : ', distance)

plt.legend(skeleton_seq.sequence_data['joint_angles'].keys(), ncol=2, loc='upper right');
plt.show()
# plt.savefig

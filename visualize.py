import argparse
from skeleton_sequence import SkeletonSequence
import seaborn as sns
import matplotlib.pyplot as plt

skeleton_seq = SkeletonSequence()

parser = argparse.ArgumentParser(description='Visualize skeleton sequence data.')
parser.add_argument('--data', help='Path to the json recording of an action.')
args = parser.parse_args()

assert args.data, "Argument --data is required for loading a recording from a json file."
skeleton_seq.load_from_json(args.data)

sns.set()

for key, values in skeleton_seq.sequence_data['joint_angles'].items():
    plt.plot(range(len(values)), values)

plt.legend(skeleton_seq.sequence_data['joint_angles'].keys(), ncol=2, loc='upper right');
plt.show()

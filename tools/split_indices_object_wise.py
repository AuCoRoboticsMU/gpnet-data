import os
import numpy as np
import math
from autolab_core import YamlConfig

general_config = YamlConfig('cfg/tools/config.yaml')
data_dir = '/data/{}'.format(general_config['dataset_name'])

files_in_dir = os.listdir('{}/tensors/'.format(data_dir))
numbers = [int(x.split('_')[-1][:-4]) for x in files_in_dir if 'labels_' in x]
numbers.sort()

train_size = 0.8
val_size = train_size + 0.1
test_size = 1 - val_size

object_labels = []
for nbr in numbers:
    object_label = np.load('{}/tensors/labels_{:07d}.npy'.format(data_dir, nbr))[1]
    object_labels.append(object_label)

object_labels = np.array(object_labels)
unique_labels = list(np.unique(object_labels))
np.random.shuffle(unique_labels)

train_indices = []
for label in unique_labels[:math.ceil(len(unique_labels) * train_size)]:
    train_indices.extend(list(np.where(object_labels == label)[0]))
train_indices.sort()
val_indices = []
for label in unique_labels[math.ceil(len(unique_labels) * train_size): math.ceil(len(unique_labels) * val_size)]:
    val_indices.extend(list(np.where(object_labels == label)[0]))
val_indices.sort()
test_indices = []
for label in unique_labels[math.ceil(len(unique_labels) * val_size):]:
    test_indices.extend(list(np.where(object_labels == label)[0]))
test_indices.sort()

np.savetxt('{}/val_indices.txt'.format(data_dir), val_indices, fmt='%d')
np.savetxt('{}/train_indices.txt'.format(data_dir), train_indices, fmt='%d')
np.savetxt('{}/test_indices.txt'.format(data_dir), test_indices, fmt='%d')

print('Done.')

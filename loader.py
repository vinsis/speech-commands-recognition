import torch
from torch.utils.data import Dataset

import glob
import librosa
import os
import numpy as np

root = 'speech_commands_v0.01'
test_file = 'testing_list.txt'
validation_file = 'validation_list.txt'

def find_classes(directory):
    def is_valid_class(d):
        return os.path.isdir(os.path.join(directory, d)) and not d.startswith('_')
    classes = [d for d in os.listdir(directory) if is_valid_class(d)]
    classes.sort()
    class_to_idx = { classes[i]: i for i in range(len(classes)) }
    return classes, class_to_idx

def get_file_names(classes):
    print('Preparing file names. This might take a while ...')
    validation_filenames, test_filenames, train_filenames = [],[],[]
    with open(os.path.join(root, validation_file)) as f:
        validation_filenames = [os.path.join(root, i) for i in f.read().strip().split('\n')]
    with open(os.path.join(root, test_file)) as f:
        test_filenames = [os.path.join(root, i) for i in f.read().strip().split('\n')]
    for c in classes:
        train_filenames.extend([i for i in glob.glob(os.path.join(root, c, '*.wav'))
            if i not in validation_filenames and i not in test_filenames])
    print('Number of samples:')
    print('Training set [{}]'.format(len(train_filenames)))
    print('Validation set [{}]'.format(len(validation_filenames)))
    print('Test set [{}]'.format(len(test_filenames)))
    return train_filenames, validation_filenames, test_filenames


class AudioLoader(Dataset):
    def __init__(self, file_list, classes, class_to_idx):
        self.file_list = file_list
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        x, sr = librosa.load(self.file_list[index])
        if len(x) < 22050:
            x = np.pad(x, (0, 22050 - len(x)), 'constant')
        x = torch.from_numpy(x)
        x = x.view(1,-1)
        y = self.class_to_idx[self.file_list[index].split('/')[1]]
        y = torch.LongTensor([y])
        return x,y

    def __len__(self):
        return len(self.file_list)

classes, class_to_idx = find_classes(root)
train_filenames, validation_filenames, test_filenames = get_file_names(classes)

train_data = AudioLoader(train_filenames, classes, class_to_idx)
validate_data = AudioLoader(validation_filenames, classes, class_to_idx)
test_data = AudioLoader(test_filenames, classes, class_to_idx)

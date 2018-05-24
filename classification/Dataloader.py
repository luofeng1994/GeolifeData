import tensorflow as tf
import random
import numpy as np
import cPickle as pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def readTxt(file_path):
    vector = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            vector.append([float(item) for item in line.split(',')])
    return vector

def padding(vector, num_steps, feature_dim):
    if len(vector) >num_steps:
        return vector[:num_steps]
    else:
        for _ in range(num_steps - len(vector)):
            vector.append([0.0]*feature_dim)
        return vector

class Dataloder():
    def __init__(self, args):
        self.is_training = args.is_training
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.classes_to_handle = set(args.classes.split(','))
        self.pointer = 0
        self.is_measured_data = args.is_measured_data
        if self.is_training == 'train':
            self.data_dir = args.train_data_dir
        elif self.is_training == 'test':
            self.data_dir = args.test_data_dir
        else:
            pass
        self.loadData()

    def loadData(self):
        self.files = []
        self.label_dict = {c: i for i, c in enumerate(self.classes_to_handle)}
        if self.is_measured_data:
            self.label_dict = {'walk': 3, 'car': 0, 'bike': 1, 'bus': 2}
        self.reverse_label_dict = dict(zip(self.label_dict.values(), self.label_dict.keys()))

        self.data = dict()
        self.data_label = dict()
        self.data_length = dict()

        for transport_mode in os.listdir(self.data_dir):
            if transport_mode in self.classes_to_handle:
                for file in os.listdir(os.path.join(self.data_dir, transport_mode)):
                    label = self.label_dict[transport_mode]
                    one_file = os.path.join(self.data_dir, transport_mode, file)
                    vector = readTxt(one_file)
                    self.files.append('{}_{}'.format(transport_mode, file))
                    self.data['{}_{}'.format(transport_mode, file)] = vector
                    self.data_label['{}_{}'.format(transport_mode, file)] = label
                    self.data_length['{}_{}'.format(transport_mode, file)] = len(vector)
        self.raw_num = len(self.files)
        self.batch_num = self.raw_num/self.batch_size+1
        self.num_classes = len(self.classes_to_handle)
        if self.is_measured_data:
            self.num_classes = 4
        self.feature_dim = len(vector[0])
        aaa = 1

    def reset(self):
        random.shuffle(self.files)
        self.pointer = 0
    def next_batch(self):
        batch_files = self.files[self.pointer: min(self.pointer + self.batch_size, len(self.files))]
        x = []
        y = []
        length = []
        for batch_file in batch_files:
            label = self.data_label[batch_file]
            y.append(label)
            vector = self.data[batch_file]
            length.append(self.data_length[batch_file])
            vector = padding(vector, self.num_steps, self.feature_dim)
            x.append(vector)
        x = np.array(x)
        x = x.astype(np.float32)
        y = np.array(y)
        y = y.astype(np.int32)
        length = np.array(length)
        length = length.astype(np.int32)
        self.pointer += self.batch_size
        return x, y, length
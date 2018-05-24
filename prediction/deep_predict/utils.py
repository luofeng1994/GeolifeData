import sys
sys.path.append("..")
import numpy as np
import cPickle as pickle
import cluster
import random


class Dataloader():
    def __init__(self, args):
        self.args = args
        if args['is_training'] == True:
            self.data_path = args['train_data_path']
        else:
            self.data_path = args['test_data_path']
        self.dbscanner_path = args['dbscanner_path']
        self.save_dir = args['save_dir']
        self.embedding_path = args['embedding_path']
        self.batch_size = args['batch_size']
        self.pointer = 0
        self.loadData()

    def loadData(self):
        dbscanner = pickle.load(open(self.dbscanner_path, 'r'))
        self.classes = dbscanner.k
        self.embedding = pickle.load(open(self.embedding_path, 'r'))
        self.feature_dim = len(self.embedding['-1'])

        self.data = []
        with open(self.data_path, 'r') as file:
            for line in file:
                data, label = line.split('|')
                data = data.strip()
                label = label.strip()
                data = data.split(' ')
                label = label.split(' ')[-1]
                self.data.append({'data':data, 'label':label})
        self.num_steps = len(self.data[0]['data'])
        self.data_length = len(self.data)
        self.batch_num = int(self.data_length / self.batch_size)

    def create_one_data(self, i):
        x = self.embedding[i]
        x = np.array(x)
        x = x.astype(np.float32)
        return x
    def create_input(self, datas):
        x = [self.create_one_data(i) for i in datas]
        x = np.array(x)
        x.astype(np.float32)
        x = x[np.newaxis, :]
        return x
    def next_batch(self):
        x = []
        y = []
        for item in self.data[self.pointer: self.pointer + self.batch_size]:
            data = item['data']
            label = item['label']
            a = [self.embedding[i] for i in data]
            f = lambda x: 0 if x == -1 else x
            # b = [f(int(j)) for j in label]
            b = f(int(label))
            x.append(a)
            y.append(b)
        x = np.array(x)
        y = np.array(y)
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        self.pointer = self.pointer + self.batch_size
        return x, y

    def reset(self):
        self.pointer = 0
        random.shuffle(self.data)


def main():
    args = dict()
    args['train_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_4numseqs_train.txt'
    args['test_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_4numseqs_test.txt'
    args['dbscanner_path'] = '../utils/dbscaner_100m_5minutes_50eps_5minPts.pkl'
    args['save_dir'] = './save'
    args['embedding_path'] = '../word2vec/variable/embedding.pkl'
    args['batch_size'] = 5
    args['is_training'] = True

    dataloader = Dataloader(args)
    dataloader.next_batch()

if __name__ == "__main__":
    main()
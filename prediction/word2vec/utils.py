from __future__ import division
import sys
sys.path.append("..")
import numpy as np
from collections import Counter
import math
import random
import cPickle as pickle

class Dataloader():
    def __init__(self, args):
        self.args = args
        self.corpus_path = args['corpus_path']
        self.neg_sample_size = args['neg_sample_size']
        self.dbscanner_path = args['dbscanner_path']
        self.batch_size = args['batch_size']
        self.alpha = args['alpha']
        self.table_size = args['table_size']
        self.min_frequency = args['min_frequency']
        self.train_data_path = args['train_data_path']
        self.min_lr = args['min_lr']
        self.lr = args['lr']
        self.epoches = args['epoches']
        self.pointer = 0
        self.build_vocab()
        # self.build_table()
        self.build_train_data()
        aaa = 1
    def build_vocab(self):
        dbscanner = pickle.load(open(self.dbscanner_path, 'r'))
        k = dbscanner.k
        self.words = dbscanner.m_reverse.keys()

        # with open(self.corpus_path) as f:
        #     words = [word for line in f.readlines() for word in line.split()]
        # self.counter = []
        # self.counter.extend([list(item) for item in Counter(words).most_common() if item[0] > self.min_frequency])
        # self.word2idx = dict()
        # for word, _ in self.counter:
        #     self.word2idx[word] = len(self.word2idx)
        self.word2idx = {str(i): c for c, i in enumerate(self.words)}
        self.vocab_size = len(self.word2idx)
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
        self.labels = np.zeros([self.batch_size, 1 + self.neg_sample_size], dtype=np.float32)
        self.labels[:,0] = 1
    def build_table(self):
        total_count_pow = 0
        for _, count in self.counter:
            total_count_pow += math.pow(count, self.alpha)
        word_idx = 0
        self.table = np.zeros([self.table_size], dtype=np.int32)
        aaa = self.counter[word_idx][1]
        word_prob = math.pow(self.counter[word_idx][1], self.alpha) / total_count_pow
        for idx in xrange(self.table_size):
            self.table[idx] = word_idx
            if idx / self.table_size > word_prob:
                word_idx += 1
                word_prob += math.pow(self.counter[word_idx][1], self.alpha) / total_count_pow
            if word_idx > self.vocab_size:
                word_idx = word_idx - 1
    def build_train_data(self):
        self.train_data = []
        with open(self.train_data_path, 'r') as file:
            for line in file:
                line = line.strip()
                team = line.split(' ')
                for i in range(len(team)-1):
                    if team[i] == '0':
                        continue
                    self.train_data.append([team[i], team[i+1]])
        self.batch_num = int(len(self.train_data)/self.batch_size)
        self.decay = (self.min_lr-self.lr)/(self.epoches * self.batch_num)
    def next_batch(self):
        x = []
        y = []
        for item in self.train_data[self.pointer: self.pointer + self.batch_size]:
            context = item[0]
            word = item[1]
            context_idx = self.word2idx[context]
            contexts = self.sample_contexts_random(context_idx)
            x.append(self.word2idx[word])
            y.append(contexts)
        self.pointer += self.batch_size
        x = np.array(x)
        y = np.array(y)
        aaa = 1
        return x, y
    def reset_pointer(self):
        self.pointer = 0
    def sample_contexts_random(self, context):
        contexts = np.ndarray(1 + self.neg_sample_size, dtype=np.int32)
        contexts[0] = context
        idx = 1
        while idx < self.neg_sample_size + 1:
            neg_context = random.randrange(self.vocab_size)
            if context != neg_context:
                contexts[idx] = neg_context
                idx += 1
        return contexts
    def sample_contexts(self, context):
        contexts = np.ndarray(1 + self.neg_sample_size, dtype=np.int32)
        contexts[0] = context
        idx = 1
        while idx < self.neg_sample_size + 1:
            neg_context = self.table[random.randrange(self.table_size)]
            if context != neg_context:
                contexts[idx] = neg_context
                idx += 1
        return contexts
def main():
    args = dict()
    args['corpus_path'] = '../vector_traing_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3windows.txt'
    args['train_data_path'] = '../corpus_train_test/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3windows_train.txt'
    args['test_data_path'] = '../corpus_train_test/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3windows_test.txt'
    args['batch_size'] = 5
    args['neg_sample_size'] = 3
    args['alpha'] = 0.75  # smooth out unigram frequencies
    args['table_size'] = 1000  # table size from which to sample neg samples
    args['min_frequency'] = 1  # threshold for vocab frequency
    dataloader = Dataloader(args)
    for i in range(dataloader.batch_num):
        print dataloader.next_batch()
if __name__ == '__main__':
    main()
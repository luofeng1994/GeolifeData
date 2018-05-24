import tensorflow as tf
import math
import numpy as np
class Model():
    def __init__(self, args):
        self.args = args
        self.neg_sample_size = args['neg_sample_size']
        self.batch_size = args['batch_size']
        # self.lr = args['lr']
        self.embed_size = args['embed_size']
        self.vocab_size = args['vocab_size']
        if args['sampling'] == True:
            self.batch_size = 1
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.int32, [self.batch_size], name='pos_x')
            self.y = tf.placeholder(tf.int32, [self.batch_size, 1 + self.neg_sample_size], name='pos_x')
            self.lr = tf.placeholder(tf.float32, name='lr')


        # init_width = 0.5 / self.embed_size
        with tf.name_scope('embed_a'):
            self.embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -0.5, 0.5), name='embed')
            # self.variable_summaries(self.embed, 'embed_a')
        with tf.name_scope('embed_b'):
            self.w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size)), name='w')
            # self.variable_summaries(self.w, 'embed_b')

        self.x_embed = tf.nn.embedding_lookup(self.embed, self.x, name='pos_embed')
        self.y_w = tf.nn.embedding_lookup(self.w, self.y, name='pos_embed')
        self.x_embed_reshape = tf.reshape(self.x_embed, [self.x_embed.shape[0].value, 1, self.x_embed.shape[1].value])
        self.mul = tf.matmul(self.x_embed_reshape, self.y_w, transpose_b=True)
        self.mul = tf.squeeze(self.mul,squeeze_dims=1)
        # self.p = tf.nn.sigmoid(self.mul)

        with tf.name_scope('loss_and_train'):
            self.loss = 0.0
            for i in range(self.mul.shape[0].value):
                self.loss -=  tf.log(tf.nn.sigmoid(self.mul[i][0]))
                for j in range(1, self.mul.shape[1].value):
                    self.loss -= tf.log(tf.nn.sigmoid(-self.mul[i][j]))
            tf.summary.scalar('loss', self.loss)

            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.p, labels = self.labels))
            self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        # writer = tf.summary.FileWriter('./log', tf.get_default_graph())
        # writer.close()
        self.merged = tf.summary.merge_all()

    def variable_summaries(self, var, name):
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
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
    args['lr'] = 0.025  # initial learning rate
    args['epochs'] = 3  # number of epochs to train
    args['embed_size'] = 100  # dimensionality of word embeddings
    args['sampling'] = False
    args['vocab_size'] = 1000
    model = Model(args)
if __name__ == '__main__':
    main()
from utils import Dataloader
from model import Model
import tensorflow as tf
import cPickle as pickle
import numpy as np
import time
def main():
    # args = dict()
    # args['corpus_path'] = '../vector_traing_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3windows.txt'
    # args['train_data_path'] = '../vector_traing_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3windows.txt'
    # args['batch_size'] = 1
    # args['neg_sample_size'] = 2
    # args['alpha'] = 0.75  # smooth out unigram frequencies
    # args['table_size'] = 1000  # table size from which to sample neg samples
    # args['min_frequency'] = 1  # threshold for vocab frequency
    # args['lr'] = 0.05
    # args['min_lr'] = 0.005
    # args['embed_size'] = 128
    # args['sampling'] = False
    # args['epoches'] = 50
    # args['save_every_n'] = 200
    # args['save_dir'] = './save'
    #
    # dataloader = Dataloader(args)
    # args['vocab_size'] = dataloader.vocab_size
    args = pickle.load(open('./variable/args.pkl', 'r'))
    dataloader = pickle.load(open('./variable/dataloader.pkl', 'r'))

    model = Model(args)
    embedding = dict()
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for word_idx in dataloader.idx2word.keys():
                word = dataloader.idx2word[word_idx]
                x = np.array([word_idx])
                feed = {model.x : x}
                x_embed = sess.run([model.x_embed], feed_dict=feed)
                x_embed = x_embed[0][0]
                embedding[word] = x_embed
                print '{}:{}'.format(word, x_embed)
    pickle.dump(embedding, open('./variable/embedding.pkl', 'w'))
if __name__ == '__main__':
    main()

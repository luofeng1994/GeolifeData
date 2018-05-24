from utils import Dataloader
from model import Model
import tensorflow as tf
import cPickle as pickle
import time
def main():
    args = dict()
    args['corpus_path'] = '../vector_traing_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_2windows.txt'
    args['train_data_path'] = '../vector_traing_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_2windows.txt'
    args['dbscanner_path'] = '../utils/dbscaner_100m_5minutes_50eps_5minPts.pkl'
    args['batch_size'] = 1
    args['neg_sample_size'] = 5
    args['alpha'] = 0.75  # smooth out unigram frequencies
    args['table_size'] = 1000  # table size from which to sample neg samples
    args['min_frequency'] = 1  # threshold for vocab frequency
    args['lr'] = 0.05
    args['min_lr'] = 0.005
    args['embed_size'] = 128
    args['sampling'] = False
    args['epoches'] = 70
    args['save_every_n'] = 200
    args['save_dir'] = './save_windowns{}'.format(args['neg_sample_size'])
    dataloader = Dataloader(args)
    args['vocab_size'] = dataloader.vocab_size
    pickle.dump(dataloader, open('./variable/dataloader.pkl', 'w'))
    pickle.dump(args, open('./variable/args.pkl', 'w'))

    model = Model(args)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log', sess.graph)
        count = 0
        for e in range(args['epoches']):
            dataloader.reset_pointer()
            for i in range(dataloader.batch_num):
                count += 1
                start = time.time()
                x, y = dataloader.next_batch()
                # labels = dataloader.labels
                args['lr'] = args['lr'] + dataloader.decay
                feed = {model.x: x,
                        model.y: y,
                        model.lr: args['lr']}
                summary, train, loss = sess.run([model.merged, model.train, model.loss], feed_dict=feed)
                summary_writer.add_summary(summary, count)
                end = time.time()
                if count % 100 == 0:
                    print('round_num: {}/{}... '.format(e + 1, args['epoches']),
                          'Training steps: {}... '.format(count),
                          'Training error: {:.4f}... '.format(loss),
                          'Learning rate: {:.4f}... '.format(args['lr']),
                          '{:.4f} sec/batch'.format((end - start)))
                if (count % args['save_every_n'] == 0):
                    saver.save(sess, "{path}/i{counter}.ckpt".format(path = args['save_dir'], counter=count))
        saver.save(sess, "{path}/i{counter}.ckpt".format(path=args['save_dir'], counter=count))
        summary_writer.close()
if __name__ == '__main__':
    main()

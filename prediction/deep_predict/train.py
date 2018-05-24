import tensorflow as tf
from model import Model
from utils import Dataloader
import cPickle as pickle
import time

def main():
    args = dict()
    args['train_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3numseqs_train.txt'
    args['test_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3numseqs_test.txt'
    args['dbscanner_path'] = '../utils/dbscaner_100m_5minutes_50eps_5minPts.pkl'
    args['save_dir'] = './save_3numsteps'
    args['embedding_path'] = '../word2vec/variable/embedding.pkl'
    args['batch_size'] = 5
    args['lstm_size'] = 128
    args['lstm_layer'] = 1
    args['weight_decay'] = 0.00001
    args['is_training'] = True
    args['keep_prob'] = 0.5
    args['lr'] = 0.001
    args['epochs'] = 70
    args['save_every_n'] = 200

    dataloader = Dataloader(args)
    args['num_steps'] = dataloader.num_steps
    args['feature_dim'] = dataloader.feature_dim
    args['classes'] = dataloader.classes

    pickle.dump(args, open('./utils/args.pkl', 'wb'))

    model = Model(args)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        count = 0
        for e in range(args['epochs']):
            dataloader.reset()
            # new_state = sess.run(model.initial_state)
            for i in range(dataloader.batch_num):
                count += 1
                x, y = dataloader.next_batch()
                start = time.time()
                feed = {model.input: x,
                        model.target: y}
                batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)
                end = time.time()
                if count % 100 == 0:
                    print('round_num: {}/{}... '.format(e + 1, args['epochs']),
                          'Training steps: {}... '.format(count),
                          'Training error: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (count % args['save_every_n'] == 0):
                    # aaa = 1
                    saver.save(sess, "{path}/i{counter}_l{lstm_size}.ckpt".format(path=args['save_dir'],
                                                                                  counter=count,
                                                                                  lstm_size=args['lstm_size']))
        saver.save(sess, "{path}/i{counter}_l{lstm_size}.ckpt".format(path=args['save_dir'],
                                                                      counter=count,
                                                                      lstm_size=args['lstm_size']))


if __name__ == "__main__":
    main()
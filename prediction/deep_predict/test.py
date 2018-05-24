from __future__ import division
import tensorflow as tf
from model import Model
from utils import Dataloader
import cPickle as pickle
import time


def main():
    args = pickle.load(open('./utils/args.pkl', 'r'))
    args['is_training'] = False
    # args['save_dir'] = './save'
    # args['train_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_5numseqs_train.txt'
    args['test_data_path'] = '../predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_3numseqs_train.txt'
    args['num_steps'] = 3

    dataloader = Dataloader(args)
    model = Model(args)

    count = 0
    count_correct = 0
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            for row in dataloader.data:
                saver.restore(sess, ckpt.model_checkpoint_path)
                new_state = sess.run(model.initial_state)
                datas = row['data']
                labels = int(row['label'])
                x = dataloader.create_input(datas)
                feed = {model.input: x,
                        model.initial_state: new_state}
                prediction = sess.run(model.prediction, feed_dict=feed)
                prediction = sess.run(tf.argmax(prediction, axis=1))
                prediction = prediction[0]

                count += 1
                if labels == -1:
                    labels = 0
                label_pred = prediction
                print '{}:{}->{}'.format(datas, labels, label_pred)
                if labels == label_pred:
                    count_correct += 1
    print 'totoal count:{}, correct count:{}, correct_rate:{}'.format(count, count_correct, count_correct/count)

if __name__ == '__main__':
    main()
from __future__ import division
import tensorflow as tf
from Dataloader import Dataloder
from model import Model
from labels import getTime
import time
import os
import logging
import numpy as np
import pandas as pd
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def appendRecord(args):
    record = pd.DataFrame([args])
    metrics_record_dir = args['metrics_record_dir']
    metrics_record_file = args['metrics_record_file']
    metrics_recprd_path = os.path.join(metrics_record_dir, metrics_record_file)
    if os.path.exists(metrics_recprd_path):
        record_old = pd.read_csv(metrics_recprd_path)
        record_new = pd.concat([record_old, record], axis=0)
        os.remove(metrics_recprd_path)
        record_new.to_csv(metrics_recprd_path, index=None)
    else:
        record.to_csv(metrics_recprd_path, index=None)

def getDate():
    return time.strftime("%Y-%m-%d_%H-%M", time.localtime()).split(' ')[0]
class Logger(object):
    def __init__(self, logger, fname=None, format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"):
        self.logFormatter = logging.Formatter(format)
        self.rootLogger = logger
        self.rootLogger.setLevel(logging.DEBUG)

        self.consoleHandler = logging.StreamHandler(sys.stdout)
        self.consoleHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)

        dir = os.path.split(fname)[0]
        if not os.path.exists(dir):
            os.mkdir(dir)
        if fname is not None:
            self.fileHandler = logging.FileHandler(fname)
            self.fileHandler.setFormatter(self.logFormatter)
            self.rootLogger.addHandler(self.fileHandler)

    def warn(self, message):
        self.rootLogger.warn(message)

    def info(self, message):
        self.rootLogger.info(message)

    def debug(self, message):
        self.rootLogger.debug(message)

    def setLevel(self, level):
        self.rootLogger.setLevel(level)

def train(args):
    logger = Logger(logging.getLogger(), os.path.join(args.log_dir, args.section, '{}_{}_{}.log'.format(args.section, args.is_training, getDate())))
    logger.info('start now!')
    if args.bi_directional and args.attention:
        raise Exception('when args.bi_derectional is {}, args.attention can not be {}'.format(args.bi_directional, args.attention))
    dataloader = Dataloder(args)
    args.num_classes = dataloader.num_classes
    args.feature_dim = dataloader.feature_dim
    logger.info('dataloder finished.')
    logger.info('num_classes: {}'.format(args.num_classes))
    logger.info('feature_dim: {}'.format(args.feature_dim))
    logger.info('num_layers: {}'.format(args.num_layers))
    logger.info('batch_size: {}'.format(args.batch_size))
    logger.info('batch_num: {}'.format(dataloader.batch_num))
    logger.info('epoches: {}'.format(args.epochs))
    logger.info('label dict: {}'.format(dataloader.label_dict))
    model = Model(args)
    logger.info('model build finished')
    logger.info('start training...')
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(args.tmp_dir, sess.graph)
        counter = 0
        for e in range(args.epochs):
            dataloader.reset()
            for _ in range(dataloader.batch_num):
                x, y, length = dataloader.next_batch()
                counter += 1
                start = time.time()
                feed = {model.input: x,
                        model.target: y,
                        model.input_length: length}
                batch_loss, batch_accuracy, _, summary = sess.run([model.loss, model.accuracy, model.optimizer, model.merged], feed_dict=feed)
                summary_writer.add_summary(summary, counter)
                end = time.time()
                # control the print lines
                if counter % 100 == 0:
                    logger.info('round_num: {}/{}... Training steps: {}... Training error: {:.4f}... Training accuracy: {:.4f}... {:.4f} sec/batch'.format(e + 1, args.epochs, counter, batch_loss, batch_accuracy, end-start))

                if (counter % args.save_every_n == 0):
                    saver.save(sess, "{path}/i{counter}_l{lstm_size}_bi({bi_directional})_attention({attention}).ckpt".format(path=args.save_dir, counter=counter,
                                                                                  lstm_size=args.lstm_size, bi_directional=args.bi_directional, attention=args.attention))
                # if counter == 400:
                #     break
        saver.save(sess, "{path}/i{counter}_l{lstm_size}_bi({bi_directional})_attention({attention}).ckpt".format(
            path=args.save_dir, counter=counter,
            lstm_size=args.lstm_size, bi_directional=args.bi_directional, attention=args.attention))
        summary_writer.close()
        if args.record:
            args.accuracy = batch_accuracy
            appendRecord(args.__flags)

def test(args):
    logger = Logger(logging.getLogger(), os.path.join(args.log_dir, args.section, '{}_{}_{}.log'.format(args.section, args.is_training, getDate())))
    if args.bi_directional and args.attention:
        raise Exception('when args.bi_derectional is {}, args.attention can not be {}'.format(args.bi_directional, args.attention))
    dataloader = Dataloder(args)
    args.num_classes = dataloader.num_classes
    args.feature_dim = dataloader.feature_dim
    model = Model(args)
    logger.info('num_classes: {}'.format(args.num_classes))
    logger.info('feature_dim: {}'.format(args.feature_dim))
    logger.info('num_layers: {}'.format(args.num_layers))
    logger.info('batch_size: {}'.format(args.batch_size))
    logger.info('batch_num: {}'.format(dataloader.batch_num))
    logger.info('epoches: {}'.format(args.epochs))
    logger.info('label dict: {}'.format(dataloader.label_dict))
    count = 0
    correct_count = 0

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for _ in range(dataloader.batch_num):
                x, y, length = dataloader.next_batch()
                predict = model.predict(sess, x, length)
                for i in range(predict.shape[0]):
                    record = '{}:{}'.format(dataloader.reverse_label_dict[y[i]], dataloader.reverse_label_dict[predict[i]])
                    logger.info(record)
                count += predict.shape[0]
                kkk = sum(y==predict)
                correct_count += sum(y==predict)
    kk = 'count:{}, correct_count{}, correct_rate:{}'.format(count, correct_count, correct_count/count)
    logger.info(kk)
    if args.record:
        args.accuracy = correct_count/count
        appendRecord(args.__flags)

def main(FLAGS):
    args = FLAGS
    if args.is_training == 'train':
        train(args)
    elif args.is_training == 'test':
        test(args)



def writeList(list_to_save, save_path):
    file_dir = os.path.split(save_path)[0]
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, 'a') as f:
        for item in list_to_save:
            if isinstance(item, basestring):
                f.write(item + "\n")
            else:
                pass

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 50, "batch_size")
    flags.DEFINE_integer("num_steps", 300, "num_steps")
    flags.DEFINE_integer("lstm_size", 128, "lstm_size")
    flags.DEFINE_integer("num_layers", 1, "num_layers")
    flags.DEFINE_integer("grad_clip", 5, "grad_clip")
    flags.DEFINE_integer("epochs", 50, "epochs")
    flags.DEFINE_integer("save_every_n", 200, "save_every_n")
    flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
    flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
    flags.DEFINE_float("minval", -0.5, "minval for random_uniform_initializer")
    flags.DEFINE_float("maxval", 0.5, "maxval for random_uniform_initializer")
    flags.DEFINE_float("stddev", 0.1, "stddev for random_norm_initializer or truncated_normal_initilizer")
    flags.DEFINE_string("train_data_dir", './Data_Geolife/train', "train_data_path")
    flags.DEFINE_string("test_data_dir", './Data_Geolife/test', "test_data_path")
    flags.DEFINE_string("classes", 'walk,bike,car,bus', "classes")
    flags.DEFINE_string("save_dir", 'save', "save_dir")
    flags.DEFINE_string("log_dir", 'log', "log_dir")
    flags.DEFINE_string("metrics_record_dir", 'metrics_record', "metrics_record_dir")
    flags.DEFINE_string("metrics_record_file", 'record.csv', "metrics_record_file")
    flags.DEFINE_string("tmp_dir", 'tmp', "tmp_dir")
    flags.DEFINE_string("initializer_mode", 'truncated_normal_initializer',
                        "initializer mode: random_uniform_initializer/ random_normal_initializer/ truncated_normal_initializer, default is truncated_normal_initializer")
    flags.DEFINE_string("rnn_mode", 'lstm', "rnn_mode: lstm/gru, default is lstm cell")
    flags.DEFINE_string("section", 'DNN', "section")
    flags.DEFINE_string("is_training", 'test', "train or test")
    flags.DEFINE_bool("bi_directional", False, "True")
    flags.DEFINE_bool("attention", False, "attention")
    flags.DEFINE_bool("is_measured_data", False, "is_measured_data")
    flags.DEFINE_bool("record", False, "whether to record the running result")
    FLAGS = flags.FLAGS
    main(FLAGS)

import tensorflow as tf
import numpy as np
from lstm_bn import BatchNormLSTMCell
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

class Model():
    def __init__(self, args):
        self.is_training = args.is_training
        self.num_steps = args.num_steps
        self.feature_dim = args.feature_dim
        self.lstm_size = args.lstm_size
        self.num_layers = args.num_layers
        self.lr = args.learning_rate
        self.grad_clip = args.grad_clip
        self.bi_directional = args.bi_directional
        self.attention = args.attention
        self.num_classes = args.num_classes
        self.minval = args.minval
        self.maxval = args.maxval
        self.stddev = args.stddev
        self.log_dir = args.log_dir
        self.initializer_mode = args.initializer_mode
        self.rnn_mode = args.rnn_mode
        self.pointer = 0
        if self.is_training == 'train':
            self.keep_prob = args.keep_prob
        elif self.is_training == 'test':
            self.keep_prob = 1.0
        else:
            pass

        self.input = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.feature_dim], name='input')
        self.target = tf.placeholder(tf.int32, shape=[None], name='target')
        self.input_length = tf.placeholder(tf.int32, shape=[None], name='input_length')
        self.build_variables()
        self.build_model()

    def build_variables(self):
        if self.initializer_mode == 'random_uniform_initializer':
            self.initializer = tf.random_uniform_initializer(minval=self.minval, maxval=self.maxval)
        elif self.initializer_mode == 'random_normal_initializer':
            self.initializer = tf.random_normal_initializer(stddev=self.seddev)
        elif self.initializer_mode == 'truncated_normal_initializer':
            self.initializer = tf.truncated_normal_initializer(stddev=self.stddev)
        else:
            self.initializer = tf.truncated_normal_initializer(stddev=self.stddev)

        if self.rnn_mode == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        elif self.rnn_mode =='gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell
        elif self.rnn_mode == 'bn_lstm':
            self.rnn_cell = BatchNormLSTMCell
        else:
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell

        self.zero_initializer = tf.zeros_initializer()
        with tf.variable_scope('full_connection'):
            self.fc_w = tf.get_variable(name='fc_weight', shape=[self.feature_dim, self.lstm_size], dtype=tf.float32, initializer=self.initializer)
            self.fc_b = tf.get_variable(name='fc_bias', shape=[self.lstm_size], dtype=tf.float32, initializer=self.zero_initializer)
            self.variable_summaries(self.fc_w, 'fc_w')
            self.variable_summaries(self.fc_b, 'fc_b')
        with tf.variable_scope('lstm_layer'):
            if self.bi_directional:
                if self.rnn_mode == 'bn_lstm':
                    cell_fw = self.rnn_cell(self.lstm_size, self.is_training)
                    cell_bw = self.rnn_cell(self.lstm_size, self.is_training)
                else:
                    cell_fw = self.rnn_cell(self.lstm_size)
                    cell_bw = self.rnn_cell(self.lstm_size)
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
                self.cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers)
                self.cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers)
                # self.initial_state_fw = self.cell_fw.zero_state(self.batch_size, dtype=tf.float32)
                # self.initial_state_bw = self.cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            else:
                if self.rnn_mode == 'bn_lstm':
                    cell = self.rnn_cell(self.lstm_size, self.is_training)
                else:
                    cell = self.rnn_cell(self.lstm_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)
                # self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

        if self.attention:
            with tf.variable_scope('attention'):
                self.attention_matrix = tf.get_variable(name='attention_matrix', shape=[self.lstm_size, self.lstm_size], initializer=self.initializer)
                self.variable_summaries(self.attention_matrix, 'attention_matrix')

        if self.attention or self.bi_directional:
            with tf.variable_scope('softmax_layer'):
                self.sm_w = tf.get_variable(name='softmax_weight', shape=[2 * self.lstm_size, self.num_classes],
                                            initializer=self.initializer)
                self.sm_b = tf.get_variable(name='softmax_bias', shape=[self.num_classes],
                                            initializer=self.zero_initializer)
                self.variable_summaries(self.sm_w, 'sm_w')
                self.variable_summaries(self.sm_b, 'sm_b')
        else:
            with tf.variable_scope('softmax_layer'):
                self.sm_w = tf.get_variable(name='softmax_weight', shape=[self.lstm_size, self.num_classes],
                                            initializer=self.initializer)
                self.sm_b = tf.get_variable(name='softmax_bias', shape=[self.num_classes], initializer=self.zero_initializer)
                self.variable_summaries(self.sm_w, 'sm_w')
                self.variable_summaries(self.sm_b, 'sm_b')

    def build_model(self):
        with tf.variable_scope('full_connection'):
            input_reshaped = tf.reshape(self.input, [-1, self.feature_dim])
            input_extend = tf.matmul(input_reshaped, self.fc_w) + self.fc_b
            tf.summary.histogram('full_connection_input/input_extend', input_extend)
            input_extend = tf.nn.relu(input_extend)
            input_act = tf.reshape(input_extend, [-1, self.num_steps, self.lstm_size])
        with tf.variable_scope('lstm_layer'):
            if self.bi_directional:
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                    self.cell_fw,
                    self.cell_bw,
                    input_act,
                    # initial_state_fw=self.initial_state_fw,
                    # initial_state_bw=self.initial_state_bw,
                    sequence_length=self.input_length,
                    dtype=tf.float32)
                outputs = tf.concat(outputs, 2)
                if self.rnn_mode == 'lstm':
                    final_state = tf.concat([final_state[0][-1][1], final_state[1][-1][1]], 1)
                elif self.rnn_mode == 'gru':
                    final_state = tf.concat([final_state[0][0], final_state[1][0]], 1)
                else:
                    final_state = tf.concat([final_state[0][-1][1], final_state[1][-1][1]], 1)
            else:
                outputs, final_state = tf.nn.dynamic_rnn(self.cell, input_act, dtype=tf.float32)
                if self.rnn_mode == 'lstm':
                    final_state = final_state[-1][1]
                elif  self.rnn_mode == 'gru':
                    final_state = final_state[-1]
                else:
                    final_state = final_state[-1][1]

        if self.attention:
            with tf.variable_scope('attention_layer'):
                outputs = self.attention_calcu(outputs, final_state)
        else:
            outputs = final_state
        with tf.variable_scope('softmax_layer'):
            self.logits = tf.matmul(outputs, self.sm_w) + self.sm_b
        with tf.variable_scope('loss'):
            self.loss = self.build_loss(self.logits, self.target)
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('accuracy'):
            self.prediction = tf.nn.softmax(self.logits, dim=1, name='prediction')
            kkk = tf.cast(tf.argmax(self.prediction, 1), tf.int32)
            correct_prediction = tf.equal(kkk, self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('acuracy', self.accuracy)
        with tf.variable_scope('optimizer'):
            self.optimizer = self.build_optimizer(self.loss)
        self.merged = tf.summary.merge_all()

    def variable_summaries(self, var, name):
        with tf.variable_scope('summaries'):
            tf.summary.histogram(name, var)
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev/' + name, stddev)

    def build_loss(self,logits, targets):
        # y_one_hot = tf.one_hot(targets, self.num_classes)
        # y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss
    def build_optimizer(self, loss):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.lr)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def attention_calcu(self, outputs, final_state):
        h_final = final_state
        output_split = tf.split(outputs, self.batch_size, axis=0)
        h_final_split = tf.split(h_final, self.batch_size, axis=0)
        score = [tf.matmul(tf.matmul(tf.squeeze(output_split[i], axis=0), self.attention_matrix), h_final_split[i], transpose_b=True) for i in range(self.batch_size)]
        score = tf.squeeze(tf.stack(score), 2)
        score_softmax = tf.nn.softmax(score, 1)
        score_softmax = tf.expand_dims(score_softmax, 2)
        attention = tf.matmul(tf.transpose(outputs, perm=[0,2,1]), score_softmax)
        attention = tf.squeeze(attention,axis=2)
        h_attention = tf.concat([h_final, attention], 1)
        return h_attention
    def predict(self, sess, x, length):
        feed = {
            self.input: x,
            self.input_length: length
        }
        probs = sess.run(self.prediction, feed_dict=feed)
        probs = np.squeeze(probs)
        results = np.argmax(probs, 1)
        return results

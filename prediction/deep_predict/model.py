import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
    def __init__(self, args):
        if not args['is_training']:
            self.batch_size = 1
            self.keep_prob = 1.0
        else:
            self.batch_size = args['batch_size']
            self.keep_prob = args['keep_prob']
        self.save_dir = args['save_dir']
        self.lstm_size = args['lstm_size']
        self.lstm_layer = args['lstm_layer']
        self.is_training = args['is_training']
        self.lr = args['lr']
        self.num_steps = args['num_steps']
        self.feature_dim = args['feature_dim']
        self.weight_decay = args['weight_decay']
        self.classes = args['classes']

        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps, self.feature_dim), name='inputs')
        # self.target = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        self.target = tf.placeholder(tf.int32, shape=(self.batch_size), name='targets')


        # with tf.variable_scope('relu'):
        #     w = tf.Variable(tf.truncated_normal([self.feature_dim, self.lstm_size], stddev=0.1))
        #     b = tf.Variable(tf.zeros(self.lstm_size))
        #     input_1 = tf.reshape(self.input, [-1, self.feature_dim])
        #     input_2 = tf.matmul(input_1, w) + b
        #     input_2_relu = tf.nn.relu(input_2)
        #     input_reshaped = tf.reshape(input_2_relu, [self.batch_size, self.num_steps, self.lstm_size])


        lstm = rnn.BasicLSTMCell(self.lstm_size, forget_bias=0.5)
        drop = rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        self.cell = rnn.MultiRNNCell([drop for _ in range(self.lstm_layer)])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # outputs, final_state = tf.nn.dynamic_rnn(self.cell, input_reshaped, initial_state=self.initial_state)
        outputs, final_state = tf.nn.dynamic_rnn(self.cell, self.input, dtype=tf.float32)
        self.final_state = final_state

        # seq_output = tf.concat(outputs, axis=1)  # tf.concat(concat_dim, values)
        # reshape
        outputs = outputs[:,-1,:]
        # x = tf.reshape(outputs, [self.batch_size, -1])

        with tf.variable_scope('full_connection'):
            w = tf.Variable(tf.truncated_normal([self.lstm_size, self.classes], stddev=0.1), name='weight')
            b = tf.Variable(tf.zeros(self.classes), name='bias')
        logits = tf.matmul(outputs, w) + b
        self.prediction = tf.nn.softmax(logits=logits, name='softmax')

        y_one_hot = tf.one_hot(self.target, self.classes)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    def getL2Reg(self):
        l2 = sum(
            tf.contrib.layers.l2_regularizer(self.weight_decay)(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)
        )
        return l2

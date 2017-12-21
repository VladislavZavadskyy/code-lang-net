import tensorflow as tf
from data import *

data_type = tf.float32

class LangNet:
    def __init__(self, lang_input, is_training=True):
        config = lang_input.config
        self._is_training = is_training
        self._input = lang_input
        self._rnn_params = None
        self._cell = None
        self.batch_size = config.batch_size
        self.out_classes = lang_input.num_classes
        self.num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_placeholder = tf.placeholder(tf.int32,[self.batch_size, None],"input_plch")
        self.label_placeholder = tf.placeholder(tf.int32,[self.batch_size, None],"label_plch")
        self.mask = tf.placeholder(tf.float32, [self.batch_size, None])
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type)
        self.inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)

#         if is_training and config.keep_prob < 1:
#             inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_lstm(self.inputs, config, self._is_training)

        softmax_w = tf.get_variable("softmax_w", [size, self.out_classes], dtype=data_type)
        softmax_b = tf.get_variable("softmax_b", [self.out_classes], dtype=data_type)
        logits = tf.matmul(output,softmax_w)+softmax_b

        logits = tf.reshape(logits, [self.batch_size, -1, self.out_classes])
        self.logits = logits
        self.softmax = tf.nn.softmax(logits,2,"out_softmax")
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = self.label_placeholder,
            logits = self.logits
        )
        self.masked_loss = tf.multiply(self.loss,self.mask)

        self._cost = tf.reduce_sum(self.masked_loss)
        tf.summary.scalar("Loss",self._cost)
        self._final_state = state

        if not is_training: return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.RMSPropOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(self.grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.summaries = tf.summary.merge_all()

    def _build_lstm(self, inputs, config, is_training):
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, 
                forget_bias=1.0, state_is_tuple=True, reuse=not is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, 
                            output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type)
        state = self._initial_state

        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                   initial_state=self._initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def cost(self):
        return self._cost
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op

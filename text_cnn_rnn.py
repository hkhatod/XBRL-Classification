#pylint: disable=I0011
#pylint: disable=C0111
#pylint: disable=C0301
#pylint: disable=C0304
#pylint: disable=C0103
#pylint: disable=W0312
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard import summary as summary_lib
from attention import attention

class TextCNNRNN(object):
	def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
	             num_classes, embedding_size, filter_sizes, num_filters, tx_labels=None, l2_reg_lambda=0.0):

		with tf.name_scope("Placeholders"):
			self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
			self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
			self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
			self.batch_size = tf.placeholder(tf.int32, [])
			self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
			self.seqlen = tf.placeholder(tf.int32, [None], name='real_len')
			self.text_labels = tx_labels
			l2_loss = tf.constant(0.0)
			
		with tf.device('/cpu:0'), tf.name_scope('Emb'):
			# self.emb_var = tf.Variable(embedding_mat, name='emb_var')
			if not non_static:
				self.emb_var = tf.constant(embedding_mat, name='emb_var')
			else:
				self.emb_var = tf.Variable(embedding_mat, name='emb_var', trainable=True)
			self.embedded_chars = tf.nn.embedding_lookup(self.emb_var, self.input_x)
		
		##########
		# Bidirectional LSTM layer
		with tf.name_scope("biDRNN_GRU"):
			GRU_fw_cell = tf.nn.rnn_cell.GRUCell(hidden_unit)
			GRU_bw_cell = tf.nn.rnn_cell.GRUCell(hidden_unit)
			GRU_fw_cell = tf.contrib.rnn.DropoutWrapper(GRU_fw_cell, output_keep_prob=self.dropout_keep_prob)
			GRU_bw_cell = tf.contrib.rnn.DropoutWrapper(GRU_bw_cell, output_keep_prob=self.dropout_keep_prob)

			self.GRU_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
				GRU_fw_cell, 
				GRU_bw_cell, 
				self.embedded_chars, 
				sequence_length=self.seqlen, 
				dtype=tf.float32,
				scope='biDRNN')
			#lstm_outputs_fw, lstm_outputs_bw = tf.split(value=self.lstm_outputs, axis=2, num_or_size_splits=2)
			#self.lstm_outputs = tf.add(lstm_outputs_fw, lstm_outputs_bw, name="lstm_outputs")
			self.GRU_outputs = tf.concat(self.GRU_outputs, axis=2, name="GRU_outputs")

		with tf.name_scope("Attention"):
			attention_output, alphas = attention(self.GRU_outputs, hidden_unit, 128, sequence_length)
			# # # self.scores = tf.layers.dense(attention_output, num_classes, activation=None)
			# # # self.predictions = tf.argmax(self.scores, 1, name='Predited_Class')
			# # # self.currect_ans = tf.argmax(self.input_y, 1, name='True_Class')

		#########


		
		with tf.name_scope('Expand'):
			self.emb = tf.expand_dims(attention_output, -1)
			# # # self.emb = tf.expand_dims(self.embedded_chars, -1)
			
		with tf.name_scope("CNN"):
			pooled_concat = []
			# # # reduced = np.int32(np.ceil((sequence_length) * 1.0 / max_pool_size))
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope('conv-mxpl-%s' % filter_size):

					# # # with tf.name_scope("Padding"):
					# # # 	''' Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel '''
					# # # 	num_prio = (filter_size-1) // 2
					# # # 	num_post = (filter_size-1) - num_prio
					# # # 	pad_prio = tf.concat(values=[self.pad] * num_prio, axis=1)
					# # # 	pad_post = tf.concat(values=[self.pad] * num_post, axis=1)
					# # # 	emb_pad = tf.concat(values=[pad_prio, self.emb, pad_post], axis=1)

					filter_shape = [filter_size, hidden_unit*2, 1, num_filters]
					with tf.name_scope('weights'):
						W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
					with tf.name_scope('baises'):
						b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
					# # # conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
					conv = tf.nn.conv2d(self.emb, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

					h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
					''' Maxpooling over the outputs '''
					pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
					# # # pooled = tf.reshape(pooled, [-1, reduced, num_filters])
					pooled_concat.append(pooled)
					
			num_filters_total = num_filters * len(filter_sizes)
			pooled_concat = tf.concat(values=pooled_concat, axis=3)
			pooled_concat = tf.reshape(pooled_concat, [-1, num_filters_total])
			pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob, name="CNN_Dropout")
			
		with tf.name_scope('output'):
			# # # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W')
			W = tf.get_variable(
                "W", 
                shape=[num_filters_total, num_classes], 
                initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(pooled_concat, W, b, name='Scores')
			self.predictions = tf.argmax(self.scores, 1, name='Predited_Class')
			self.currect_ans = tf.argmax(self.input_y, 1, name='True_Class')

		with tf.name_scope("Evaluation_Metrics"):
			with tf.name_scope('Probabilities'):
				self.probabilities = tf.nn.softmax(self.scores)

			with tf.name_scope('Confidence'):
				self.conf = tf.reduce_max(self.probabilities, reduction_indices=[1])
		
			with tf.name_scope('Low_confidence'):
				self.conf_low = tf.reduce_min(self.conf, name='low_conf')
		
			with tf.name_scope('Avg_confidence'):
				self.Avg_conf = tf.reduce_mean(self.conf, name='confidence')

			with tf.name_scope('high_confidence'):
				self.conf_high = tf.reduce_max(self.conf, name='high_conf')

			with tf.name_scope('loss'):
				losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y, logits=self.scores, pos_weight=100)
				# losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores) #  only named arguments accepted
				self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

			with tf.name_scope('accuracy'):
				correct_predictions = tf.equal(self.predictions, self.currect_ans)
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

			with tf.name_scope('num_correct'):
				correct = tf.equal(self.predictions, self.currect_ans)
				self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))

			# with tf.name_scope('confusion_matrix'):
			# 	self.batch_confusion = tf.confusion_matrix(self.currect_ans, self.predictions, num_classes=num_classes, name='Confusion_txt')
			# 	self.confusion_matrix = tf.as_string(self.batch_confusion)
			# 	self.confusion_var = tf.Variable( tf.zeros([num_classes,num_classes],dtype=tf.int32 ),name='confusion_img' )
			# 	self.confusion_update = self.confusion_var.assign(self.confusion_var + self.batch_confusion)
			# 	self.confusion_image = tf.reshape( tf.cast(self.confusion_var, tf.float32), [1, num_classes, num_classes, 1])


		# if self.text_labels is not None:
		# 	for i in range(num_classes):
		# 		with tf.name_scope('%s' % self.text_labels[i]):
		# 			_, self.update_op = summary_lib.pr_curve_streaming_op('pr_curve', 
		# 																	predictions=self.probabilities[:,i], 
		# 																	labels=tf.cast(self.input_y,tf.bool)[:,i], 
		# 																	num_thresholds=self.batch_size, 
		# 																	metrics_collections='pr' ,
		# 																	display_name='n - ' + self.text_labels[i])
#pylint: disable=I0011
#pylint: disable=C0111
#pylint: disable=C0301
#pylint: disable=C0304
#pylint: disable=C0103
#pylint: disable=W0312
#pylint: disable=W0105
#pylint: disable=C0330
#pylint: disable=E0611
#pylint: disable=E1129
#pylint: disable=E1101
#pylint: disable=
#pylint: disable=
#pylint: disable=

import os
import sys
import json
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard import summary as summary_lib
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import data_helper
from text_cnn_rnn import TextCNNRNN

'''
python3 ./code/predict.py 'small_test/AssetsCurrent/' 'SFP/small_test/AssetsCurrent_dataset.csv'
'''
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype=np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels):
	if test_file.endswith('.pickle'):
		df = pd.read_pickle(test_file, compression='gzip')
	elif test_file.endswith('.csv'):
		df = pd.read_csv(test_file, sep=',', quotechar='"')
	else:
		logging.critical('File formate not supported.')

	#df = df.head(100000)
	df['element'] = df['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
	select = ['element']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'category' in df.columns:
		select.append('category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()
	if 'documentation' in df.columns:
		select.append('documentation')
		df['documentation'] = df['documentation'].replace('\n','', regex=True)
		
	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	#logging.warning('df.head(5) = ' + str(df.head(5)))
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def predict_unseen_data():
	trained_dir = './training/pickles/standard and documentation/training_sets/SFP/' +  sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = './training/pickles/standard and documentation/training_sets/' + sys.argv[2]

	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, y_, df = load_test_data(test_file, labels)
	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)
	directory, file = os.path.split(test_file)
	foldername = os.path.splitext(file)[0]
	logging.warning('foldername: ' + foldername)
	i = 1
	predicted_dir = trained_dir +'predicted_results_' + foldername + '/'
	if os.path.exists(predicted_dir):
		while os.path.exists(trained_dir +'predicted_results_%s%s' %(foldername, str(i))):
			i += 1
		logging.warning('i = '+ str(i))
		foldername = foldername+str(i)
		logging.warning('foldername = '+ foldername)
	predicted_dir = trained_dir +'predicted_results_' + foldername + '/'
	os.makedirs(predicted_dir)

	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(embedding_mat=embedding_mat,
			                     non_static=params['non_static'],
                                 hidden_unit=params['hidden_unit'],
			                     sequence_length=len(x_test[0]),
			                     max_pool_size=params['max_pool_size'],
			                     filter_sizes=map(int, params['filter_sizes'].split(",")),
			                     num_filters=params['num_filters'],
			                     num_classes=len(labels),
			                     embedding_size=params['embedding_dim'],
			                     tx_labels=labels,
								 l2_reg_lambda=params['l2_reg_lambda'])

			# ''' PR summaries '''	
			# pr_summary_op = tf.summary.merge_all()
			# pr_summary_dir = os.path.join(predicted_dir, "summaries", "pr")
			# pr_summary_writer = tf.summary.FileWriter(pr_summary_dir, sess.graph)
						
			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch, y_batch=None):
				if y_batch is not None:
					feed_dict = {cnn_rnn.input_x: x_batch,
								cnn_rnn.input_y:y_batch,
								cnn_rnn.dropout_keep_prob: 1.0,
								cnn_rnn.batch_size: len(x_batch),
								cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
								cnn_rnn.real_len: real_len(x_batch),}
				
				else:
					feed_dict = {cnn_rnn.input_x: x_batch,
								cnn_rnn.dropout_keep_prob: 1.0,
								cnn_rnn.batch_size: len(x_batch),
								cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
								cnn_rnn.real_len: real_len(x_batch),}
					
				
				
				predictions, outputs, scores, probabilities, confidences = sess.run([cnn_rnn.predictions, cnn_rnn.output, cnn_rnn.scores, cnn_rnn.probabilities, cnn_rnn.conf], feed_dict)
				sess.run(tf.local_variables_initializer())
				# _, pr_summaries = sess.run([cnn_rnn.update_op, pr_summary_op],feed_dict)
				# pr_summary_writer.add_summary(pr_summaries)				
				return predictions, outputs, scores, probabilities, confidences

			logging.warning(trained_dir)
			checkpoint_dir = trained_dir + params['runname']
			logging.warning('checkpoint_dir : {}'.format(checkpoint_dir))
			checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir) + '.meta'
			logging.warning(checkpoint_file)
			saver = tf.train.Saver(tf.global_variables())
			saver = tf.train.import_meta_graph(checkpoint_file)
			saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
			logging.critical('{} has been loaded'.format(checkpoint_file))

			predictions, predict_labels, scores, outputs, probabilities,predict_prob = [], [], [], [], [],[]
			if y_ is not None:
				batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
				for batch in batches:
					x_batch, y_batch = zip(*batch)
					batch_predictions, batch_outputs, batch_scores, batch_probabilities, batch_confidences = predict_step(x_batch,y_batch)
					for batch_prediction in batch_predictions:
						predictions.append(batch_prediction)
						predict_labels.append(labels[batch_prediction])
					for batch_output in batch_outputs:
						outputs.append(batch_output)
					for batch_score in batch_scores:
						scores.append(batch_score)
					for batch_probability in batch_probabilities:
						probabilities.append(batch_probability)
					for batch_confidence in batch_confidences:
						predict_prob.append(batch_confidence)

			else:
				batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
				for x_batch in batches:
					batch_predictions, batch_outputs, batch_scores, batch_probabilities, batch_confidences = predict_step(x_batch)
					for batch_prediction in batch_predictions:
						predictions.append(batch_prediction)
						predict_labels.append(labels[batch_prediction])
					for batch_output in batch_outputs:
						outputs.append(batch_output)
					for batch_score in batch_scores:
						scores.append(batch_score)
					for batch_probability in batch_probabilities:
						probabilities.append(batch_probability)
					for batch_confidence in batch_confidences:
						predict_prob.append(batch_confidence)

			
			# for index, probablity in enumerate(probabilities):
			# 	predict_prob.append(probablity[predictions[index]])

			np_outputs =  np.array(outputs)
			

			df['Predicted'] = predict_labels
			#df['max_score'] = predictions
			#df['scores'] = scores
			#df['probablities'] =  probabilities
			#df['true_prop'] =  y_
			df['Confidence_old'] = predict_prob
			df['Confidence'] = [round(x*100,2) for x in predict_prob]
			columns = sorted(df.columns, reverse=True)
			if y_ is not None:
				df['Errors'] = np.where(df['category'] == df['Predicted'], 'Positive', 'Negetive')
			df.to_csv(predicted_dir + '/' + foldername +'_metadata.tsv', sep='\t',index=False, line_terminator='\n', quotechar='"', doublequote=True)
			np.savetxt(predicted_dir + '/' + foldername + '_Viz_data.tsv',np_outputs,delimiter='\t',newline='\n')	
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
			
			if y_test is not None:
				y_test_arg = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test_arg) / float(len(y_test_arg))
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

			session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
			sess = tf.Session(config=session_conf)
			embedding_var = tf.Variable(np_outputs, name=foldername + 'predict_viz')
			saver_embed = tf.train.Saver([embedding_var])
			sess.run(embedding_var.initializer)
			config = projector.ProjectorConfig()
			config.model_checkpoint_path = predicted_dir  + 'viz' +'.ckpt'
			emb_writer = tf.summary.FileWriter(predicted_dir,sess.graph)
			embedding = config.embeddings.add() 
			embedding.metadata_path = foldername + '_metadata.tsv'
			embedding.tensor_name = embedding_var.name
			projector.visualize_embeddings(emb_writer, config)
			saver_embed.save(sess, predicted_dir + 'viz' +'.ckpt')
			#print_tensors_in_checkpoint_file(predicted_dir +'viz' +'.ckpt',tensor_name='',all_tensors=True)	

	''' PR summaries '''
	probabilities = np.array(probabilities)
	pr_graph = tf.Graph()
	with pr_graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		pr_sess = tf.Session(config=session_conf)
		with pr_sess.as_default():
			for cat in range(y_test.shape[1]):
				with tf.name_scope('Predicted_%s' % labels[cat]):
					_, update_op = summary_lib.pr_curve_streaming_op('pr_curve',
																		predictions=probabilities[:, cat],
																		labels=tf.cast(y_test[:, cat], tf.bool),
																		num_thresholds=500,
																		metrics_collections='pr')
			pr_summary_op = tf.summary.merge_all()
			pr_summary_dir = os.path.join(predicted_dir, "summaries", "pr")
			pr_summary_writer = tf.summary.FileWriter(pr_summary_dir, pr_sess.graph)
			pr_sess.run(tf.local_variables_initializer())
			pr_sess.run([update_op])
			pr_summary_writer.add_summary(pr_sess.run(pr_summary_op))

if __name__ == '__main__':
	predict_unseen_data()

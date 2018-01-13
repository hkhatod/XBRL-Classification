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
#pylint: disable=W1202
#pylint: disable=
#pylint: disable=

'''

Run :
python3 ./code/train.py 'small_test/AssetsCurrent.pickle'
'''

import os
import io
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorboard import summary as summary_lib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.python import debug as tf_debug
from textwrap import wrap
import tfplot
from PIL import Image
import tfplot
import matplotlib
import re
#import matplotlib.pyplot as plt
from text_cnn_rnn import TextCNNRNN
import data_helper
# E1101:Module 'numpy' has no 'float32' member
# W0105:W0105:String statement has no effect
# C0330:Wrong continued indentation
# E0611:No name 'tensorboard' in module 'LazyLoader'
# E1129:Context manager 'generator' doesn't implement __enter__ and __exit__.

'''
To implement:
1. Implement reading pretrained embeddings.
3. NCE weights.
'''


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def load_trained_params(trained_dir):
	# params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	# words_index = json.loads(open(trained_dir + 'words_index.json').read())
	# labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype=np.float32)
	return embedding_mat

def gather_all(batchs, labels=None):
	items = []
	label_items = []
	if labels is None:
		for batch in batchs:
			items.append(batch)
		return items
	else:
		for batch in batchs:
			items.append(batch)
			label_items.append(labels[batch])
		return items, label_items


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=True):
	''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
	cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
	if normalize:
		cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
		cm = np.nan_to_num(cm, copy=True)
		cm = cm.astype('int')
	
	np.set_printoptions(precision=2)
	###fig, ax = matplotlib.figure.Figure()
	
	fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)
	im = ax.imshow(cm, cmap='Oranges')
	
	classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
	classes = ['\n'.join(wrap(l, 40)) for l in classes]

	tick_marks = np.arange(len(classes))

	ax.set_xlabel('Predicted', fontsize=12 if len(classes) < 10 else 4)
	ax.set_xticks(tick_marks)
	c = ax.set_xticklabels(classes, fontsize=8 if len(classes) < 10 else 4, rotation=-90,  ha='center')
	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()
	
	ax.set_ylabel('True Label', fontsize=12 if len(classes) < 10 else 7)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize=8 if len(classes) < 10 else 4, va ='center')
	ax.yaxis.set_label_position('left')
	ax.yaxis.tick_left()

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=10 if len(classes) < 10 else 6 , verticalalignment='center', color= "black")
	fig.set_tight_layout(True)
	summary = tfplot.figure.to_summary(fig, tag=tensor_name)
	return summary



def train_cnn_rnn():
	path = './training/pickles/standard and documentation/training_sets/SFP/'
	base_dir = path + sys.argv[1] +'/'
	#base_dir = path + 'Base AssetsCurrent/'
	for f in os.listdir(base_dir):
		if f.endswith(".pickle"):
			input_file = base_dir + f

	# try:
	# 	training_config = path + sys.argv[2]
	# 	params = json.loads(open(training_config).read())
	# 	params['continue_training']=True
	# except IndexError:
	#training_config = './code/training_config.json'
	training_config = base_dir + 'training_config.json'
	params = json.loads(open(training_config).read())
	params['continue_training'] = False

	directory, file = os.path.split(input_file)
	foldername = os.path.splitext(file)[0]
	runname =  'do:' + str(params['dropout_keep_prob']) + ' ed:' + str(params['embedding_dim'])+ \
			 ' fs:' + params['filter_sizes']  +' hu:'+ str(params['hidden_unit']) + ' l2:'+ \
			 str(params['l2_reg_lambda'])+ ' mxps:' + str(params['max_pool_size']) + ' ep:'+ str(params['num_epochs'])


	if params['continue_training']:
		''' Continue training...'''
		checkpoint_dir = os.path.dirname(training_config)
		checkpoint_dir = checkpoint_dir + '/' +params['runname'] +'/'
		i = str(params['folder_suffix'])
	else:
		checkpoint_dir = directory +'/'+ 'CNN_RNN_' +  foldername + '/'
		''' i is a folder increment varaiable. Its also used to update the name of tsv file.'''
		i = 1
		if os.path.exists(checkpoint_dir):
			while os.path.exists(directory  +'/'+ 'CNN_RNN_'+  foldername  + str(i) + '/'):
				i += 1
				''' dont del i as emb_viz is using for incremnting '''
			checkpoint_dir = directory  +'/'+  'CNN_RNN_' +foldername + str(i) + '/' + runname + '/'
		else:
			''' This del is OK '''
			del i
			i = ''
			checkpoint_dir = directory  +'/'+ 'CNN_RNN_' + foldername + '/' + runname + '/'
		params['folder_suffix'] = str(i)
		os.makedirs(checkpoint_dir)
		
	checkpoint_prefix = os.path.join(checkpoint_dir, foldername)

	x_, y_, vocabulary, vocabulary_inv, vocabulary_count, df, labels = data_helper.load_data(input_file)

	'''
	Assign a embedding_dim dimension vector to each word
	'''
	if params['continue_training']:
		embedding_mat = load_trained_params(os.path.dirname(training_config)+'/')
	else:
		word_embeddings = data_helper.load_embeddings(vocabulary, params['embedding_dim'])
		embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
		embedding_mat = np.array(embedding_mat, dtype=np.float32)
	'''
	Split the original dataset into train set and test set
	'''
	t_sz = 0.1
	if len(x_) > 100000:
		t_sz = 10000/len(x_)
	indices = np.arange(len(x_))
	x, x_test, y, y_test, ind, ind_test = train_test_split(x_, y_, indices, test_size=t_sz)
	x_train, x_dev, y_train, y_dev, ind_train, ind_dev = train_test_split(x, y, ind, test_size=t_sz)
	logging.warning('y_train.shape[1] is : {}'.format(y_train.shape[1]))
	params['sequence_length'] = x_train.shape[1]
	params['runname'] = runname
	with open(os.path.dirname(os.path.dirname(checkpoint_dir)) + '/trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
	'''
	use folder name as subscript instead of 'model'
	'''

	'''
	Save trained parameters and files since predict.py needs them
	'''
	with open(os.path.dirname(os.path.dirname(checkpoint_dir))  + '/words_index.json', 'w') as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(os.path.dirname(os.path.dirname(checkpoint_dir))  + '/embeddings.pickle', 'wb') as outfile:
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(os.path.dirname(os.path.dirname(checkpoint_dir))  + '/labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)
	'''
	Emdeddings labels not trained
	'''
	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(embedding_mat=embedding_mat, sequence_length=x_train.shape[1],
                                 num_classes=y_train.shape[1], non_static=params['non_static'],
                                 hidden_unit=params['hidden_unit'], max_pool_size=params['max_pool_size'],
                                 filter_sizes=map(int, params['filter_sizes'].split(",")),
                                 num_filters=params['num_filters'], embedding_size=params['embedding_dim'],
                                 tx_labels=labels, l2_reg_lambda=params['l2_reg_lambda']
								 )
			#optimizer = tf.train.AdamOptimizer()
			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
			grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			''' Keep track of gradient values and sparsity (optional) '''
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)

			''' Output directory for models and summaries '''
			print("Writing to {}\n".format(checkpoint_dir))

			''' Summaries for loss and accuracy '''
			loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
			acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)
			conf_low_summary = tf.summary.scalar("confidence_low", cnn_rnn.conf_low, collections='confidence_low')
			conf_summary = tf.summary.scalar("confidence", cnn_rnn.Avg_conf, collections='confidence')
			conf_high_summary = tf.summary.scalar("confidence_high", cnn_rnn.conf_high, collections='confidence_high')
			logging.warning('conf high summay : {}'.format(conf_high_summary))

			''' Train Summaries '''
			train_summary_op = tf.summary.merge([loss_summary, acc_summary, conf_summary, conf_low_summary, conf_high_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(checkpoint_dir, "s", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

			''' Dev summaries '''
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary, conf_summary, conf_low_summary, conf_high_summary, grad_summaries_merged])
			dev_summary_dir = os.path.join(checkpoint_dir, "s", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

			''' Test summaries '''
			test_summary_dir = os.path.join(checkpoint_dir, "s", "test")
			test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
			checkpoint_viz_prefix = os.path.join(test_summary_dir, foldername)
			
			saver = tf.train.Saver(tf.global_variables())

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def train_step(x_batch, y_batch):
				feed_dict = {cnn_rnn.input_x: x_batch,
                             cnn_rnn.input_y: y_batch,
                             cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
                             cnn_rnn.batch_size: len(x_batch),
                             cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                             cnn_rnn.real_len: real_len(x_batch),}
				_, step, summaries,_ = sess.run([train_op, global_step, train_summary_op, cnn_rnn.confusion_update], feed_dict)
				#_, step, predicts, corr_anws, summaries,_ = sess.run([train_op, global_step, cnn_rnn.predictions, cnn_rnn.currect_ans, train_summary_op, cnn_rnn.confusion_update], feed_dict)
				train_summary_writer.add_summary(summaries, step)
				# sess.run(tf.local_variables_initializer())
				# _, step, pr_summaries = sess.run([cnn_rnn.update_op, global_step, pr_summary_op], feed_dict)
				# pr_summary_writer.add_summary(pr_summaries, step)
				# return predicts, corr_anws
			

			def dev_step(x_batch, y_batch):
				feed_dict = {cnn_rnn.input_x: x_batch,
                             cnn_rnn.input_y: y_batch,
                             cnn_rnn.dropout_keep_prob: 1.0,
                             cnn_rnn.batch_size: len(x_batch),
                             cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                             cnn_rnn.real_len: real_len(x_batch),}
				step,predicts, corr_anws, summaries, accuracy, num_correct = sess.run([global_step, cnn_rnn.predictions, cnn_rnn.currect_ans, dev_summary_op, cnn_rnn.accuracy, cnn_rnn.num_correct], feed_dict)
				dev_summary_writer.add_summary(summaries, step)
				return accuracy, num_correct,predicts, corr_anws

			def test_step(x_batch, y_batch):
				feed_dict = {cnn_rnn.input_x: x_batch,
                             cnn_rnn.input_y: y_batch,
                             cnn_rnn.dropout_keep_prob: 1.0,
                             cnn_rnn.batch_size: len(x_batch),
                             cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                             cnn_rnn.real_len: real_len(x_batch),}
				predicts, corr_anws, otpts, confs, n_crt, probs = sess.run([cnn_rnn.predictions, cnn_rnn.currect_ans, cnn_rnn.scores, cnn_rnn.conf, cnn_rnn.num_correct, cnn_rnn.probabilities], feed_dict)
				return predicts, corr_anws, otpts, confs, n_crt, probs

				#print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(checkpoint_dir), tensor_name='', all_tensors=True)

			sess.run(tf.global_variables_initializer())
			logging.critical('Training Started')


			''' Training starts here '''
			num_batches = (int(len(x_train) / params['batch_size']) + 1)* params['num_epochs']
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			''' Train the model with x_train and y_train '''
			logging.critical('Train the model with x_train and y_train')
			for train_batch in train_batches:
				x_train_batch, y_train_batch = zip(*train_batch)
				train_step(x_train_batch, y_train_batch)
				#batch_predictions, batch_correct_anws = train_step(x_train_batch, y_train_batch)
				current_step = tf.train.global_step(sess, global_step)
				# _, l_t = gather_all(batch_predictions, labels)
				# _, cl_t = gather_all(batch_correct_anws, labels)
				# # Compute confusion matrix
				# img_tr_summary = plot_confusion_matrix(cl_t, l_t, labels, tensor_name='train/cm')
				# img_tr_summary_writer.add_summary(img_tr_summary, current_step)

				''' Evaluate the model with x_dev and y_dev '''
				if current_step % params['evaluate_every'] == 0:
					dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					predictions, predict_labels, correct_anws, correct_labels = [],[],[],[]
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						acc, num_dev_correct, batch_predictions, batch_correct_anws = dev_step(x_dev_batch, y_dev_batch)
						total_dev_correct += num_dev_correct
						p, l = gather_all(batch_predictions, labels)
						predictions = predictions + p
						predict_labels = predict_labels +l
						
						c, l = gather_all(batch_correct_anws, labels)
						correct_anws = correct_anws + c
						correct_labels = correct_labels + l
						
					# Compute confusion matrix
					img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm', normalize=True)
					dev_summary_writer.add_summary(img_d_summary, current_step)
					

				
				
					accuracy = float(total_dev_correct) / len(y_dev)
					logging.info('Calculated - Accuracy on dev set: {}'.format(accuracy))
					logging.info('Model-Accuracy on dev set: {}'.format(acc))
					if accuracy >= best_accuracy:
						best_accuracy, best_at_step = accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=global_step)
						#path = saver.save(sess, checkpoint_prefix +str(current_step) +'.ckpt')
						#logging.critical('Saved model {} at step {} of total step {}'.format(path, best_at_step, num_batches))
						logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
					logging.critical('....................................Completed {} steps of total {} steps. {} % Completed.'.format(current_step, num_batches,int(current_step/num_batches*100)))

			logging.critical('Training is complete, testing the best model on x_test and y_test')
			''' Evaluate x_test and y_test '''
			saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
			#saver.restore(sess, checkpoint_prefix + str(best_at_step) +'.ckpt')
			test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
			total_test_correct = 0
			predictions, predict_labels, correct_anws, correct_labels, outputs, confidences, probs = [], [], [], [], [], [], []
			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				batch_predictions, batch_correct_anws, batch_outputs, batch_confidences, num_test_correct, batch_probs = test_step(x_test_batch, y_test_batch)
				total_test_correct += int(num_test_correct)
				
				p, l = gather_all(batch_predictions, labels)
				predictions = predictions + p
				predict_labels = predict_labels +l
				
				c, l = gather_all(batch_correct_anws, labels)
				correct_anws = correct_anws + c
				correct_labels = correct_labels + l

				outputs = outputs + gather_all(batch_outputs)
				confidences = confidences + gather_all(batch_confidences)
				probs = probs + gather_all(batch_probs)

			probs = np.array(probs)
			y_test = np.array(y_test)

			df_meta = pd.DataFrame(columns=['Predicted', 'Category', 'Confidence', 'Element'])
			df_meta['Element'] = pd.concat([df['element_name'][ind_test]], ignore_index=True).replace('\n', '-n')
			#df_meta['element'] = pd.concat([df['element'][ind_test]], ignore_index=True).replace('\n', '', regex=True)
			df_meta['Predicted'] = predict_labels
			df_meta['Category'] = correct_labels
			df_meta['Confidence'] = confidences[1]
			df_meta['Errors'] = np.where(df_meta['Category'] == df_meta['Predicted'], 'Positive', 'Negetive')
			np_test_outputs = np.array(outputs)
			df_meta.to_csv(test_summary_dir + '/' + foldername +'_metadata.tsv', sep='\t', index=False, line_terminator='\n', quotechar='"', doublequote=True)

			lst = zip(vocabulary_inv, vocabulary_count)
			tsv_df = pd.DataFrame.from_records(lst, columns=['Label', 'Count'])
			tsv_df.to_csv(test_summary_dir + '/metadata.tsv', sep='\t', columns=['Label', 'Count'], index=False)


			logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
			df_meta.to_pickle(test_summary_dir + '/' + foldername +'_metadata.pickle', compression='gzip')
			output_var = tf.Variable(np_test_outputs, name=foldername + 'predict_viz')
			sess.run(output_var.initializer)

			final_embed_matrix = sess.run(cnn_rnn.emb_var)
			embedding_var = tf.Variable(final_embed_matrix, name='embedding_viz' + str(i))
			saver_embed = tf.train.Saver([embedding_var, output_var])
			sess.run(embedding_var.initializer)
			config = projector.ProjectorConfig()
			config.model_checkpoint_path = test_summary_dir + '/' + foldername + str(best_at_step)+'viz' +'.ckpt'
			
			embedding = config.embeddings.add()
			embedding.metadata_path = foldername + '_metadata.tsv'
			embedding.tensor_name = output_var.name

			embedding = config.embeddings.add()
			embedding.metadata_path = 'metadata.tsv'
			embedding.tensor_name = embedding_var.name

			projector.visualize_embeddings(test_summary_writer, config)
			saver_embed.save(sess, checkpoint_viz_prefix + str(best_at_step)+'viz' +'.ckpt')
			#print_tensors_in_checkpoint_file(checkpoint_viz_prefix + str(best_at_step)+'viz' +'.ckpt', tensor_name='', all_tensors=True)

			# Compute confusion matrix
			img_summary = plot_confusion_matrix(correct_labels, predict_labels, labels,tensor_name='test/cm', normalize=False)
			test_summary_writer.add_summary(img_summary)

			


	''' PR summaries and Confusion Matrix '''
	pr_graph = tf.Graph()
	with pr_graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		pr_sess = tf.Session(config=session_conf)
		with pr_sess.as_default():
			for cat in range(y_train.shape[1]):
				with tf.name_scope('%s' % labels[cat]):
					_, update_op = summary_lib.pr_curve_streaming_op('pr_curve', predictions=probs[:, cat],	labels=tf.cast(y_test[:, cat], tf.bool), num_thresholds=500, metrics_collections='pr')
			pr_summary_op = tf.summary.merge_all()
			pr_sess.run(tf.local_variables_initializer())
			pr_sess.run([update_op])
			test_summary_writer.add_summary(pr_sess.run(pr_summary_op))
			test_summary_writer.close()


	''' Result summaries accross all runs '''
	result = '\n'+ foldername + ',' + str(params['documentation']) + ',' + str(params['standard_element']) + ',' + str(params['standard_ngrams']) + ',' + \
			str(params['custom_elements']) + ',' + str(params['custom_ngrams']) + ',' + str(y_train.shape[1]) + ',' + str(len(x_)) + ',' + str(params['num_epochs']) + ',' + \
			str(params['batch_size']) + ',' + str(params['dropout_keep_prob']) + ',' +  str(params['embedding_dim']) +',' +  '"'+str(params['filter_sizes'])+'"'+',' +\
			str(params['hidden_unit']) +',' + str(params['l2_reg_lambda']) + ',' + str(params['max_pool_size']) + ',' + str(params['non_static']) + ',' +\
			str(params['num_filters']) + ',' + str(float(total_test_correct)/len(y_test))

	fd = open('Result_Summary.csv', 'a')
	fd.write(result)
	fd.close()









if __name__ == '__main__':
	train_cnn_rnn()

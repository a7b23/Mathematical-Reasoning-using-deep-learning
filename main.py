import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from inputPipeline import *
import os

def get_initial_lstm(features,hidden_size):
	net = features
	with tf.variable_scope('initial_lstm'):
		with slim.arg_scope([slim.fully_connected],weights_initializer = tf.contrib.layers.xavier_initializer(),activation_fn = tf.nn.tanh) :
			
			c = slim.fully_connected(net,hidden_size)
			h = slim.fully_connected(net,hidden_size)

			return c,h

def decode_lstm(x, h, vocab_size, hidden_size, M, dropout=False, reuse=False)	:
	initializer = tf.contrib.layers.xavier_initializer()
	with tf.variable_scope('logits', reuse=reuse):
		w_h = tf.get_variable('w_h', [hidden_size, M], initializer=initializer)
		b_h = tf.get_variable('b_h', [M], initializer=initializer)
		

		if dropout:
			h = tf.nn.dropout(h, 0.5)
		h_logits = tf.matmul(h, w_h) + b_h

		h_logits += x
		h_logits = tf.nn.tanh(h_logits)
		out_logits = slim.fully_connected(h_logits,vocab_size,activation_fn = None)
		
		return out_logits



def getModel(img,txt,labels,batch_size) :
	net = img
	print net.get_shape()
	with tf.variable_scope('img_encoder') :

		with slim.arg_scope([slim.conv2d],weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn = tf.nn.relu,
													weights_regularizer=slim.l2_regularizer(0.0005)):
							
			net = slim.conv2d(net,kernel_size=[5,5], num_outputs=32, stride=[2,2],padding='VALID',scope = 'conv1')
			
			net = slim.conv2d(net,kernel_size = [5,5],num_outputs = 32,stride = [1,1],padding = 'SAME',scope = 'conv2')
			
			net = slim.conv2d(net,kernel_size=[5,5], num_outputs=64, stride=[2,2],padding='VALID',scope = 'conv3')
			
			net = slim.conv2d(net,kernel_size=[4,4],num_outputs=64, stride=[1,1],padding='VALID',scope = 'conv4')
			
			net = slim.conv2d(net,kernel_size=[3,3],num_outputs=64, stride=[2,2],padding='VALID',scope = 'conv5')
			
			img_features = net
			print(img_features.get_shape())	
	with tf.variable_scope('text_encoder') :

		txt_unrolled = tf.unstack(txt,txt.get_shape()[1],1)
		
		gru_cell = tf.contrib.rnn.GRUCell(64)
		outputs,_ = tf.contrib.rnn.static_rnn(gru_cell,txt_unrolled,dtype=tf.float32)
		
		txt_features = outputs[-1]

	
	merged = []
	for i in range(batch_size) :
		merged.append(tf.nn.conv2d(img_features[i:i+1],tf.reshape(txt_features[i:i+1],[1,1,-1,1]),strides = [1,1,1,1],padding = 'VALID'))
	merged = tf.stack(merged)
		
	
	merged = slim.flatten(merged)
	
	print merged.get_shape()
	hidden_size = 128
	T = 4
	M = 32
	vocab_size = 17
	dropout = True

	mask_in = labels[:,:T]
	mask_out = labels[:, 1:]
	x = tf.to_float(mask_in)

	x = slim.fully_connected(x, M, scope='fc')

	loss = 0.0
	with tf.variable_scope('decoder') :
		c,h = get_initial_lstm(merged,hidden_size)
		for t in range(T):
			if t == 0 :
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True,reuse = False)
			else :
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True,reuse = True)
			with tf.variable_scope('lstm'):
				_, (c, h) = lstm_cell(inputs = x[:,t,:], state=[c, h])


				logits = decode_lstm(x[:,t,:], h,vocab_size,hidden_size,M, dropout=dropout, reuse=(t!=0))
				
				loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = mask_out[:, t]))
					

	return loss/tf.to_float(batch_size)		


save_dir = './models/'
train_file = 'merged_train.txt'	
batch_size = 128

if not os.path.exists(save_dir) :
	os.makedirs(save_dir)

img,text_batch,labels_batch = prepareExampleListQueue(train_file,batch_size = batch_size,num_threads=4)


text_batch = tf.one_hot(text_batch,21)

labels_batch = tf.one_hot(labels_batch,17)

loss = getModel(img,text_batch,labels_batch,batch_size) + tf.add_n(slim.losses.get_regularization_losses())

train_step = tf.train.AdamOptimizer(learning_rate = 1.5e-4).minimize(loss)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config = config)

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

saver = tf.train.Saver(max_to_keep=5)

step = 0
while 1 :
	_,losses = sess.run([train_step,loss])
	if step%100 == 0 :
		print('after %d steps the loss is %g'%(step,losses))
	if step%20000 == 0 and step > 0:
		saver.save(sess,save_dir + 'model_'+str(step))
		print 'models saved'	
	step+=1
	
	
coord.request_stop()

coord.join(threads)

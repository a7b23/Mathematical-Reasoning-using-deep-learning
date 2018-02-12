import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

def get_initial_lstm(features,hidden_size):
	net = features
	with tf.variable_scope('initial_lstm'):
		with slim.arg_scope([slim.fully_connected],weights_initializer = tf.contrib.layers.xavier_initializer(),activation_fn = tf.nn.tanh) :
			
			c = slim.fully_connected(net,hidden_size)
			h = slim.fully_connected(net,hidden_size)

			return c,h

def decode_lstm(x, h, vocab_size, hidden_size, M, dropout=False, reuse=False)   :
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

hidden_size = 128
T = 4
M = 32
vocab_size = 17
dropout = False
batch_size = 1


def getModel(img,txt,batch_size) :
	net = img
	with tf.variable_scope('img_encoder') :

		with slim.arg_scope([slim.conv2d],weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn = tf.nn.relu,
													weights_regularizer=slim.l2_regularizer(0.0005)):
							
			net = slim.conv2d(net,kernel_size=[5,5], num_outputs=32, stride=[2,2],padding='VALID',scope = 'conv1')
			
			net = slim.conv2d(net,kernel_size = [5,5],num_outputs = 32,stride = [1,1],padding = 'SAME',scope = 'conv2')

			net = slim.conv2d(net,kernel_size=[5,5], num_outputs=64, stride=[2,2],padding='VALID',scope = 'conv3')
			
			net = slim.conv2d(net,kernel_size=[4,4],num_outputs=64, stride=[1,1],padding='VALID',scope = 'conv4')
			
			net = slim.conv2d(net,kernel_size=[3,3],num_outputs=64, stride=[2,2],padding='VALID',scope = 'conv5')
			
			img_features = net

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
	

	initial_inp = tf.constant([12],dtype = tf.int32)
	initial_inp = tf.one_hot(initial_inp,vocab_size,dtype = tf.float32)

	x = slim.fully_connected(initial_inp, M, scope='fc')

	loss = 0.0
	total_logits = []
	with tf.variable_scope('decoder') :
		c,h = get_initial_lstm(merged,hidden_size)
		for t in range(1):
			
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True,reuse = False)
			with tf.variable_scope('lstm'):
				_, (c, h) = lstm_cell(inputs = x, state=[c, h])


				logits = decode_lstm(x, h,vocab_size,hidden_size,M, dropout=dropout, reuse=(t!=0))
				total_logits.append(tf.argmax(logits,axis = 1))

				
	
	for t in range(T-1) :
		next_input = tf.one_hot(tf.argmax(logits,1),vocab_size,dtype = tf.float32)
		next_input = slim.fully_connected(next_input,M,scope = 'fc',reuse = True)
		with tf.variable_scope('decoder',reuse = True) :
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True,reuse = True)
			with tf.variable_scope('lstm'):
				_, (c, h) = lstm_cell(inputs = next_input, state=[c, h])


				logits = decode_lstm(next_input, h,vocab_size,hidden_size,M, dropout=dropout, reuse=True)
				total_logits.append(tf.argmax(logits,axis = 1))

	total_logits = tf.stack(total_logits,axis = 1)            

	return total_logits   
	

x = tf.placeholder(shape = [1,70,70,3],dtype = tf.float32)
text_b = tf.placeholder(shape = [1,8],dtype = tf.int32)
text_batch = tf.one_hot(text_b,21)

out = getModel(x,text_batch,batch_size)


config=tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config = config)

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver(max_to_keep=5)
ckpt = tf.train.get_checkpoint_state('models') 
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, ckpt.model_checkpoint_path)

step = 0
f = open('merged_test.txt','r')
data = f.readlines()
count = 0
total = len(data)
for idx,line in enumerate(data) :
	img = np.array(Image.open(line.split(',')[0]))
	img = np.reshape(img,[1,70,70,3])
	labels_actual = []
	text_actual = []
	val = line.split(',')
	for i in range(1,9) :
		text_actual.append(int(val[i]))
	for i in range(9,len(val)) :
		labels_actual.append(int(val[i]))

	text_actual = np.reshape(np.array(text_actual),[1,-1])    
	labels_actual = np.reshape(np.array(labels_actual),[1,-1])
	model_out = sess.run(out,feed_dict = {x : img,text_b : text_actual})
	
	
	for i in range(4) :
		if not labels_actual[0][i+1] == model_out[0][i] :
			count+=1
			break	
	  
	if idx%100 == 0 :
		print('%d steps reached'%idx)
	

print('the accuracy is %g'%(float(total - count)/float(total)))
	

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

maxSentenceLength = 8

def tokenizeSentence(sentence,word_dict) :
    words = sentence.split(' ')
    arr = []
    for word in words :
        arr.append(word_dict[word.lower()])
    for i in range(maxSentenceLength - len(words)) :
        arr.append(word_dict['<unk>']) 
    return np.array(arr)

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
    hidden_size = 128
    T = 4
    M = 32
    vocab_size = 17
    dropout = False

    initial_inp = tf.constant([12],dtype = tf.int32)
    initial_inp = tf.one_hot(initial_inp,17,dtype = tf.float32)

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

                

                
    
    for t in range(3) :
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
    
batch_size = 1

x = tf.placeholder(shape = [1,70,70,3],dtype = tf.float32)
text_b = tf.placeholder(shape = [1,8],dtype = tf.int32)
text_batch = tf.one_hot(text_b,21)

out = getModel(x,text_batch,batch_size)


config=tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config = config)

saver = tf.train.Saver(max_to_keep=5)
ckpt = tf.train.get_checkpoint_state('models') 
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

step = 0
f = open('merged_test.txt','r')
data = f.readlines()
count = 0
total = len(data)
input_vocab = ['What','Sum','Units','Tens','Max','Value','digit','given','is','of','in','x','y','+','-','?','>','<','odd','even','<unk>']
output_vocab = ['0','1','2','3','4','5','6','7','8','9','True','False','<s>','</s>','<append>','+','-']

word_dict = {}
for idx,word in enumerate(input_vocab) :
    word_dict[word.lower()] = idx

img = np.array(Image.open('type8/1000.png'))
img = np.reshape(img,[1,70,70,3])

questions = ["What is x ?","What is y ?","What is x + y ?","What is x - y ?","Is x > y ?","Is x < y ?","Is x odd ?","Is y odd ?",
            "Is x even ?","Is y even ?","Units digit of x ?","Units digit of y ?","Tens digit of x ?","Tens digit of y ?",
            "Max digit in x ?","Max digit in y ?","Max digit in x + y ?","Max digit in x - y ?","Is value of x given ?",
            "Is value of y given ?"]

for question in questions :

	text_actual = tokenizeSentence(question,word_dict)

	text_actual = np.reshape(np.array(text_actual),[1,-1])   
	model_out = sess.run(out,feed_dict = {x : img,text_b : text_actual})
	answer = ""
	for i in range(4) :
    		if output_vocab[model_out[0][i]] == "</s>" :
        		break
    		else :
        		answer+=output_vocab[model_out[0][i]]

	print question
	print answer            




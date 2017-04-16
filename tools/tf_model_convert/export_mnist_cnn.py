import tensorflow as tf
import numpy as np
import os
import random
import cv2
import codecs
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data

std_width = 28
std_height = 28
classes = 10


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
	
def build_cnn(x,keep_prob):
	#NHWC format
	x_image = tf.reshape(x, [-1,std_height,std_width,1])
	
	#conv-pool 1x28x28 -> 6x14x14
	W_conv1 = weight_variable([3, 3, 1, 6])      
	b_conv1 = bias_variable([6])       		
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool(h_conv1)

	#conv-pool 6x14x14 -> 12x7x7
	W_conv2 = weight_variable([3, 3, 6, 12])
	b_conv2 = bias_variable([12])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool(h_conv2)
	
	#conv-pool 12x7x7 -> 24x4x4
	W_conv3 = weight_variable([3, 3, 12, 24])
	b_conv3 = bias_variable([24])
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool(h_conv3)
	
	#NHWC(tensorflow) -> NCHW(easyCNN)
	h_trans = tf.transpose(h_pool3, [0,3,1,2])
	h_pool_flat = tf.reshape(h_trans, [-1, 4 * 4 * 24])	
	
	W_fc1 = weight_variable([4 * 4 * 24, 512])
	b_fc1 = bias_variable([512])		
	h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([512, classes])
	b_fc2 = bias_variable([classes])
	y_predict=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	probs = tf.nn.softmax(y_predict)
	
	params = (W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2)
	return y_predict,params
	
def train(train_XY,validate_XY,model_path):
	epochs = 100	
	batch_size = 128
	if train_XY.num_examples < batch_size:
		batch_size = train_XY.num_examples
	val_per_steps = 1000
	learning_rate = 0.1	
	
	keep_prob = tf.placeholder(tf.float32)
	lr = tf.placeholder(tf.float32)
	x = tf.placeholder(tf.float32, [None,std_width*std_height])
	predict,_ = build_cnn(x,keep_prob)
	y = tf.placeholder(tf.float32, [None,classes])
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
	train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())	
		
		for epoch in range(epochs):				
			steps = int(train_XY.num_examples / batch_size)
			for step in range(steps):
				batch_X,batch_Y = train_XY.next_batch(batch_size)	
				if batch_X is None or batch_Y is None:
					break
				_,train_loss,train_acc = sess.run([train_step,cost,accuracy], feed_dict={x:batch_X,y:batch_Y,lr:learning_rate,keep_prob:0.5})
				print('Epoch[%d] Step: %d/%d train_loss: %.4f train_acc: %.4f%% learning_rate:%.5f' % (epoch,step,steps,train_loss,train_acc*100.0,learning_rate))
				if step % val_per_steps == 0:
					val_batch_size = min(128,validate_XY.num_examples)
					val_steps = int(validate_XY.num_examples / val_batch_size)
					val_loss = 0.0
					val_acc = 0.0
					for val_step in range(val_steps):
						val_batch_X,val_batch_Y = validate_XY.next_batch(val_batch_size)	
						if val_batch_X is None or val_batch_Y is None:
							break					
						cur_val_loss,cur_val_acc = sess.run([cost,accuracy], feed_dict={x:val_batch_X,y:val_batch_Y,keep_prob:1.0})						
						val_loss = val_loss + cur_val_loss
						val_acc = val_acc + cur_val_acc
					val_loss = val_loss/val_steps
					val_acc = val_acc/val_steps
					print('Epoch[%d] Step: %d/%d val_loss: %.4f val_acc: %.4f%%' % (epoch,step,steps,val_loss,val_acc*100.0))
			learning_rate = max(learning_rate * 0.8,0.001)
			saver.save(sess, model_path)
		saver.save(sess, model_path)
	
def test(test_XY,model_path):
	batch_size = 2
	if test_XY.num_examples < batch_size:
		batch_size = test_XY.num_examples
	keep_prob = tf.placeholder(tf.float32)
	x = tf.placeholder(tf.float32, [None,std_width*std_height])
	predict,params = build_cnn(x,keep_prob)
	y = tf.placeholder(tf.float32, [None,classes])	
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
	correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,model_path)
		test_steps = int(test_XY.num_examples / batch_size)
		test_loss = 0.0
		test_acc = 0.0
		for test_step in range(test_steps):
			test_batch_X,test_batch_Y = test_XY.next_batch(batch_size)
			if test_batch_X is None or test_batch_Y is None:
				break					
			cur_test_loss,cur_test_acc = sess.run([cost,accuracy], feed_dict={x:test_batch_X,y:test_batch_Y,keep_prob:1.0})						
			test_loss = test_loss + cur_test_loss
			test_acc = test_acc + cur_test_acc
			'''
			layer_data = params[10].eval(feed_dict={x:test_batch_X,y:test_batch_Y,keep_prob:1.0})
			print(layer_data.shape)
			if len(layer_data.shape) == 4:
				layer_data = layer_data.transpose([0,3,1,2])
				for i in range(layer_data.shape[0]):
					print(layer_data[i][0][:1])
			elif len(layer_data.shape) == 2:
				for i in range(layer_data.shape[0]):
					print(layer_data[i][:100])
			break
			'''
		test_loss = test_loss/test_steps
		test_acc = test_acc/test_steps
		print('test_loss: %.4f test_acc: %.4f%%' % (test_loss,test_acc*100.0))		
			
def export_input(f,channel,width,height):
	f.write('InputLayer %d %d %d\n' % (channel,width,height))
	
def export_layer(f,name):
	f.write(name + '\n')
	
def export_conv(f,conv_weight,conv_bias,padding_type):
	#ConvolutionLayer 64 64 3 3 1 1 1 -0.0533507
	print('export conv layer.')
	print(conv_weight.shape)			
	print(conv_bias.shape)	
	#hwcn -> nchw
	#3x3x1x32 -> 32x1x3x3
	conv_weight = np.transpose(conv_weight,[3,2,0,1])
	oc,ic,kw,kh,sw,sh,bias = conv_weight.shape[0],conv_weight.shape[1],conv_weight.shape[2],conv_weight.shape[3],1,1,1
	f.write('ConvolutionLayer %d %d %d %d %d %d %d %d ' % (oc,ic,kw,kh,sw,sh,bias,padding_type))				
	f.write(' '.join(map(str,conv_weight.flatten().tolist())) + ' ')		
	f.write(' '.join(map(str,conv_bias.flatten().tolist())) + ' ')
	f.write('\n')
	
def export_pool(f,channel):
	#PoolingLayer [pool_type] 1 32 2 2 2 2 1
	f.write('PoolingLayer 0 1 %d 2 2 2 2 1\n' % (channel))
	
def export_fc(f,fc_weight,fc_bias):
	#FullconnectLayer 1 512 1 1 1 0.139041
	print(fc_weight.shape)			
	print(fc_bias.shape)
	f.write('FullconnectLayer 1 %d 1 1 1 ' % fc_bias.shape[0])
	fc_weight = np.transpose(fc_weight,[1,0])
	f.write(' '.join(map(str,fc_weight.flatten().tolist())) + ' ')		
	f.write(' '.join(map(str,fc_bias.flatten().tolist())) + ' ')
	f.write('\n')
	
def export_model(tf_model_path,easycnn_model_path):	
	keep_prob = tf.placeholder(tf.float32)
	x = tf.placeholder(tf.float32, [None,std_width*std_height])
	predict,params = build_cnn(x,keep_prob)
	y = tf.placeholder(tf.float32, [None,classes])

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,tf_model_path)
		
		f = open(easycnn_model_path,'w')	
		#input
		export_input(f,1,std_width,std_height)
		#conv1
		conv_weight = params[0].eval()
		conv_bias = params[1].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#conv2
		conv_weight = params[2].eval()
		conv_bias = params[3].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#conv3
		conv_weight = params[4].eval()
		conv_bias = params[5].eval()		
		export_conv(f,conv_weight,conv_bias,1)		
		export_layer(f,'ReluLayer')
		export_pool(f,conv_weight.shape[3])
		
		#fc1
		fc_weight = params[6].eval()
		fc_bias = params[7].eval()		
		export_fc(f,fc_weight,fc_bias)
		export_layer(f,'ReluLayer')
		
		#fc2
		fc_weight = params[8].eval()
		fc_bias = params[9].eval()		
		export_fc(f,fc_weight,fc_bias)		
		export_layer(f,'SoftmaxLayer')
		
def main():
	if len(sys.argv) != 2:
		print('usage:\n\t%s [mode](0-train,1-test,2-export)' % sys.argv[0])
		sys.exit(0)
	mode = int(sys.argv[1])
	if mode != 0 and mode != 1 and mode != 2:
		print('mode is invalidate. mode: (0-train,1-test,2-export)')
		sys.exit(0)
		
	mnist_data_path = 'd:/tmp/mnist/'
	tf_model_path = '../../res/model/tf_mnist.model'
	easycnn_model_path = '../../res/model/mnist.model'
	
	print('loading mnist dataset...')
	mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
	
	#0->train
	#1->test 
	#2->export weight	
	if mode == 0:
		#do training 
		print('do training...')		
		train(mnist.train,mnist.validation,tf_model_path)
	elif mode == 1:
		#do testing
		print('do testing...')
		test(mnist.test,tf_model_path)
	elif mode == 2:
		export_model(tf_model_path,easycnn_model_path)
	
if __name__ == '__main__':
	main()
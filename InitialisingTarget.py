'''
this file is intent to generate the Q-value in the begaining of the work.
the step is input the image and output the networks value.
save it in the Targets200_News.txt file
'''
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.05, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def getBatchInput(inputs,start,batchSize) :
	first = start
	start = start + batchSize
	end = start
	return inputs[first:end],start 	

imageSize = [25,25,3]
batchSize = 100
games = 200
totalImages = games*imageSize[0]*imageSize[1]
learningRate = 0.001
lr_decay_rate = 0.9
lr_decay_step = 2000

folderName = 'NewCheckpoints'

if os.path.exists(folderName):
	shutil.rmtree(folderName)

os.mkdir(folderName)

checkpointFile = 'NewCheckpoints/Checkpoint3.ckpt'

x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])
W_conv1 = weight_variable([2, 2, 3, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

W_conv2 = weight_variable([2, 2, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

h_conv2_flat = tf.reshape(h_conv2, [-1, 25*25*20])
W_fc1 = weight_variable([25*25*20, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([100, 2])
b_fc2 = bias_variable([2])

y_out=tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.square(y_out),1)
avg_loss = tf.reduce_mean(loss)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learningRate,
								global_step,
								lr_decay_step,
								lr_decay_rate,
								staircase=True)

train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()  # 实际上每调用一次保存操作会创建后3个数据文件并创建一个检查点（checkpoint）文件，简单理解就是权重等参数被保存到 .chkp.data 文件中，以字典的形式；图和元数据被保存到 .chkp.meta 文件中，可以被 tf.train.import_meta_graph 加载到当前默认的图。

sess.run(tf.initialize_all_variables())

inputs = np.zeros([totalImages,imageSize[0],imageSize[1],imageSize[2]])
print('reading inputs')
for i in range(totalImages):
	temp = imageSize[0]*imageSize[1]
	inputs[i] = cv2.imread('trainImages/image_'+str(int(i/temp))+'_'+str(i%temp)+'.png')


print('inputs read')

start = 0
initialTarget = []
iterations = int(totalImages/batchSize)
save_path = saver.save(sess, checkpointFile)
print("Model saved in file: %s" % save_path)

print('number of iterations is %d'%iterations)
for i in range(iterations):
	batchInput,start = getBatchInput(inputs,start,batchSize)
	
	batchOutput = sess.run(y_out,feed_dict={x: batchInput})
	if i%50 == 0 :
		print('%d iterations reached'%i)
	for j in range(batchSize):
		initialTarget.append(batchOutput[j])

print(start)
np.savetxt('Targets200_New.txt',initialTarget)		


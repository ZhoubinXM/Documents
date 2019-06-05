'''
This code use to run the project.
It is the main python file in this project.
It is also use the Tensorboard code to visualized the network model.
'''
import numpy as np
import tensorflow as tf
import cv2
import ReCalculateTargets2

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.05, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

imageSize = [25,25,3]
batchSize = 50
games = 200
limit = imageSize[0]*imageSize[1]  # limit=625
totalImages = games*imageSize[0]*imageSize[1]
#gamesArr = np.arange(totalImages)
start = 0


def randomiseGames():           # 定义一个函数，用于让训练数据集中的数据打乱顺序  类似于replay memory
	global gamesArr
	np.random.shuffle(gamesArr)  # 将序列的所有元素随机排序。

def getInput(index):            # 定义一个函数，获取相应index的image
	global gamesArr
	global imageSize
	temp = imageSize[0]*imageSize[1]
	img = cv2.imread('trainImages/image_'+str(int(index/temp))+'_'+str(index%temp)+'.png')
	return img

def getRightOutput(index,points,target):  # 获取Right的Q-value 迭代公式 TD-ERROR  r+r*maxQ' -Q
	global gamesArr
	global imageSize

	pos = index % limit
	if (pos+1) % imageSize[0] == 0 or points[index+1] == -100:
		output = -100
	elif points[index+1] == 100:
		#print 'hi'
		output = 100
	else:
		output = points[index+1] - points[index]+max(target[index+1])
	#if index == 13 :
	#	print('the output for index 14 is %g'%output)	
	return output

def getBottomOutput(index,points,target) :
	global gamesArr
	global limit
	global imageSize
	
	pos = index % limit
	if (pos+imageSize[0]>=limit) or (points[index+imageSize[0]] == -100):
		#print 'hi'
		output = -100
	elif points[index+imageSize[0]] == 100:
		#print 'hi'
		output = 100	
	else:
		output = points[index+imageSize[0]] - points[index]+max(target[index+imageSize[0]])

	return output	

def getBatchInputOutput(batchSize,points,target):
	global start
	first = start
	start = start+batchSize
	global totalImages
	global gamesArr
	if start > len(gamesArr):
		print('randomise called again')
		randomiseGames()
		first = 0 
		start = batchSize
	end = start
	inputs = []
	output = []
	inputIndices = gamesArr[first:end]	
	for i in range(batchSize):
		inputs.append(getInput(inputIndices[i]))
	# points = np.genfromtxt('points.txt')
	# target = np.genfromtxt('Targets.txt')
	# print inputIndices
	for i in range(batchSize):
		output1 = getRightOutput(inputIndices[i], points, target)
		output2 = getBottomOutput(inputIndices[i], points, target)
		output.append([output1, output2])

	return np.array(inputs), np.array(output)

steps = 50
count = 0
restoreCheckpointFile = 'NewCheckpoints/Checkpoint3.ckpt'
saveCheckpointFile = 'NewCheckpoints/Checkpoint2.ckpt'
# 模型运行产生的所有数据保存到 NewCheckpoints 文件夹供 TensorBoard 使用
# 注意 writer 一定要写在大循环外面
sess = tf.Session()
# writer = tf.summary.FileWriter('/home/isap/Research/SiyuZhou/DQN_PathPlanning/NewCheckpoints/', sess.graph)
while count < steps:

	print('count is %d' % count)
	gamesArr = []
	points = np.genfromtxt('pointsNew.txt')  # pointsNew.txt 是 DataGeneration 生成的标志每个状态 s 的reward
	target = np.genfromtxt('Targets200_New.txt')  # Targets200_New.txt 是 InitialisingTarget 生成的初始目标 Q 值

	for i in range(totalImages):
		if points[i] == 0:  # 既没遇到障碍也没到达目标点的状态
			gamesArr.append(i)  # gameArr 记录了 125000 状态中既没遇到障碍也没到达目标点状态的序号
	# print(len(gamesArr)) --> 109800 = 125000-75(障碍的数量)x200(地图数量)-200(终点)
	randomiseGames()  # 将gameArr中的序列随机排序

	outputLength = 2
	epochs = 4
	learningRate = 0.001
	lr_decay_step = 2000
	lr_decay_rate = 0.9

	#restoreCheckpointFile = 'ModelCheckpoints/initialCheckpoint.ckpt'

	iterations = int(epochs*(len(gamesArr)/batchSize))
	with tf.Graph().as_default():

		x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])
		y = tf.placeholder(tf.float32, shape=[None, outputLength])
		W_conv1 = weight_variable([2, 2, 3, 10])
		# tf.summary.histogram('weight1', W_conv1)
		b_conv1 = bias_variable([10])
		# tf.summary.histogram('bias1', b_conv1)
		h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
		# layer1_image = h_conv1[0:1, :, :, 0:10]
		# layer1_image = tf.transpose(layer1_image, perm=[3, 1, 2, 0])
		W_conv2 = weight_variable([2, 2, 10, 20])
		# tf.summary.histogram('weight2', W_conv2)
		b_conv2 = bias_variable([20])
		# tf.summary.histogram('bias2', b_conv2)

		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
		# layer2_image = h_conv2[0:1, :, :, 0:10]
		# layer2_image = tf.transpose(layer2_image, perm=[3, 1, 2, 0])
		# layer_combine_1 = tf.concat([layer2_image, layer1_image], 2)
		# list_lc1 = tf.split(value=layer_combine_1, axis=0, num_or_size_splits=10)
		# layer_combine_1 = tf.concat(list_lc1, 1)
		# tf.summary.image("filtered_images_1", layer_combine_1)

		h_conv2_flat = tf.reshape(h_conv2, [-1, 25*25*20])

		W_fc1 = weight_variable([25*25*20, 100])
		b_fc1 = bias_variable([100])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

		W_fc2 = weight_variable([100, outputLength])
		b_fc2 = bias_variable([outputLength])

		y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

		loss = tf.reduce_mean(tf.square(y-y_out), 1)
		avg_loss = tf.reduce_mean(loss)
		# tf.summary.scalar('Average Losses', avg_loss)

		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(learningRate,
										global_step,
										lr_decay_step,
										lr_decay_rate,
										staircase=True)

		train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

		sess = tf.Session()
		# init = tf.global_variables_initializer()
		# sess.run(init)
		saver = tf.train.Saver()
		saver.restore(sess, restoreCheckpointFile)
		print('Model restored from file %s' % restoreCheckpointFile)
		# 调用 merge_all() 收集所有的操作数据
		# merged = tf.summary.merge_all()
		for i in range(iterations):
			inputs, outputs = getBatchInputOutput(batchSize, points, target)

			if i % 50 == 0:
				losses = sess.run(avg_loss,feed_dict={x: inputs, y: outputs})
				print('%d steps reached and the loss is %g'%(i,losses))
				
			sess.run(train_step,feed_dict={x: inputs, y: outputs})

			if i == iterations-1 :
				save_path = saver.save(sess, saveCheckpointFile)
				print("Model saved in file: %s" % save_path)
			
	ReCalculateTargets2.CalculateTargets(saveCheckpointFile)  # 更新一下 Q-target交换网路的参数方便继续进行后续output的计算
	# # 将 saveCheckpointFile 和 restoreCheckpointFile 交换
	# saver.restore(sess, restoreCheckpointFile)
	# print('Model restored from file %s' % restoreCheckpointFile)
	tempor = restoreCheckpointFile
	restoreCheckpointFile = saveCheckpointFile
	saveCheckpointFile = tempor
	count+=1


			



import os
import time
import math
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Select device
deviceType = "/gpu:0"

# Simple Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def simpleModel():
    with tf.device(deviceType):
        wConv = tf.get_variable("wConv", shape=[7, 7, 3, 32])
        bConv = tf.get_variable("bConv", shape=[32])
        w = tf.get_variable("w", shape=[5408, 10]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
        b = tf.get_variable("b", shape=[10])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(x, wConv, strides=[1, 2, 2, 1], padding='VALID') + bConv # Stride [batch, height, width, channels]
        h = tf.nn.relu(a)
        hFlat = tf.reshape(h, [-1, 5408]) # Flat the output to be size 5408 each row
        yOut = tf.matmul(hFlat, w) + b

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

def train(Model, xT, yT, xV, yV, xTe, yTe, batchSize=1000, epochs=100, printEvery=10):
    # Train Model
    trainIndex = np.arange(xTrain.shape[0])
    np.random.shuffle(trainIndex)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Mini-batch
            losses = []
            accs = []
            # For each batch in training data
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                currentBatchSize = yTrain[idX].shape[0]

                # Train
                loss, acc, _ = sess.run(Model, feed_dict={x: xT[idX, :], y: yT[idX]})

                # Collect all mini-batch loss and accuracy
                losses.append(loss * currentBatchSize)
                accs.append(acc * currentBatchSize)

            totalAcc = np.sum(accs) / float(xTrain.shape[0])
            totalLoss = np.sum(losses) / xTrain.shape[0]
            if e % printEvery == 0:
                print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100), end='')
                loss, acc = sess.run(Model[:-1], feed_dict={x: xV, y: yV})
                print(' Validate loss = {0:.3f} and validate accuracy = {1:.2f}%'.format(loss, acc * 100))
        loss, acc = sess.run(Model[:-1], feed_dict={x: xTe, y: yTe})
        print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))

# Start training simple model
print("\n################ Simple Model #########################")
train(simpleModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Complex Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def complexModel():
    with tf.device(deviceType):
        yOut = None
        filter = tf.get_variable(name = "wConv", shape = [7, 7, 3, 64])
        bConv = tf.get_variable(name = "bConv", shape = [64])

        #Define Convolutional layer
        a = tf.nn.conv2d(x, filter, strides = [1, 2, 2, 1], padding = 'VALID') + bConv
        Ra = tf.nn.relu(a) # output size = (32-7) / 2 + 1 = 13

        #Define maxpooling layer
        h = tf.nn.max_pool(Ra, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding='VALID')
        # output size = (13 - 2) / 1 + 1 = 12

        #Define hidden layer
        w1 = tf.get_variable(name = "w1", shape = [9216, 1024]) # parameter to learn : 12*12*64 = 9216
        b1 = tf.get_variable(name = "b1", shape = [1024])
        h_shift = tf.reshape(h, [-1, 9216])
        hidden_output = tf.matmul(h_shift, w1) + b1
        hidden_output1 = tf.nn.relu(hidden_output)

        #Define output layer
        w2 = tf.get_variable(name = "w2", shape = [1024, 10])
        b2 = tf.get_variable(name = 'b2', shape = [10])
        yOut = tf.matmul(hidden_output1, w2) + b2



        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]


#Start training complex model
print("\n################ Complex Model #########################")
train(complexModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Your Own Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def yourOwnModel():
    with tf.device(deviceType):
        yOut = None
        """
            I tried several times
            I find the response normalization does work well and it takes a lot of time which influence the train speed
            Finally discard it
        """
        #Define Convolutional layer
        filter1 = tf.get_variable(name = "wConv", shape = [3, 3, 3, 64])
        bConv1 = tf.get_variable(name = "bConv", shape = [64])

        conv1 = tf.nn.conv2d(x, filter1, strides = [1, 1, 1, 1], padding = 'SAME') + bConv1
        Rconv1 = tf.nn.relu(conv1) # output size = (32-3 + 2)/1 +1 =32 )

        #Define convolutional layer
        filter2 = tf.get_variable(name = "wCvon2", shape = [3,3,64,64])
        bConv2 = tf.get_variable(name = "bConv2", shape=[64])

        conv2 = tf.nn.conv2d(Rconv1, filter2, strides = [1, 1, 1, 1], padding = 'SAME') + bConv2
        Rconv2 = tf.nn.relu(conv2)#output size = (32-3+2) /1 +1 =32

        #Define maxpool layer
        h = tf.nn.max_pool(Rconv2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        #output size = (32-2) /2 +1 = 16

        #Define Convolutional layer
        filter3 = tf.get_variable(name = "wConv3", shape = [3, 3, 64, 64])
        bConv3 = tf.get_variable(name = "bConv3", shape = [64])

        conv3 = tf.nn.conv2d(h, filter3, strides = [1, 1, 1, 1], padding = 'SAME') + bConv3
        Rconv3 = tf.nn.relu(conv3) # output size = (16-3 + 2)/1 +1 =16

        #Define convolutional layer
        filter4 = tf.get_variable(name = "wCvon4", shape = [3,3,64,64])
        bConv4 = tf.get_variable(name = "bConv4", shape=[64])

        conv4 = tf.nn.conv2d(Rconv3, filter4, strides = [1, 1, 1, 1], padding = 'SAME') + bConv4
        Rconv4 = tf.nn.relu(conv4)#output size = (16-3+2) /1 +1 =16

        # Define maxpool layer
        h = tf.nn.max_pool(Rconv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # output size = (16-2) /2 +1 = 8

        #Define hidden layer
        w1 = tf.get_variable(name = "w1", shape = [4096, 512],) # parameter to learn : 8*8*64 = 4096
        b1 = tf.get_variable(name = "b1", shape = [512])
        h_shift = tf.reshape(h, [-1, 4096])
        hidden_output = tf.matmul(h_shift, w1) + b1
        hidden_output = tf.nn.relu(hidden_output)
        hidden_output1 = tf.nn.dropout(hidden_output, 0.8)#dropout to avoid overfitting

        w2 = tf.get_variable(name = "w2", shape=[512,64])
        b2 = tf.get_variable(name = "b2", shape = [64])
        fc = tf.matmul(hidden_output1, w2) + b2
        fc = tf.nn.relu(fc)

        #Define output layer
        w3 = tf.get_variable(name = "w3", shape = [64, 10])
        b3 = tf.get_variable(name = 'b3', shape = [10])
        yOut = tf.matmul(fc, w3) + b3

        # Define Loss
        totalLoss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=yOut) + 5e-4 * tf.nn.l2_loss(w1) + 5e-4 * tf.nn.l2_loss(w2) + 5e-4 * tf.nn.l2_loss(w3)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate= 0.001)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

# Start your own Model model
print("\n################ Your Own Model #########################")
train(yourOwnModel(), xTrain, yTrain, xVal, yVal, xTest, yTest, batchSize=100, epochs=100)



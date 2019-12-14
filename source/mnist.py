import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
y_label = tf.argmax(y_, axis=1)
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
'''
print('='*100)

print('The shape of the output of layer fc1:')
print(h_fc1.shape)

print('='*100)
'''
sess.run(tf.initialize_all_variables())
for i in range(2000):
	batch = mnist.train.next_batch(50)
	
	if i%500 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

print('='*100)
print("dropout %g, test accuracy %g"%(0.75, accuracy.eval(feed_dict={
	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
print('='*100)

# generate selected features of mnist
x_train = mnist.train.images[:-15000]
y_train = mnist.train.labels[:-15000]
x_train_m = sess.run(h_pool2_flat, feed_dict={x: x_train, y_: y_train})
y_train_m = tf.reshape(tf.argmax(y_train, axis=1), shape=[-1, 1])
#x_train_m = x_train_m.eval()
y_train_m = y_train_m.eval()
train_data = np.concatenate((y_train_m, x_train_m), axis=1)
flags = train_data!=0
with open('mnist.select', mode='w') as f:
	for i in range(train_data.shape[0]):
		f.write(str(int(train_data[i][0])))
		f.write(' ')
		j=0
		for flag in flags[i][1:]:
			if flag:
				f.write('{}:{} '.format(j+1, train_data[i][j+1]))
			j+=1
		f.write('\n')

x_test = mnist.test.images
y_test = mnist.test.labels
x_test_m = sess.run(h_pool2_flat, feed_dict={x: x_test, y_: y_test})
y_test_m = tf.reshape(tf.argmax(y_test, axis=1), shape=[-1, 1])
#x_test_m = x_test_m.eval()
y_test_m = y_test_m.eval()
test_data = np.concatenate((y_test_m, x_test_m), axis=1)
flags = test_data!=0
with open('mnist.select.t', mode='w') as f:
	for i in range(test_data.shape[0]):
		f.write(str(int(test_data[i][0])))
		f.write(' ')
		j=0
		for flag in flags[i][1:]:
			if flag:
				f.write('{}:{} '.format(j+1, test_data[i][j+1]))
			j+=1
		f.write('\n')
	

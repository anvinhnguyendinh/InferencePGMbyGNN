#%% (0) Important libraries
import tensorflow as tf
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from IPython import display
# % matplotlib inline



#%% (1) Dataset creation.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Data MNIST/', one_hot=False)



#%% (2) Model definition.

import tensorflow as tf
from tensorflow.layers import Flatten, Conv2D

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

dtype = tf.float32
conv_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype, uniform=False) # tf.initializers.truncated_normal(dtype=dtype, mean=0, stddev=0.1)

0
time_step = 10
label_size = 2
hidden_size = 5
message_size = 5
bp_time_step = 30
input_dimensions = 28
message_hidden_size = 64

input_dimensions_2 = input_dimensions * 2
input_dimensions_m1 = input_dimensions - 1

input_layer = tf.placeholder(dtype=dtype, shape=(None, input_dimensions, input_dimensions, 1), name='input') # [0, 1]
expected_output = tf.placeholder(dtype=tf.int32, shape=(None, input_dimensions, input_dimensions), name='expected_output') # [0, 1]

hyper_h = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, ), mean=0, stddev=1), name='hyper_h') #
hyper_eta = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, input_dimensions, input_dimensions, 1), mean=0, stddev=1), name='hyper_eta')
hyper_beta = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, input_dimensions_2, input_dimensions_m1, 1), mean=0, stddev=1), name='hyper_beta')

batch = tf.shape(input_layer)[0]


def node2edge(node, name=None):
	hor0 = tf.slice(node, [0, 0, 0, 0], [-1, input_dimensions, input_dimensions_m1, -1])
	hor1 = tf.slice(node, [0, 0, 1, 0], [-1, input_dimensions, input_dimensions_m1, -1])
	hor  = tf.concat([hor0, hor1], axis=-1)
	ver0 = tf.transpose(tf.slice(node, [0, 0, 0, 0], [-1, input_dimensions_m1, input_dimensions, -1]), [0, 2, 1, 3])
	ver1 = tf.transpose(tf.slice(node, [0, 1, 0, 0], [-1, input_dimensions_m1, input_dimensions, -1]), [0, 2, 1, 3])
	ver  = tf.concat([ver0, ver1], axis=-1)
	return tf.concat([hor, ver], axis=1, name=name)


potential_size = 3
unary = tf.square(hyper_h) - (input_layer * 2 - 1) * tf.square(hyper_eta) #
unary_edge = node2edge(unary, 'unary_edge')
binary_edge = tf.tile(tf.square(hyper_beta), [batch, 1, 1, 1], name='binary_edge')
potential = tf.concat([unary_edge, binary_edge], axis=-1, name='potential')
# potential_ = tf.reshape(potential, [1, -1, input_dimensions_2, input_dimensions_m1, potential_size])
# potential_dup = tf.tile(tf.expand_dims(potential, axis=0), [time_step, 1, 1, 1, 1], name='potential_dup')


# depth_size = hidden_size * 2 + potential_size
# H_0 = tf.tensordot(input_layer, tf.zeros(dtype=dtype, shape=(1, hidden_size)), axes=1, name='H_0')
# E_t = node2edge(H_0, 'E_t') # H_tm1
# E_t_bar = tf.concat([E_t, potential], axis=-1) # X_t
# M_t_2 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer)(E_t_bar)
# M_t_1 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer)(M_t_2)
# M_t_0 = Conv2D(filters=message_size       , kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer)(M_t_1)
# M_t_bar = edge2node(M_t_0, 'M_t_bar')
# H_t = GRU(H_tm1, M_t_bar)
# gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
# O, H_t = tf.nn.dynamic_rnn(gru_cell, tf.expand_dims(tf.reshape(tf.slice(M_t_bar, [0, 0, 0, 0], [-1, 1, 1, -1]), [-1, hidden_size]), axis=0), \
# 	initial_state=tf.reshape(tf.slice(H_0, [0, 0, 0, 0], [-1, 1, 1, -1]), [-1, hidden_size]), time_major=True) # H_tm1


class GNN_GRU:
	def __init__(self):
		self.pad0 = [[0, 0], [0, 0], [0, 1], [0, 0]]
		self.pad1 = [[0, 0], [0, 0], [1, 0], [0, 0]]

		self.CM_2 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer, name='CM_2')
		self.CM_1 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer, name='CM_1')
		self.CM_0 = Conv2D(filters=message_size * 2   , kernel_size=1, activation=tf.nn.relu, kernel_initializer=conv_initializer, name='CM_0') # * 2

		# Weights for input vectors of shape (message_size, hidden_size)
		self.Wr = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Wr')
		self.Wz = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Wz')
		self.Wh = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Wh')
		
		# Weights for hidden vectors of shape (hidden_size, hidden_size)
		self.Ur = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Ur')
		self.Uz = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Uz')
		self.Uh = Conv2D(filters=hidden_size, kernel_size=1, activation=None, kernel_initializer=conv_initializer, use_bias=True, name='C_Uh')
		
		# Biases for hidden vectors of shape (hidden_size,)
		# self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, 1, hidden_size), mean=0, stddev=0.01), name='br')
		# self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, 1, hidden_size), mean=0, stddev=0.01), name='bz')
		# self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(1, 1, hidden_size), mean=0, stddev=0.01), name='bh')
		# self.Br = tf.tile(self.br, [input_dimensions, input_dimensions, 1], name='C_Br')
		# self.Bz = tf.tile(self.bz, [input_dimensions, input_dimensions, 1], name='C_Bz')
		# self.Bh = tf.tile(self.bh, [input_dimensions, input_dimensions, 1], name='C_Bh')
		
		self.X = tf.ones([time_step, 1])
		self.H_0 = tf.tensordot(input_layer, tf.zeros(dtype=dtype, shape=(1, hidden_size)), axes=1, name='H_0')
		self.H_ts = tf.scan(self.forward_pass, self.X, initializer=self.H_0, name='H_ts')
		self.H_cur = tf.squeeze(tf.slice(self.H_ts, [time_step - 1, 0, 0, 0, 0], [1, -1, -1, -1, -1]), axis=0, name='H_cur')


	def get_current_state(self):
		return self.H_cur

	def forward_pass(self, H_tm1, X_t):
		E_t = node2edge(H_tm1)
		E_t_bar = tf.concat([E_t, potential], axis=-1) #

		M_t_2 = self.CM_2(E_t_bar)
		M_t_1 = self.CM_1(M_t_2)
		M_t_0 = self.CM_0(M_t_1)
		M_t_bar = self.edge2node(M_t_0)

		Z_t = tf.sigmoid(self.Wz(M_t_bar) + self.Uz(H_tm1))
		R_t = tf.sigmoid(self.Wr(M_t_bar) + self.Ur(H_tm1))
		
		H_proposal = tf.tanh(self.Wh(M_t_bar) + self.Uh(tf.multiply(R_t, H_tm1)))
		
		H_t = tf.multiply(1 - Z_t, H_tm1) + tf.multiply(Z_t, H_proposal)
		
		return H_t

	def edge2node(self, edge, size=message_size, name=None): #
		hor0 = tf.slice(edge, [0, 0, 0,    0], [-1, input_dimensions, input_dimensions_m1, size])
		hor1 = tf.slice(edge, [0, 0, 0, size], [-1, input_dimensions, input_dimensions_m1, size])
		ver0 = tf.slice(edge, [0, input_dimensions, 0,    0], [-1, input_dimensions, input_dimensions_m1, size])
		ver1 = tf.slice(edge, [0, input_dimensions, 0, size], [-1, input_dimensions, input_dimensions_m1, size])
		hor_ = tf.pad(hor0, self.pad0) + tf.pad(hor1, self.pad1)
		ver_ = tf.pad(ver0, self.pad0) + tf.pad(ver1, self.pad1)
		return tf.add(hor_, tf.transpose(ver_, [0, 2, 1, 3]), name=name)

main_layers = GNN_GRU()
H_cur = main_layers.get_current_state()


map_2 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu   , kernel_initializer=conv_initializer, name='CR_2')(H_cur)
map_1 = Conv2D(filters=message_hidden_size, kernel_size=1, activation=tf.nn.relu   , kernel_initializer=conv_initializer, name='CR_1')(map_2)
map_0 = Conv2D(filters=label_size         , kernel_size=1, activation=tf.nn.softmax, kernel_initializer=conv_initializer, name='CR_0')(map_1)
output_layer = map_0 # tf.subtract(1., map_0, name='soft_output') #
label_layer = tf.cast(tf.argmax(output_layer, axis=-1), dtype=tf.int32, name='output')


ones_repar_unary = np.ones((1, 1, 1, label_size))
ones_repar_unary[:, :, :, :1] *= -1
ones_repar_binary = np.array([[1, -1]])
ones_repar_binary = - ones_repar_binary.T.dot(ones_repar_binary).reshape((1, 1, 1, label_size, -1))

repar_unary = tf.multiply(unary, tf.constant(ones_repar_unary, dtype=dtype), name='repar_unary')
repar_binary = tf.multiply(tf.expand_dims(binary_edge, axis=-1), tf.constant(ones_repar_binary, dtype=dtype), name='repar_binary')
repar_degree = main_layers.edge2node(tf.ones([1, input_dimensions_2, input_dimensions_m1, 2]), size=1, name='repar_degree') # 2

def loop(E_0, V_0, time=time_step):
	pad0 = [[0, 0], [0, 0], [0, 1], [0, 0]]
	pad1 = [[0, 0], [0, 0], [1, 0], [0, 0]]
	E_ts, V_ts = [E_0], [V_0]
	for i in range(time):
		E, V = E_ts[-1], V_ts[-1]
		hor0 = tf.slice(V, [0, 0, 0, 0], [-1, input_dimensions, input_dimensions_m1, -1])
		hor1 = tf.slice(V, [0, 0, 1, 0], [-1, input_dimensions, input_dimensions_m1, -1])
		ver0 = tf.transpose(tf.slice(V, [0, 0, 0, 0], [-1, input_dimensions_m1, input_dimensions, -1]), [0, 2, 1, 3])
		ver1 = tf.transpose(tf.slice(V, [0, 1, 0, 0], [-1, input_dimensions_m1, input_dimensions, -1]), [0, 2, 1, 3])
		v0 = tf.expand_dims(tf.concat([hor0, ver0], axis=1), axis=-1) # b, 2*, -1, 2 1
		v1 = tf.transpose(tf.expand_dims(tf.concat([hor1, ver1], axis=1), axis=-1), [0, 1, 2, 4, 3]) # 1 2
		mbu0 = tf.reduce_min(E + v1, axis=-1, keepdims=True)
		mbu1 = tf.reduce_min(E + v0, axis=-2, keepdims=True)
		nnu0 = tf.divide(v0 - mbu0, 2)
		nnu1 = tf.divide(v1 - mbu1, 2)
		E_ts.append(E + (nnu0 + nnu1) / 4) #
		npu0 = tf.squeeze(v0 - nnu0, axis=-1)
		npu1 = tf.squeeze(tf.transpose(v1 - nnu1, [0, 1, 2, 4, 3]), axis=-1)
		npu2 = tf.pad(npu0, pad0) + tf.pad(npu1, pad1)
		hor_ = tf.slice(npu2, [0, 0, 0, 0], [-1, input_dimensions, input_dimensions, -1])
		ver_ = tf.slice(npu2, [0, input_dimensions, 0, 0], [-1, input_dimensions, input_dimensions, -1])
		V_t_bar = hor_ + tf.transpose(ver_, [0, 2, 1, 3])
		V_ts.append(tf.divide(V_t_bar, repar_degree))
	return V_ts[-1]

V_t = loop(repar_binary, repar_unary, bp_time_step)
V_softmin = tf.nn.softmax(- V_t, axis=-1, name='bp_output') #
V_label = tf.cast(tf.argmin(V_t, axis=-1), dtype=tf.int32, name='bp_label') #
one_hot_V_t = tf.one_hot(V_label, depth=label_size, dtype=dtype)


bp_lambda = 0.5
one_hot_output = tf.one_hot(expected_output, depth=label_size, dtype=dtype)
loss = tf.reduce_mean(- (one_hot_output + tf.stop_gradient(bp_lambda * one_hot_V_t)) * tf.log(output_layer), name='loss') + 0.001 * tf.square(hyper_h) #
accuracy = 1 - tf.reduce_mean(tf.cast(tf.equal(label_layer, expected_output), dtype=dtype), name='accuracy')
# loss = tf.reduce_mean(- one_hot_output * tf.log(V_softmin), name='loss') + 0.01 * tf.square(hyper_h)
# accuracy = 1 - tf.reduce_mean(tf.cast(tf.equal(V_label, expected_output), dtype=dtype), name='accuracy')

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), len(tf.get_default_graph().get_operations()))
print([v.name for v in tf.trainable_variables()])
hhh = [v for v in tf.trainable_variables() if v.name == 'hyper_h:0'][0]
# session = tf.Session()
# init_variables = tf.global_variables_initializer()
# session.run(init_variables)
# iii = mnist.test.images[:4]
# l, aaa, pu = session.run([loss, accuracy, V_t], feed_dict={input_layer: iii.reshape((-1, 28, 28, 1)), expected_output: iii.reshape((-1, 28, 28)).astype(np.int32)})
# print(l, aaa, pu.shape)
# exit()



#%% (3) Initialize and train the model.

# Initialize a session
session = tf.Session()

# Use the Adam optimizer for training
num_epoch = 2
batch_size = 16
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss) #

# Initialize all the variables
init_variables = tf.global_variables_initializer()
session.run(init_variables)

# Initialize the losses
train_losses = []
validation_losses = []

# Perform all the iterations

np.random.seed(1000) # 2000, 8000, 100, .01, 2
display_step = 100#625
num_train = 8000#mnist.train.num_examples
Y_train = mnist.train.images[:num_train]
Y_train = (np.round(Y_train.reshape((-1, input_dimensions, input_dimensions, 1)))).astype(np.int32)
X_train = Y_train * 1.0
Y_train = Y_train.squeeze()
p, q = .14, .5
flipped  = np.random.choice([True, False], size=Y_train.shape, p=[p, 1 - p])
salted   = np.random.choice([True, False], size=Y_train.shape, p=[q, 1 - q])
peppered = ~ salted
X_train[flipped & salted] = 1
X_train[flipped & peppered] = 0
num_test  = 2000#mnist.test .num_examples
Y_test  = mnist.test .images[:num_test ]
Y_test  = (np.round(Y_test .reshape((-1, input_dimensions, input_dimensions, 1)))).astype(np.int32)
X_test  = Y_test  * 1.0
Y_test  = Y_test .squeeze()
flipped  = np.random.choice([True, False], size=Y_test .shape, p=[p, 1 - p])
salted   = np.random.choice([True, False], size=Y_test .shape, p=[q, 1 - q])
peppered = ~ salted
X_test [flipped & salted] = 1
X_test [flipped & peppered] = 0

for epoch in range(num_epoch):
	avg_cost = 0
	total_batch = num_train // batch_size
	for i in range(0, total_batch * batch_size, batch_size):
		batch_x, batch_y = X_train[i : i + batch_size], Y_train[i : i + batch_size]
		_, c = session.run([train_step, accuracy], feed_dict={input_layer: batch_x, expected_output: batch_y})
		avg_cost += c / total_batch
		iba1 = i // batch_size + 1
		if iba1 % display_step == 0:
			print('Iteration %04d, cost: %.9f' % (i + batch_size, avg_cost * total_batch / iba1))
	train_loss = avg_cost

	avg_test = 0
	total_batch = num_test  // batch_size
	for i in range(0, total_batch * batch_size, batch_size):
		batch_x, batch_y = X_test [i : i + batch_size], Y_test [i : i + batch_size]
		c = session.run(accuracy, feed_dict={input_layer: batch_x, expected_output: batch_y})
		avg_test += c / total_batch
	validation_loss = avg_test
	X_nn, X_bp = session.run([label_layer, V_label], feed_dict={input_layer: X_test[:batch_size], expected_output: Y_test[:batch_size]}) #
	
	train_losses += [train_loss]
	validation_losses += [validation_loss]
	print('Epoch %03d, train loss: %.4f, test loss: %.4f' % (epoch + 1, train_loss, validation_loss))
	
	if (epoch + 1) % num_epoch == 0:
		plt.plot(train_losses, '-b', label='Train loss')
		plt.plot(validation_losses, '-r', label='Validation loss')
		plt.title('Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(); plt.show()

print(session.run(tf.square(hhh)))
plt.imshow(Y_test[0], cmap='Greys')
plt.show()
plt.imshow(X_nn  [0], cmap='Greys')
plt.show()
plt.imshow(X_bp  [0], cmap='Greys')
plt.show()
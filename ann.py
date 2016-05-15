import os
import timeit
import numpy as np
from struct import unpack
from scipy.optimize import fmin_cg
import pickle

"""
Created by: Ahad Shoaib
"""

ZERO_EPSILON = 0.0000000000000000000000000000000000000000000000000000000000000000000000000001


TRAINING_IMAGES_NAME = "train-images-idx3-ubyte"
TRAINING_LABELS_NAME = "train-labels-idx1-ubyte"

TEST_IMAGES_NAME = "t10k-images-idx3-ubyte"
TEST_LABELS_NAME = "t10k-labels-idx1-ubyte"


class DataProcess(object):

	global TRAINING_IMAGES_NAME
	global TRAINING_LABELS_NAME

	global TEST_IMAGES_NAME
	global TEST_LABELS_NAME

	global ZERO_EPSILON

	def __init__(self, training, m):
		self.training = training
		self.examples = m
		self.X = None
		self.y = None

		self.y_2d = None # TODO: make a 3D matrix for [m][row_pixel][col_pixel]

	# training = true/false
	# returns (X,y)
	def get_data(self):

		if (self.training):
			names_set_filename = os.path.join(os.path.dirname(__file__), TRAINING_IMAGES_NAME)
			labels_set_filename = os.path.join(os.path.dirname(__file__), TRAINING_LABELS_NAME)
		else:
			names_set_filename = os.path.join(os.path.dirname(__file__), TEST_IMAGES_NAME)
			labels_set_filename = os.path.join(os.path.dirname(__file__), TEST_LABELS_NAME)

		images_set = open(names_set_filename, "rb+")
		labels_set = open(labels_set_filename, "rb+")

		images_set.read(4) # skips magic number
		labels_set.read(4)

		# int
		num_images = images_set.read(4) # reduce this number for testing
		num_images = unpack('>I', num_images)[0] # >I == big endian unsigned int

		num_images = self.examples

		num_rows = images_set.read(4)
		num_rows = unpack('>I', num_rows)[0]

		num_cols = images_set.read(4)
		num_cols = unpack('>I', num_cols)[0]

		# int
		num_labels = unpack('>I', labels_set.read(4))[0]
		# print num_rows, num_cols, num_labels

		pixel_features = num_rows * num_cols

		X_shape = (num_images, pixel_features)
		y_shape = (num_images, 1)

		X = np.empty(X_shape, dtype=np.float64)
		y = np.empty(y_shape, dtype=np.float64)

		for m in range(num_images): # training examples
			# print "Adding example [%d]" % (m + 1)
			for x_i in range(pixel_features): # features
				X[m, x_i] = unpack('>B', images_set.read(1))[0]
			y[m] = unpack('>B', labels_set.read(1))[0]

		if self.training:
			print "added training examples..."
		else:
			print "added test examples..."

		self.X = self.normalize_features(X);
		self.y = y;

		return (X, y)

	def normalize_features(self, X):
		min_arr = np.amin(X, axis=0)
		max_arr = np.amax(X, axis=0)

		norm_arr = np.linalg.norm(X, axis=0)
		m = X.shape[0]
		for i in xrange(m):
			X[i, :] = X[i, :] / (norm_arr + ZERO_EPSILON)
		print "normalization complete..."

		return X

	# ??? 
	def visualize(self):
		pass





# ANN with 3 total layers, 1 hidden layer
class ANN(object):

	"""
	PSEUDOCODE:
	- randomly init thetas
	- FP to get h_theta(x^i)
	- compute J(theta)
	- BP to compute d(J(theta))/d-theta [partial derivative]
	- optimization to min theta J

	TESTING: compare gradient/cost computed by BP/FP with grad desc
	"""

	global ZERO_EPSILON

	INPUT_NEURONS = 784 # 28 x 28 image sizes
	HIDDEN_NEURONS = 100 # may change later
	OUTPUT_NEURONS = 10

	# INPUT_NEURONS = 2
	# HIDDEN_NEURONS = 2
	# OUTPUT_NEURONS = 4

	"""
	theta1 = HIDDEN_NEURONS  x (INPUT_NEURONS + 1)
	theta2 = OUTPUT_NEURONS x (HIDDEN_NEURONS + 1)
	X = M x (HIDDEN_NEURONS + 1)
	y = M x 1
	"""

	def __init__(self, theta = None, X = None, y = None, LAMBDA = 0.00001):
		# TODO Load from theano/numpy/something as an array
		self.X = X # 6000 x (784 + 1)
		self.y = y # 6000 x 1
		self.M = X.shape[0]
		self.LAMBDA = LAMBDA

		if theta is None: # == will be elementwise check
			theta1, theta2 = self.random_init()
			# this is an array
			self.theta = np.append(np.asarray(theta1.flatten()), 
								np.asarray(theta2.flatten()))
		else:
			self.theta = theta



		self.theta1_total = self.HIDDEN_NEURONS * (self.INPUT_NEURONS + 1)
		self.theta2_total = self.OUTPUT_NEURONS * (self.HIDDEN_NEURONS + 1)

		# print ("totals 1. [%d] 2. [%d]" % (self.theta1_total, self.theta2_total))
		# print ("t1 [%s] t2 [%s]" % (theta1.shape, theta2.shape))


	def get_theta_mat(self):
		return np.asmatrix(self.theta)

	# default
	def get_theta(self):
		return np.asarray(self.theta)

	def random_init(self):
		# TODO: better random init?
		theta1 = np.random.randn(self.HIDDEN_NEURONS, self.INPUT_NEURONS + 1)
		theta2 = np.random.randn(self.OUTPUT_NEURONS, self.HIDDEN_NEURONS + 1)
		return (theta1, theta2)


	# large z -> 1~, small z -> 0~
	def sigmoid_fn(self, z):
		return 1 / (1 + np.exp(-z))


	# z = 0 --> 0.25, large neg/pos z --> ~0
	def sigmoid_gradient_fn(self, z):
		return np.multiply(self.sigmoid_fn(z), (1 - self.sigmoid_fn(z)))


	# theta1, theta2 should be randomly init before
	def cost(self, theta, X, y, reg_const):
		theta1, theta2 = self.unravel_theta(theta)

		m = self.M

		# cost value
		J = 0

		# add bias column to X
		bias_col = np.ones((m, 1), dtype=np.float64)
		X = np.append(bias_col, X, axis=1)

		for i in range(m):
			# forward propogation
			#######################################################################

			# print "M, theta1, X[i], bias, X"
			# print self.M, theta1.shape, np.asmatrix(X[i]).shape, bias_col.shape, X.shape, type(X)

			z_2 = np.dot(theta1, np.transpose(X[i]))

			a_2 = self.sigmoid_fn(z = z_2) # HIDDEN_NEURONS x 1
			a_2 = np.vstack([np.asmatrix([1.0]), a_2]) # (HIDDEN_NEURONS + 1) x 1

			z_3 = np.dot(theta2, a_2)
			a_3 = self.sigmoid_fn(z = z_3) 

			h_theta = a_3 # should be OUTPUT_NEURONS x 1

			# calculate cost
			#######################################################################

			# array of [OUTPUT_NEURONS] columns, 1 if y[i], 0 else
			y_ik = np.asmatrix(np.arange(self.OUTPUT_NEURONS), dtype=np.float64).transpose() == y[i]

			# fun python note: -y_ik converts all False->True, all True->False..
			# I spent a whole cup of coffee figuring that out
			# TODO: make the zero_epsilon solution nicer
			J += np.sum(np.multiply((-1.0 * y_ik), np.log(h_theta + ZERO_EPSILON)) 
				- np.multiply((1.0 - y_ik), np.log(1 - (h_theta + ZERO_EPSILON))))
		# print "Before REG: J is [%f]" % J

		### REGULARIZATION ###

		regsum = 0
		# theta1 sum
		t1_rows = theta1.shape[0]
		t2_rows = theta2.shape[0]

		for i in xrange(t1_rows):
			regsum += np.sum(np.square(theta1[i, 2:]))

		for i in xrange(t2_rows):
			regsum += np.sum(np.square(theta2[i, 2:]))

		regsum *= (reg_const / (2.0 * m))

		J += regsum
		# print "After REG: J is [%f]" % J

		return J


	def grad(self, theta, X, y, reg_const):

		theta1, theta2 = self.unravel_theta(theta)

		m = self.M

		# add bias column to X
		bias_col = np.ones((m, 1), dtype=np.float64)
		X = np.append(bias_col, X, axis=1)

		# init for BP
		Delta1 = np.zeros(theta1.shape)
		Delta2 = np.zeros(theta2.shape)
		delta3 = None

		for i in range(m):
			# forward propogation
			#######################################################################

			a_1 = X[i, :]
			z_2 = np.dot(theta1, np.transpose(X[i]))

			a_2 = self.sigmoid_fn(z = z_2) # HIDDEN_NEURONS x 1
			a_2 = np.vstack([np.asmatrix([1.0]), a_2]) # (HIDDEN_NEURONS + 1) x 1

			z_3 = np.dot(theta2, a_2)
			a_3 = self.sigmoid_fn(z = z_3) 

			h_theta = a_3 # should be OUTPUT_NEURONS x 1

			# calculate gradient - backward propogation
			#######################################################################

			# array of [OUTPUT_NEURONS] columns, 1 if y[i], 0 else
			y_ik = np.asmatrix(np.arange(self.OUTPUT_NEURONS)).transpose() == y[i]

			# i.e. [0 1 2 3 ... 10] --> [0 0 1 0 ... 0] if y[i] = 3
			delta3 = a_3 - y_ik

			grad_sig = np.vstack([np.matrix([1]), self.sigmoid_gradient_fn(z_2)])
			delta2 = np.multiply(np.dot(np.transpose(theta2), delta3), grad_sig)
			delta2 = np.delete(delta2, 1, axis=0) # remove first row of delta2, corresponds to bias unit

			Delta1 = Delta1 + np.dot(delta2, a_1)
			Delta2 = Delta2 + np.dot(delta3, np.transpose(a_2))


		### REGULARIZATION ###

		theta1_reg = theta1.copy()
		theta1_reg[0] = 0

		theta2_reg = theta2.copy()
		theta2_reg[0] = 0

		theta1_grad_reg = (float(reg_const) / m) * theta1_reg
		theta2_grad_reg = (float(reg_const) / m) * theta2_reg

		theta1_grad = Delta1 * (1.0 / m) + theta1_grad_reg
		theta2_grad = Delta2 * (1.0 / m) + theta2_grad_reg

		# theta1_grad = Delta1 * (1.0 / m)
		# theta2_grad = Delta2 * (1.0 / m)


		gradient1 = np.asarray(theta1_grad).ravel() # .shape = theta1.shape
		gradient2 = np.asarray(theta2_grad).ravel() # .shape = theta2.shape
		gradient = np.hstack([gradient1, gradient2]) # returns 1xsize ARRAY (should not be MATRIX because np.dot doesnt work)

		return gradient.transpose() # .transpose() allows fmin_cg to optimize properly

	# requires trained theta1 and theta2
	def predict(self, theta, X_test):
		theta1, theta2 = self.unravel_theta(theta)
		print "predicting..."

		num_test = X_test.shape[0]

		# bias unit
		X_test = np.append(np.ones((num_test, 1)), X_test, axis=1)

		a_2 = self.sigmoid_fn(np.dot(theta1, np.transpose(X_test)))
		a_2 = np.vstack([np.ones((1, X_test.shape[0])), a_2])
		a_3 = self.sigmoid_fn(np.dot(theta2, a_2))

		prediction = np.argmax(a_3, axis=0)

		return np.transpose(prediction)

	def unravel_theta(self, theta):

		theta1 = np.asmatrix(np.reshape(np.take(theta, np.arange(self.theta1_total)), (self.HIDDEN_NEURONS, self.INPUT_NEURONS + 1)))
		theta2 = np.asmatrix(np.reshape(np.take(theta, np.arange(self.theta1_total, self.theta1_total + self.theta2_total)),
				(self.OUTPUT_NEURONS, self.HIDDEN_NEURONS + 1)))
		return (theta1, theta2)


	# will use SGD later maybe, this is temporary/faster
	def train(self, myiter = 100):
		print "training..."
		init_theta = self.get_theta()
		theta = fmin_cg(self.cost, init_theta, args = (self.X, self.y, self.LAMBDA), 
			fprime = self.grad, maxiter = myiter)

		return theta
		


if __name__ == '__main__':

	# True to load from files, False to train ANN freshly
	# LOAD_FROM_FILE = True # UNCOMMENT THIS TO LOAD TRAINING DATA
	# LOAD_FROM_FILE = False # UNCOMMENT THIS TO TRAIN NEW SET

	TRAINING_EXAMPLES = 200
	TESTING_EXAMPLES = 300
	LAMBDA = 1

	print "--------------"

	print "[TRAINING_EXAMLES = %d]"

	train_data = DataProcess(training = True, m = TRAINING_EXAMPLES)
	X, y = train_data.get_data()

	test_data = DataProcess(training = False, m = TESTING_EXAMPLES)
	X_test, y_test = test_data.get_data()

	print "--------------"

	if LOAD_FROM_FILE:
		trained_theta = np.load("trained_params.npy", allow_pickle = True)
		ann = ANN(theta = trained_theta, X = np.asmatrix(X), y = np.asmatrix(y))
	else:
		ann = ANN(X = np.asmatrix(X), y = np.asmatrix(y))
		trained_theta = ann.train()
		np.save("trained_params", trained_theta, allow_pickle = True)

	print "--------------"

	prediction = ann.predict(trained_theta, X_test)
	print "[PREDICTIONS]:"
	print prediction
	print "[ACTUAL]:"
	print y_test
	print "Accuracy = [%.2f]" % ((y_test[np.where(prediction == y_test)].size / float(y_test.size)) * 100.00)

	print "--------------"

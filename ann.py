import os
import timeit
import numpy as np
from struct import unpack
from scipy.optimize import fmin_cg

from pymongo import MongoClient
from bson.binary import Binary
import pickle

# mName = matrix of Name, p x q


# NOT NEEDED?
class MongoDatabase(object):
	
	def __init__(self, new = False, X = None, y = None, theta1 = None, theta2 = None):



		self._client = MongoClient("localhost", 27017)
		self.db = self._client['digits-nn-database']

		curr_set_number = retrieve_from_db()

		if new:
			curr_set_number = curr_set_number + 1
			# make new entry with values of X, y, and theta1/2 


		coll = db.test_collection # TODO Learn
		obj = None
		bytes = pickle.dump(obj)
		coll.insert({'random-data': Binary(bytes)})


	def update_X(X):
		pass

	def update_y():
		pass

	def update_theta1():
		pass

	def update_theta2():
		pass

	# retrieve from DB
	def get_X():
		pass

	def get_y():
		pass

	def get_theta1():
		pass

	def get_theta2():
		pass

	def save():
		# save all fields, i.e. create a new "finalized" entry
		pass

TRAINING_IMAGES_NAME = "train-images-idx3-ubyte"
TRAINING_LABELS_NAME = "train-labels-idx1-ubyte"

TEST_IMAGES_NAME = "t10k-images-idx3-ubyte"
TEST_LABELS_NAME = "t10k-labels-idx1-ubyte"


class DataProcess(object):

	global TRAINING_IMAGES_NAME
	global TRAINING_LABELS_NAME

	global TEST_IMAGES_NAME
	global TEST_LABELS_NAME

	def __init__(self, training):
		self.training = training
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
		# num_images = images_set.read(4) # reduce this number for testing
		# num_images = unpack('>I', num_images)[0] # >I == big endian unsigned int
		num_images = 5

		num_rows = images_set.read(4)
		num_rows = unpack('>I', num_rows)[0]

		num_cols = images_set.read(4)
		num_cols = unpack('>I', num_cols)[0]

		# int
		num_labels = unpack('>I', labels_set.read(4))[0]
		print num_rows, num_cols, num_labels

		pixel_features = num_rows * num_cols

		X_shape = (num_images, pixel_features)
		y_shape = (num_images, 1)

		X = np.empty(X_shape, dtype = np.int32)
		y = np.empty(y_shape, dtype = np.int32)

		for m in range(num_images): # training examples
			print "Adding example [%d]" % m
			for x_i in range(pixel_features): # features
				X[m, x_i] = unpack('>B', images_set.read(1))[0]
			y[m] = unpack('>B', labels_set.read(1))[0]

		self.X = X;
		self.y = y;

		return (X, y)

	def serialize_arr(val):
		pass


	def visualize(self):
		pass






# ANN with 3 total layers, 1 hidden layer
class ANN(object):

	"""
	ANN:
	1. randomly init thetas
	2. implement FP to get h_theta(x^i) for any x^i
	3. implement code to compute J(theta) [formula]
	4. implement BP to compute d(J(theta))/d-theta [partial derivative]
	5. use gradient checking to compare grad using BP with gradApprox
	6. use grad desc or "advanced optimization" to min theta J
	"""

	INPUT_NEURONS = 784 # 28 x 28 image sizes
	HIDDEN_NEURONS = 100 # may change later
	OUTPUT_NEURONS = 10

	M = 100 # training examples
	TESTING_EXAMPLES = 0

	LAMBDA = 1

	"""
	theta1 = HIDDEN_NEURONS  x (INPUT_NEURONS + 1)
	theta2 = OUTPUT_NEURONS x (HIDDEN_NEURONS + 1)
	X = M x (HIDDEN_NEURONS + 1)
	y = M x 1
	"""

	def __init__(self, theta1 = None, theta2 = None, X = None, y = None):
		# TODO Load from theano/numpy/something as an array
		self.X = "" # 6000 x (784 + 1)
		self.y = "" # 6000 x 1

		if theta1 == None and theta2 == None:
			theta1, theta2 = random_init(theta1, theta2)

		# this is an array
		self.theta = np.append(np.asarray(theta1.flatten()), 
								np.asarray(theta2.flatten()))

		self.theta1_total = HIDDEN_NEURONS * (INPUT_NEURONS + 1)
		self.theta2_total = OUTPUT_NEURONS * (HIDDEN_NEURONS + 1)


		# ma.item((x, y)) or ma.item(x)
		# 


	def random_init(self, theta1, theta2):
		# TODO: better random init?
		theta1 = np.random.randn(HIDDEN_NEURONS, INPUT_NEURONS + 1)
		theta2 = np.random.randn(OUTPUT_NEURONS, HIDDEN_NEURONS + 1)
		return (theta1, theta2)


	# large z -> 1~, small z -> 0~
	def sigmoid_fn(self, z):
		return 1 / (1 + np.exp(-z))


	# z = 0 --> 0.25, large neg/pos z --> ~0
	def sigmoid_gradient_fn(self, z):
		return sigmoid_fn(z) * (1 - sigmoid_fn(z))


	# theta1, theta2 should be randomly init before
	def cost(self, theta, X, y):
		theta1, theta2 = unravel(theta)

		# cost value
		J = 0

		# add bias column to X
		bias_col = np.ones((M, 1))
		X = np.append(bias_col, X, axis=1)

		for i in range(M):

			# forward propogation
			#######################################################################

			z_2 = np.dot(theta1, np.transpose(X[i]))
			a_2 = sigmoid_fn(z = z_2) # HIDDEN_NEURONS x 1
			a_2 = np.append(np.ones(1), a_2, axis = 1) # (HIDDEN_NEURONS + 1) x 1

			z_3 = np.dot(theta2, a_2)
			a_3 = sigmoid_fn(z = z_3) 

			h_theta = a_3 # should be OUTPUT_NEURONS x 1

			# calculate cost
			#######################################################################

			# array of [OUTPUT_NEURONS] columns, 1 if y[i], 0 else
			y_ik = np.asmatrix(np.arrange(OUTPUT_NEURONS)).transpose() == y[i]

			J += np.sum((-y_ik) * log(h_theta) - (1 - y_ik) * log(1 - h_theta))

		# TODO: regularization

		return J


			"""
	theta1 = (HIDDEN_NEURONS + 1) x (INPUT_NEURONS + 1)
	theta2 = OUTPUT_NEURONS x (HIDDEN_NEURONS + 1)
	X = M x (HIDDEN_NEURONS + 1)
	y = M x 1
	"""

	def grad(self, theta, X, y):
		theta1, theta2 = unravel(theta)

		# add bias column to X
		bias_col = np.ones((M, 1))
		X = np.append(bias_col, X, axis=1)

		# init for BP
		Delta1 = np.zeros(theta1.shape)
		Delta1 = np.zeros(theta2.shape)
		delta3 = np.zeros(OUTPUT_NEURONS, 1)

		for i in range(M):
			# forward propogation
			#######################################################################

			z_2 = np.dot(theta1, np.transpose(X[i]))
			a_2 = sigmoid_fn(z = z_2) # HIDDEN_NEURONS x 1
			a_2 = np.append(np.ones(1), a_2, axis = 1) # (HIDDEN_NEURONS + 1) x 1

			z_3 = np.dot(theta2, a_2)
			a_3 = sigmoid_fn(z = z_3) 

			h_theta = a_3 # should be OUTPUT_NEURONS x 1

			# calculate gradient - backward propogation
			#######################################################################

			# array of [OUTPUT_NEURONS] columns, 1 if y[i], 0 else
			y_ik = np.asmatrix(np.arrange(OUTPUT_NEURONS)).transpose() == y[i]

			# i.e. [0 1 2 3 ... 10] --> [0 0 1 0 ... 0] if y[i] = 3
			delta3 = a_3 - y_ik

			grad_sig = np.append(np.asmatrix(np.ones(1)), sigmoid_gradient_fn(z_2), axis=1)
			delta2 = np.dot(np.transpose(theta2), delta3) * grad_sig
			delta2 = np.delete(delta2, 1, axis=0) # remove first row of delta2, corresponds to bias unit

			Delta1 = Delta1 + np.dot(delta2, a_1)
			Delta2 = Delta2 + np.dot(delta3, a_2)

		# TODO REGULARIZATION FOR J

		theta1_reg = theta1.copy()
		theta1_reg[0] = 0

		theta2_reg = theta2.copy()
		theta2_reg[0] = 0

		theta1_grad_reg = (LAMBDA / m) * theta1_reg
		theta2_grad_reg = (LAMBDA / m) * theta2_reg

		theta1_grad = Delta1 * (1 / m) + theta1_grad_reg
		theta2_grad = Delta2 * (1 / m) + theta2_grad_reg


		gradient1 = theta1_grad.flatten() # .shape = theta1.shape
		gradient2 = theta2_grad.flatten() # .shape = theta2.shape
		gradient = np.append(gradient1, gradient2, axis=0)

		return gradient

	# requires trained theta1 and theta2
	def predict(self, theta1, theta2, X_test):
		print "Predicting..."

		num_test = X_test.shape[0]

		# bias unit
		X_test = np.append(np.ones((num_test, 1)), X_test, axis=1)

		a_2 = sigmoid_fn(np.dot(theta1, np.transpose(X_test)))
		a_3 = sigmoid_fn(np.dot(theta2, a_2))

		values, prediction = argmax(a_3, [], axis=1)

		return np.transpose(prediction)

	def unravel_theta(self, theta):
		theta1 = np.unravel(np.take(theta, np.arange(self.theta1_total)), 
				(HIDDEN_NEURONS, INPUT_NEURONS + 1))
		theta2 = np.unravel(np.take(theta, np.arange(self.theta1_total, self.theta2_total)),
				(OUTPUT_NEURONS, HIDDEN_NEURONS + 1))
		return (theta1, theta2)

	# can use gradient descent, but this is faster
	def train(self, myiter = 50):
		print "Training..."
		theta = fmin_cg(cost, self.theta, fprime = grad, maxiter = myiter)
		return unravel(theta)
		


if __name__ == '__main__':
	train_data = DataProcess(training = True)
	# test_data = DataProcess(training = False)
	# database = MongoDatabase()

	(X, y) = train_data.get_data()

	neural_net = ANN(X=X, y=y)


	print X
	print y
	# prediction = None
	# print "Accuracy = [%.2f]" % ((y[np.where(prediction == y)].size / float(y.size)) * 100.00)




	# print X
	# print y

# tested on Python 3.6.3
# tf version 1.2
# work dir must contain validation_predictons.txt

import tensorflow as tf
s = tf.InteractiveSession()


# function computes MSE 
with tf.name_scope("MSE"):
    y_true = tf.placeholder("float32", shape=(None,), name="y_true")
    y_predicted = tf.placeholder("float32", shape=(None,), name="y_predicted")
    mse = tf.reduce_mean((y_true - y_predicted)**2)
def compute_mse(vector1, vector2):
    return mse.eval({y_true: vector1, y_predicted: vector2})

# test
vector1 = [1,1,3]
vector2 = [1,1,1]
print(vector1, vector2)
print('mse =',compute_mse(vector1,vector2))


# # 2nd assignment: Logistic regression
# To implement the logistic regression
# 
# Plan:
# * Use a shared variable for weights
# * Use a matrix placeholder for `X`
#  
# We shall train on a two-class MNIST dataset

# load data
from sklearn.datasets import load_digits
mnist = load_digits(2)

X, y = mnist.data, mnist.target

print("y [shape - %s]:" % (str(y.shape)), y[:10])
print("X [shape - %s]:" % (str(X.shape)))
print('X:\n',X[:3,:10])
print('y:\n',y[:10])

# split data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train.shape, y_train.shape, X_test.shape


# Model parameters - weights and bias
weights = tf.Variable(tf.random_normal([X.shape[1],1], stddev=0.35),\
                      name="weights") 
b = tf.Variable(0, dtype='float32', name="biases")

weights, b


# Placeholders for the input data
input_X = tf.placeholder('float32', shape=(None,None))
input_y = tf.placeholder('float32', shape=(None, ))
input_X, input_y


# The model code

# Compute a vector of predictions, resulting shape should be [input_X.shape[0],]
# This is 1D, if you have extra dimensions, you can  get rid of them with tf.squeeze.
# Don't forget the sigmoid.
a = tf.matmul(input_X, weights)
predicted_y =  tf.nn.sigmoid(tf.squeeze(a)+b)

# Loss. Should be a scalar number - average loss over all the objects
loss = tf.reduce_mean(-tf.log(predicted_y+1e-07)*input_y-\
                      tf.log(1-predicted_y+1e-07)*(1 - input_y))

# optimization function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
    loss, var_list=(weights,b))


predicted_y, loss, optimizer, a


# A test to help with the debugging

import numpy as np

my_scalar = tf.placeholder('float32')
my_vector = tf.placeholder('float32', [None])

# provided function
# Warning! Trying to understand the meaning of that function may result
# in permanent brain damage
weird_psychotic_function = tf.reduce_mean(
    (my_vector+my_scalar)**(1+tf.nn.moments(my_vector,[0])[1]) + 
    1./ tf.atan(my_scalar))/(my_scalar**2 + 1) + 0.01*tf.sin(
    2*my_scalar**1.5)*(tf.reduce_sum(my_vector)* my_scalar**2
                      )*tf.exp((my_scalar-4)**2)/(
    1+tf.exp((my_scalar-4)**2))*(1.-(tf.exp(-(my_scalar-4)**2)
                                    )/(1+tf.exp(-(my_scalar-4)**2)))**2

validation_weights = 1e-3 * np.fromiter(map(lambda x:
        s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 0.1, 2]}),
                            0.15 * np.arange(1, X.shape[1] + 1)),
                            count=X.shape[1], dtype=np.float32)[:, np.newaxis]

# Compute predictions for given weights and bias
prediction_validation = s.run(
    predicted_y, {
    input_X: X,
    weights: validation_weights,
    b: 1e-1})

# Load the reference values for the predictions
validation_true_values = np.loadtxt("validation_predictons.txt"),
print(prediction_validation.shape == (X.shape[0],),
      "Predictions must be a 1D array with length equal to the number of \
      examples in input_X")


print(np.allclose(validation_true_values, prediction_validation),
      'Predictions are correct')


loss_validation = s.run(
        loss, {
            input_X: X[:100],
            input_y: y[-100:],
            weights: validation_weights+1.21e-3,
            b: -1e-1})


print(np.allclose(loss_validation, 0.728689), 'Correct loss on validation')


from sklearn.metrics import roc_auc_score
s.run(tf.global_variables_initializer())
for i in range(4):
    s.run(optimizer, {input_X: X_train, input_y: y_train})
    loss_i = s.run(loss, {input_X: X_train, input_y: y_train})
    print("loss at iter %i:%.4f" % (i, loss_i))
    print("train auc:", roc_auc_score(y_train, s.run(predicted_y,\
                                                     {input_X:X_train})))
    print("test auc:", roc_auc_score(y_test, s.run(predicted_y,\
                                                   {input_X:X_test})))

test_weights = 1e-3 * np.fromiter(map(lambda x:
    s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]}),
                              0.1 * np.arange(1, X.shape[1] + 1)),
                              count=X.shape[1], dtype=np.float32)[:,np.newaxis]


# First, test prediction and loss computation. This part doesn't require a fitted model.
prediction_test = s.run(
    predicted_y, {
    input_X: X,
    weights: test_weights,
    b: 1e-1})


prediction_test.shape == (X.shape[0],)
# "Predictions must be a 1D array with length equal to the number of examples in X_test"

loss_test = s.run(
    loss, {
        input_X: X[:100],
        input_y: y[-100:],
        weights: test_weights+1.21e-3,
        b: -1e-1})
# Yes, the X/y indices mistmach is intentional
loss_test, roc_auc_score(y_test, s.run(predicted_y, {input_X:X_test}))
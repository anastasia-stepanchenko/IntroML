# tested on Python 3.6.3
# work dir must contain: preprocessed_mnist.py
# implements multiclass logistic regression and applies to mnist

import tensorflow as tf
import numpy as np
#import os

#os.chdir("D:\Programming\Python\TensorFlow\MultiClassReg")
from preprocessed_mnist import load_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print("X and y train shape:", X_train.shape, y_train.shape)
print("X and y test shape:", X_test.shape, y_test.shape)
#print(X_val.shape, y_val.shape)


# # Logistic regression

s = tf.Session()

# Reshape features to flat format
X_train_flat = s.run(tf.reshape(X_train, [X_train.shape[0],-1]))
X_test_flat  = s.run(tf.reshape(X_test, [X_test.shape[0],-1]))
X_val_flat   = s.run(tf.reshape(X_val, [X_val.shape[0],-1]))

# Categorical labels to binaries
y_train_oh = s.run(tf.one_hot(y_train, 10))
y_test_oh  = s.run(tf.one_hot(y_test, 10))
y_val_oh   = s.run(tf.one_hot(y_val, 10))


# Model parameters - weights and bias
nuniq = len(np.unique(y_train))
weights = tf.Variable(tf.random_normal([X_train_flat.shape[1],nuniq], stddev=0.35),
                      name="weights") 

b = tf.Variable(tf.zeros(nuniq), dtype='float32', name="biases")

# Placeholders for the input data
input_X = tf.placeholder('float32', shape=(None,X_train_flat.shape[1]))
input_y = tf.placeholder('float32', shape=(None, nuniq))
input_X, input_y, weights, b


# model
#predicted_y =  tf.nn.softmax(tf.matmul(input_X, weights)+b)
predicted_y =  tf.matmul(input_X, weights)+b

# Loss. Should be a scalar number - average loss over all the objects
#loss = tf.reduce_mean(-tf.reduce_sum(tf.log(predicted_y+1e-07)*input_y, reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=predicted_y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(
    loss, var_list=(weights,b))

# train
from sklearn.metrics import roc_auc_score

s.run(tf.global_variables_initializer())

for i in range(100):
    #batchX, batchY = s.run(tf.train.batch([X_train_flat, y_train_oh],100,enqueue_many=True, capacity=1))
    s.run(optimizer, {input_X: X_train_flat, input_y: y_train_oh})
    #s.run(optimizer, {input_X: batchX, input_y: batchY})
    loss_i = s.run(loss, {input_X: X_train_flat, input_y: y_train_oh})
    print("loss at iter %i:%.4f" % (i, loss_i))
    print("train auc:", roc_auc_score(y_train_oh, s.run(predicted_y, {input_X:X_train_flat})))
    print("test auc:", roc_auc_score(y_test_oh, s.run(predicted_y, {input_X:X_test_flat})))


# compute accuracy
correct_prediction = tf.equal(tf.argmax(predicted_y,1), tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(s.run(accuracy, feed_dict={input_X:X_train_flat, input_y: y_train_oh}))
print(s.run(accuracy, feed_dict={input_X:X_val_flat, input_y: y_val_oh}))
print(s.run(accuracy, feed_dict={input_X:X_test_flat, input_y: y_test_oh}))


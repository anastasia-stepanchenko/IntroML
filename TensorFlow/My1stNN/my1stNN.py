# tested on Python 3.6.3
# work dir must contain: preprocessed_mnist.py
# builds 2-layer NN and applies to mnist

import tensorflow as tf
import numpy as np
#import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


#os.chdir("D:\Programming\Python\TensorFlow\My1stNN")
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
nhid   = 50
nclass = len(np.unique(y_train))
weights_hid = tf.Variable(tf.random_normal([X_train_flat.shape[1], nhid], stddev=0.35),
                      name="weights_h") 

b_hid = tf.Variable(tf.zeros([nhid]), dtype='float32', name="biases_h")

weights_out = tf.Variable(tf.random_normal([nhid, nclass], stddev=0.35),
                      name="weights") 

b_out = tf.Variable(tf.zeros([nclass]), dtype='float32', name="biases")

# Placeholders for the input data
input_X = tf.placeholder('float32', shape=(None,X_train_flat.shape[1]))
input_y = tf.placeholder('float32', shape=(None, nclass))
input_X, input_y, weights_hid, weights_out, b_hid, b_out


# model
#predicted_y =  tf.nn.softmax(tf.matmul(input_X, weights)+b)
predicted_y_hid =  tf.nn.relu(tf.matmul(input_X, weights_hid)+b_hid)
predicted_y     =  tf.matmul(predicted_y_hid, weights_out)+b_out

# Loss. Should be a scalar number - average loss over all the objects
#loss = tf.reduce_mean(-tf.reduce_sum(tf.log(predicted_y+1e-07)*input_y, reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=predicted_y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(
    loss, var_list=(weights_hid, b_hid, weights_out, b_out))

# compute accuracy
correct_prediction = tf.equal(tf.argmax(predicted_y,1), tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train
trainloss = list()
testloss  = list()
acctrain  = list()
acctest   = list()
s.run(tf.global_variables_initializer())

for i in range(400):
    #batchX, batchY = s.run(tf.train.batch([X_train_flat, y_train_oh],100,enqueue_many=True, capacity=1))
    s.run(optimizer, {input_X: X_train_flat, input_y: y_train_oh})
    #s.run(optimizer, {input_X: batchX, input_y: batchY})
    loss_i = s.run(loss, {input_X: X_train_flat, input_y: y_train_oh})
    trainloss.append(loss_i)
    loss_i = s.run(loss, {input_X: X_test_flat, input_y: y_test_oh})
    testloss.append(loss_i)
    acctrain.append(s.run(accuracy, feed_dict={input_X:X_train_flat, input_y: y_train_oh}))
    acctest.append(s.run(accuracy, feed_dict={input_X:X_test_flat, input_y: y_test_oh}))
    print("loss at iter %i:%.4f" % (i, loss_i))
    print("train auc:", roc_auc_score(y_train_oh, s.run(predicted_y, {input_X:X_train_flat})))
    print("test auc:", roc_auc_score(y_test_oh, s.run(predicted_y, {input_X:X_test_flat})))


# plot loss and accuracy for each iteration

plt.plot(trainloss, label = "loss train")
plt.plot(testloss, label="loss test")
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.plot(acctrain, label = "accuracy train")
plt.plot(acctest, label  = "accuracy test")
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


# print accuracy
correct_prediction = tf.equal(tf.argmax(predicted_y,1), tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(s.run(accuracy, feed_dict={input_X:X_train_flat, input_y: y_train_oh}))
print(s.run(accuracy, feed_dict={input_X:X_val_flat, input_y: y_val_oh}))
print(s.run(accuracy, feed_dict={input_X:X_test_flat, input_y: y_test_oh}))





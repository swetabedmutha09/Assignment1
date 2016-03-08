#
# The code mentioned below is obtained from Professor Abram Hindle git repository
# it has been slightly modified for the purpose of assignment 



# first off we load up some modules we want to use
import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
import collections
import theautil

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)


mupdates = 1000
#Both the dataset are downloaded from uci machine learning datasets
#Load data

#winedata set
data = np.loadtxt("wine.data", delimiter=",")

#seprate data and labels

#Inputs have 12 attributes
inputs  = data[0:,1:13].astype(np.float32)


#Output contains 3 classes each class for different cultivater
outputs = data[0:,0:1].astype(np.int32)

#shuffle input and output
theautil.joint_shuffle(inputs,outputs)

train_and_valid, test = theautil.split_validation(80, inputs, outputs)
train, valid = theautil.split_validation(80, train_and_valid[0], train_and_valid[1])

def linit(x):
    return x.reshape((len(x),))

train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
test  = (test[0] ,linit(test[1]))


# oneNN classifier

def euclid(pt1,pt2):
    return sum([ (pt1[i] - pt2[i])**2 for i in range(0,len(pt1)) ])

def oneNN(data,labels):
    def func(input):
        distance = None
        label = None
        for i in range(0,len(data)):
            d = euclid(input,data[i])
            if distance == None or d < distance:
                distance = d
                label = labels[i]
        return label
    return func

learner = oneNN(train[0],train[1])

oneclasses = np.apply_along_axis(learner,1,test[0])
print "1-NN classifier!"
print "%s / %s " % (sum(oneclasses == test[1]),len(test[1]))
print theautil.classifications(oneclasses,test[1])

print '''
########################################################################
# Using neural networks! Architecture type 1
########################################################################
'''

# try different combos here

#for wine dataset
net = theanets.Classifier([12,(6, 'relu'),3])
net.train(train, valid, algo='layerwise', max_updates=mupdates, patience=1)
net.train(train, valid, algo='rprop',     max_updates=mupdates, patience=1)

print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])


print net.layers[2].params[0].get_value()
print net.layers[2].params[0].get_value()

print '''
########################################################################
# Using neural networks! Architecture type 2
########################################################################
'''

# try different combos here

#for wine dataset
net = theanets.Classifier([12,(6,'sigmoid'),3,3])
net.train(train, valid, algo='layerwise', max_updates=mupdates, patience=1)
net.train(train, valid, algo='rprop',     max_updates=mupdates, patience=1)

print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])


print net.layers[2].params[0].get_value()
print net.layers[2].params[0].get_value()

print '''
########################################################################
# Using neural networks! Architecture type 3
########################################################################
'''

# try different combos here

#for wine dataset
net = theanets.Classifier([12,(4,'tanh'),(4,'tanh'),(4,'tanh'),3])
net.train(train, valid, algo='layerwise', max_updates=mupdates, patience=1)
net.train(train, valid, algo='rprop',     max_updates=mupdates, patience=1)

print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])


print net.layers[2].params[0].get_value()
print net.layers[2].params[0].get_value()

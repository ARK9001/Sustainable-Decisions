# -*- coding: utf-8 -*-

from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import os
import cv2
import numpy
import scipy

import PIL
from PIL import Image

# image size to be parsed
basewidth = 10
hsize = 10

# enums for the node outputs
category_to_int = {
    "compost":(0,0),
    "paper":(0,1),
    "recycle":(1,0),
    "trash":(1,1)
}

int_to_category = {
    (0,0):"compost",
    (0,1):"paper",
    (1,0):"recycle",
    (1,1):"trash"
}

# resize image
def resize(img):
    return scipy.misc.imresize(img, (basewidth, hsize))

# flattens image to 1d feature vector list
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# load image, resize, and flatten to feature vector
def loadImage(path):
    image = cv2.imread(path)
    im = resize(image)
    return flatten(im)

# takes output of neural net and returns category
def categorize(result):
    if result[0] > .5:
        first = 1
    else:
        first = 0
    if result[1] > .5:
        second = 1
    else:
        second = 0
    return int_to_category[(first, second)]

# build neural network and train
# would be nice to reduce dimension of feature vector to get rid of uneccessary data
# before inputting into neural network (future work)
if __name__ == "__main__":
    t = basewidth * hsize * 3
    net = buildNetwork(t, t * t / 20, t, 2, bias = True)
    ds = SupervisedDataSet(t, 2)
    for category in ["compost","paper","recycle","trash"]:
        for filename in os.listdir(category+"/"):
            if filename.endswith(".png"): 
                ds.addSample(loadImage(category + "/" + filename),category_to_int[category])
 
    trainer = BackpropTrainer(net, ds)
    error = 1
    iteration = 0
    while error > 0.001 and iteration < 40: 
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)
    
    # currently tested on dumb data, use with data that has been separated into
    # training and test data, and only test on non trained data
    print "\nResult: ", categorize(net.activate(loadImage('a.png')))
    print "\nResult: ", categorize(net.activate(loadImage('a_r.png')))
    print "\nResult: ", categorize(net.activate(loadImage('b.png')))
    print "\nResult: ", categorize(net.activate(loadImage('b_r.png')))
    print "\nResult: ", categorize(net.activate(loadImage('c.png')))
    print "\nResult: ", categorize(net.activate(loadImage('c_r.png')))
    print "\nResult: ", categorize(net.activate(loadImage('d.png')))
    print "\nResult: ", categorize(net.activate(loadImage('d_r.png')))
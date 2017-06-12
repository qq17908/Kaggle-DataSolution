# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 22:49:23 2017

@author: Paul.lu
"""

import pandas as pd
import matplotlib.pylab as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

# Loading the data
labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images,test_images,train_labels,test_labels = train_test_split(images,labels,train_size=0.8,random_state=0)

# Viewing an Image
i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])

# Examining the Pixel Values
plt.hist(train_images.iloc[i])

# Training our model
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)

test_images[test_images>0] = 1
train_images[train_images>0] =1

img = train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

# Examining the Pixel Values
plt.hist(train_images.iloc[i])

# Retaining our model
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)

# Labelling the test data
test_data = pd.read_csv('test.csv')
test_data[test_data>0] = 1
results = clf.predict(test_data[0:5000])

results


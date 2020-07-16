# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from Inception_cnn.MiniGoogleNet import MiniGoogLeNet
from trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
from simplepreprocessor import SimplePreprocessor
from dataArgument.aspectawarepreprocessor import AspectAwarePreprocessor
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

epochs = 10
optimizer = 'adam'
isDataArgumentation = False
imageWidth = 64
imageHeight = 64
imageDepth = 1
imageClasses = 184
imageDataSetPath = "/home/devpc/Downloads/leafsnap-dataset/dataset/images/field"

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha


imagePaths = list()
label = list()

# walk through the directory to get all the images paths
for x in os.walk(imageDataSetPath):
    for filename in x[2]:
        filePath = f'{x[0]}/{filename}'
        imagePaths.append(filePath)

    s_index = x[0].rindex('/')
    # labels of classes in folder
    label.append(x[0][s_index + 1:])

    # Intialize Preprocessor for image

# AspectAwarePreprocessor crop the image with respect to short dimensions
aap = AspectAwarePreprocessor(imageWidth, imageHeight)
# ImageToArrayPreprocessor convert the image to array
iap = ImageToArrayPreprocessor()

# SimpleDatasetLoader convert image to array and process the image as per given processor
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), len(le.classes_))
# set the labels with sequence as per labelEncoder arranges them to global variable

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training, and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.10, stratify=labels, random_state=42)
# partition the data into training and validation split using 80% of the data for
# training and the remaining 20% for testing
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.20, stratify=trainY, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

callbacks = [TrainingMonitor("fig.png", jsonPath="jsonPath.json"), LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=64, height=64, depth=3, classes=imageClasses)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // 64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save("model.hdf5")

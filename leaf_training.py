# the necessary packages
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
from lenet import LeNet
from vgg_net import VGG_Net
from AlexNet.AlexNet import AlexNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
from simplepreprocessor import SimplePreprocessor
from dataArgument.aspectawarepreprocessor import AspectAwarePreprocessor
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import imutils
import cv2
import pandas as pd
import random
from keras.preprocessing.image import img_to_array

# global class list
classes = list()

def getResultMaskModel(model_, epochs, optimizer, isDataArgumentation, imageWidth, imageHeight, imageDepth,
                       imageClasses, imageDataSetPath):

    # preparing name to save graph and model
    model_name = ""
    if model_ == LeNet:
        model_name = "Lenet_"
    elif model_ == VGG_Net:
        model_name = "Vgg_"
    elif model_ == AlexNet:
        model_name = "AlexNet_"

    # do we want to apply data argumentation or not
    if isDataArgumentation:
        model_name = model_name + "with_data_argumentation"
    else:
        model_name = model_name + "without_data_argumentation"


    print("[INFO] loading images ...")
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
    sdl = SimpleDatasetLoader(isGreyRequire=True, preprocessors=[aap, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0

    # convert the labels from integers to vectors
    le = LabelEncoder().fit(labels)
    labels = np_utils.to_categorical(le.transform(labels), len(le.classes_))
    # set the labels with sequence as per labelEncoder arranges them to global variable
    global classes
    classes = le.classes_
    print(le.classes_)

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

    # initialize the model
    print("[INFO] compiling model...")
    model = model_.build(width=imageWidth, height=imageHeight, depth=imageDepth, classes=imageClasses)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # if DataArgumentation required or not
    if isDataArgumentation:
        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=180, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                 zoom_range=0.5, horizontal_flip=True, fill_mode="nearest")
        # train the network
        print("[INFO] training network...")
        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(valX, valY),
                                epochs=epochs, verbose=1, class_weight=classWeight)
    else:
        # train the network
        print("[INFO] training network...")
        H = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=64, epochs=epochs, verbose=1,
                      class_weight=classWeight)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                   target_names=le.classes_, output_dict=True)
    # saving the classification report in csv format
    df = pd.DataFrame(report).transpose()
    df.to_csv(model_name + "_classification_report.csv")

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(f"model_{model_name}.hdf5")

    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(f"plot_{model_name}.png")
    #plt.show()



def detectClasses(modelPath, testSetPath, imageWidth, imageHeight, imageDepth):
    imagePaths = list()
    label = list()
    labels = list()
    count = 0

    # walk through the directory to get all the images paths
    for x in os.walk(testSetPath):
        for filename in x[2]:
            filePath = f'{x[0]}/{filename}'
            imagePaths.append(filePath)
            labels.append(filePath.split(os.path.sep)[-2])

        s_index = x[0].rindex('/')
        label.append(x[0][s_index + 1:])
        count += 1

    # select the random index to select few images for testing
    x = np.random.randint(0, len(labels), size=5)
    imagePaths = np.array(imagePaths)
    imagePaths = imagePaths[x]
    labels = np.array(labels)
    labels = labels[x]

    global classes
    classList = classes

    # load model
    model = load_model(modelPath)
    # Apply preprocessing
    aap = AspectAwarePreprocessor(imageWidth, imageHeight)

    # set correction count to zero
    correct_count = 0

    # loop through imagepaths to check each image for possible leaf type
    for index, imagePath in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        print(index)
        frameClone = image
        image = aap.preprocess(image)
        image = image.astype("float") / 255.0
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        pred_prob = model.predict(image)
        prob_list = pred_prob[0]
        res = sorted(range(len(prob_list)), key=lambda sub: prob_list[sub])[-10:]
        arr_class = np.array(classList)
        print(arr_class[res])
        label = classList[int(np.argmax(pred_prob[0]))]
        print(labels[index],"------->", label)
        if labels[index] == label:
            correct_count += 1
        cv2.putText(frameClone, "Label: {}".format(label), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frameClone, "original Label: {}".format(labels[index]), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Face", frameClone)
        cv2.waitKey(0)
    print(f'correct prediction percentage = {(correct_count / len(labels)) * 100}')


# calling each model to create the model
for model_ in [VGG_Net, LeNet, AlexNet]:
    for isDataArgumentAllowed in [True, False]:
        getResultMaskModel(model_=model_, epochs=10, optimizer='adam', isDataArgumentation=isDataArgumentAllowed, imageWidth=64,
                           imageHeight=64, imageDepth=1, imageClasses=185,
                           imageDataSetPath="leafsnap-dataset/dataset/images/field")

detectClasses(testSetPath="leafsnap-dataset/dataset/images/field", modelPath="LeafSnap_AlexNet/model_AlexNet_with_data_argumentation.hdf5",
              imageWidth=64, imageHeight=64, imageDepth=3)


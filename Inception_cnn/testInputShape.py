from keras.models import load_model

# model = load_model("/home/devpc/Downloads/Inception_model_100_lab_sgd.hdf5")
# for layer in model.layers:
#     print(layer.output_shape)

import os
import cv2
from LeafSnapProject.Inception_cnn.imageProcessing.imageSegmentation import ImageSegmentation


imagePath = "/home/devpc/Downloads/NewLeafLabData/acer_saccharum/pi0029-01-2.jpg"
image = cv2.imread(imagePath)
processor1 = ImageSegmentation(500,500)
image1 = processor1.preprocess(image)
cv2.imshow("image1", image1)

processor2 = ImageSegmentation(64,64)
image2 = processor2.preprocess(image)
cv2.imshow("image2", image2)

cv2.waitKey(0)

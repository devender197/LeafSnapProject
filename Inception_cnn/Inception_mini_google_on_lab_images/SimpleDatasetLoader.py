import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, isGreyRequire=False, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        self.isGreyRequire = isGreyRequire
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input image
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            if self.isGreyRequire:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                                                      len(imagePaths)))

        # return a tuple of the data and labels
        return np.array(data), np.array(labels)

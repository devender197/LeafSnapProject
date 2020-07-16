import cv2

class MeanProcessing:

    def __init__(self, rMean, gMean, bMean):
        self.bMean = bMean
        self.gMean = gMean
        self.rMean = rMean

    def preprocess(self,image):
        #split the image into its respective Red, Green, and Blue channel
        (B, G, R) = cv2.split(image.astype("float32"))

        # substract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])
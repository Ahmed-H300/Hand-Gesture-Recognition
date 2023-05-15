# imports
import numpy as np
import cv2
from sklearn import mixture as GMM
from sklearn.cluster import KMeans


class PreprocessModel:

    def __init__(self):
        # making the gmm_model with number of components = 2 (background, skin-like colour)
        self.GMM_model = GMM.GaussianMixture(n_components=2, covariance_type='full', warm_start=True,
                                             init_params='kmeans', random_state=0)
        # creating 2 Kmean models (one with K = 2 and the other one with K = 3)
        self.KMeans2_model = KMeans(n_clusters=2, random_state=0, n_init='auto')

    def preProcess(self, img):

        # resize the image
        ratio = 480 / img.shape[1]
        resizedImage = cv2.resize(img, (480, int(img.shape[0] * ratio)))

        # converting to YCbCr colour space (illuminance, Chromium blue, Chormium red)
        imgYCbCr = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2YCR_CB)

        # discarding the illumninance paramter
        imgYCbCr = imgYCbCr[:, :, 1:]

        # saving original image dimensions
        original_shape = imgYCbCr.shape

        # rechaping to the 2D matrix for each line to a single vector
        reshapedImg = imgYCbCr.reshape((-1, 2))

        # applying guassian mixture model with 2 components (background, skin-like colour)
        gmm_model = self.GMM_model.fit(reshapedImg)
        gmm_labels = gmm_model.predict(reshapedImg)

        # from researching on the internet, the RGB color (232, 190, 172) is a typical natural skin
        typicalSkinColorRGB = np.array([[[232, 190, 172]]], dtype=np.uint8)

        # convert to YCbCr color space
        typicalSkinColorYCbCr = cv2.cvtColor(typicalSkinColorRGB, cv2.COLOR_RGB2YCR_CB)
        typicalSkinColorYCbCr = typicalSkinColorYCbCr[:, :, 1:]
        typicalSkinColorYCbCr = typicalSkinColorYCbCr.reshape((-1, 2))

        # get the which cluster label represent the skin color
        typicalSkinColourLabel = gmm_model.predict(typicalSkinColorYCbCr)

        # since the output of guassian mixtures is 0,1,2 and range of colours from 0 to 255 so we map colours to be seen
        gmm_labels[gmm_labels == typicalSkinColourLabel] = 255
        gmm_labels[gmm_labels != 255] = 0

        # converting the labels to an image
        maskImage = gmm_labels.astype(np.uint8)
        maskImage = maskImage.reshape(original_shape[0], original_shape[1])

        # apply openning and closing to clean noise and fill black holes
        structureElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        maskImage = cv2.morphologyEx(maskImage, cv2.MORPH_OPEN, structureElement)
        maskImage = cv2.morphologyEx(maskImage, cv2.MORPH_CLOSE, structureElement)

        # get the counter of the largest area and fill it with white to remove unwanted noise
        contours, _ = cv2.findContours(maskImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxAreaContour = max(contours, key=cv2.contourArea)
        blankImage = np.zeros(maskImage.shape, dtype='uint8')
        cv2.drawContours(blankImage, [maxAreaContour], 0, 255, cv2.FILLED)
        maskImage = blankImage

        # apply bit wise and with the original image
        maskImage = np.repeat(maskImage[:, :, np.newaxis], 3, axis=2)
        segmentedImgDueToGMM = cv2.bitwise_and(resizedImage, maskImage)

        # preparing image for Kmeans  (reshping the image)
        reshapedImg = cv2.cvtColor(segmentedImgDueToGMM, cv2.COLOR_BGR2YCR_CB)
        reshapedImg = reshapedImg[:, :, 1:]
        reshapedImg = reshapedImg.reshape((-1, 2))

        # fitting and predicting the images
        predictedKmeans2 = self.KMeans2_model.fit_predict(reshapedImg)

        # predicting the skin color
        skinLabelKmeans2 = self.KMeans2_model.predict(typicalSkinColorYCbCr)

        # writing 255 at skin color pixels (constructing the mask)
        predictedKmeans2[predictedKmeans2 == skinLabelKmeans2] = 255
        predictedKmeans2[predictedKmeans2 != 255] = 0


        # check what K is more meaningful to be used
        usedImg = predictedKmeans2


        # reshaping the image
        segmentedImg = usedImg.reshape(original_shape[0], original_shape[1]).astype(np.uint8)

        # apply openning and closing to clean noise and fill black holes
        structureElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        segmentedImg = cv2.morphologyEx(segmentedImg, cv2.MORPH_CLOSE, structureElement)
        segmentedImg = cv2.morphologyEx(segmentedImg, cv2.MORPH_OPEN, structureElement)

        # get the counter of the largest area and fill it with white to fill black holes
        contours, _ = cv2.findContours(segmentedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxAreaContour = max(contours, key=cv2.contourArea)
        blankImage = np.zeros(segmentedImg.shape, dtype='uint8')
        cv2.drawContours(blankImage, [maxAreaContour], 0, 255, cv2.FILLED)
        segmentedImg = blankImage

        # apply bit wise and with the original image
        maskImage2 = np.repeat(segmentedImg[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        segmentedImgDueToKMeans = cv2.bitwise_and(resizedImage, maskImage2)

        
        # fill background with White
        backGround = np.invert(segmentedImg.astype(np.uint8))
        backGround = np.repeat(backGround[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        segmentedImgDueToKMeans = cv2.bitwise_or(segmentedImgDueToKMeans, backGround)

        # saving the image
        segmented = cv2.cvtColor(segmentedImgDueToKMeans, cv2.COLOR_BGR2GRAY)

        segmented = cv2.blur(segmented, (10, 10)) 
        ratio = 64 / segmented.shape[1]
        resizedImage = cv2.resize(segmented, (64, int(segmented.shape[0] * ratio)))
        
        return resizedImage #cv2.resize(segmented, (64, 64))
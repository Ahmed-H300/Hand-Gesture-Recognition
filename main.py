import cv2
import os
import numpy as np
from config.constants import *
from src.preProcess import PreprocessModel
from src.sift import SIFT
from src.hog import HOG
from src.SVM import SVM
from src.performanceAnalysis import Utils


# Function test
def test():
        # list for labels
        labels = np.array([])
        # create the preprocces model
        preprocessModel = PreprocessModel()
        # create the feature extraction model
        #choose whether SIFT or HOG to RUN
        model = None
        if hog_or_sift == 'sift':
                # create SIFT Model
                model = SIFT()
        elif hog_or_sift == 'hog':
                # create HOG Model
                model = HOG()
        else:
                print('wrong choice for SIFT or HOG')
                exit()
        # create the utils module
        utils = Utils()
        # load SVM train file
        svm = utils.getModel(output_dir, 'svm_model')
        # set the train path
        path_test = data_dir_test
        # loop through the original images
        for filename in os.listdir(path_test):
                # Load image
                img_path = os.path.join(path_test, filename)
                img = cv2.imread(img_path)
                # preprocess the image
                Image = preprocessModel.preProcess(img)
                # get the features
                descriptors = model.compute(Image)
                # predict with SVM
                prediction = svm.predict([descriptors])
                # append it to labels
                labels = np.append(labels, prediction)
        
        # write labels in file output
        # note 0 means one decimal place
        utils.writeListToFile(labels.astype(np.float32), output_dir, results_file_name, 0)
        
        # Training Ended
        print('Test Ended Successfully')
        # Exit Training
        print('Exit Testing...')

# Train Function
def train():
        # list for labels
        labels = np.array([])
        # list for features
        features = []
        # create the preprocces model
        preprocessModel = PreprocessModel()
        # create the feature extraction model
        #choose whether SIFT or HOG to RUN
        model = None
        if hog_or_sift == 'sift':
                # create SIFT Model
                model = SIFT()
        elif hog_or_sift == 'hog':
                # create HOG Model
                model = HOG()
        else:
                print('wrong choice for SIFT or HOG')
                exit()
        # create the SVM module
        svm = SVM()
        # create the utils module
        utils = Utils()
        # set the train path
        path_train = data_dir_train
        # loop through the original images
        for class_name in os.listdir(path_train):
                class_dir = os.path.join(path_train, class_name)
                for foldername in os.listdir(class_dir):
                    folder_path = os.path.join(class_dir,foldername)
                    for filename in os.listdir(folder_path):
                        # insert label
                        labels = np.append(labels, foldername)
                        # Load image
                        img_path = os.path.join(folder_path, filename)
                        img = cv2.imread(img_path)
                        # preprocess the image
                        Image = preprocessModel.preProcess(img)
                        # get the features
                        descriptors = model.compute(Image)
                        # insert the feature in feature list
                        features.append(descriptors)
        
        features = np.array(features)
        # Train SVM
        svm_model = svm.train(features, labels)
        # save the model
        utils.saveModel(svm_model, output_dir, 'svm_model')
        # Training Ended
        print('Training Ended Successfully')
        # Exit Training
        print('Exit Training...')

# print the welcome message
def print_welcome_message():
        # print the welcome message
        print('''
                Welcome to Hand Geasture Recognition
                in the folder data
                you will find two sub-directories
                1. train
                2. test
                the /test directory will contain the files as follows :
                1.png
                2.png
                3.png
                .....
                while the /train directory will contain the files as following
                        /train
                        |
                men <- -> women
                |          |
                0          0
                1          1
                2          2
                3          3
                4          4
                5          5
                and each number from 0 -> 5 is a subdirectory that contains images as follows:
                1.png
                2.png
                3.png
                .....

        ''')

# main function
def main():

        # print the welcome message
        print_welcome_message()

        # get the choice whether to run train or test
        userChoice = int(input("enter 1 to to test, 2 to train.\n"))

        # if the user choice is 1 then run test
        if userChoice == 1:
                print("Entering test mode...")
                test()
        # if the user choice is 2 then run train
        elif userChoice == 2:
                print("Entering train mode...")
                train()
        # if the user choice is not 1 or 2 then print wrong choice
        else:
                print("wrong choice")
        
        # print exit
        print('Exit')


if __name__ == '__main__':

        # enter main function
        main()


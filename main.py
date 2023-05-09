import cv2
import os
import numpy as np
from config.constants import *
from src.preProcess import PreprocessModel
from src.sift import SIFT
from src.hog import HOG
from src.SVM import SVM
from src.performanceAnalysis import Utils
from sklearn.model_selection import train_test_split



# Function test
def test():
        # list for labels
        labels = np.array([])
        # create the preprocces model
        preprocessModel = PreprocessModel()
        # create the utils module
        utils = Utils()
        # create the feature extraction model
        #choose whether SIFT or HOG to RUN
        model = None
        min_feature_length = None
        if hog_or_sift == 'sift':
                # create SIFT Model
                model = SIFT()
                # load min_feature_length
                min_feature_length_file_name = 'min_feature_length_sift'
                min_feature_length = utils.loadListFromFile(output_dir, min_feature_length_file_name)
                min_feature_length = int(min_feature_length)
        elif hog_or_sift == 'hog':
                # create HOG Model
                model = HOG()
        else:
                print('wrong choice for SIFT or HOG')
                exit()
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
                if hog_or_sift == 'sift':
                        descriptors = descriptors[:min_feature_length]
                        # Flatten the descriptors to 1D array
                        descriptors = descriptors.flatten()
                
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
        # list to store feature lengths
        feature_lengths = None
        # create the preprocces model
        preprocessModel = PreprocessModel()
        # create the feature extraction model
        #choose whether SIFT or HOG to RUN
        model = None
        if hog_or_sift == 'sift':
                # create SIFT Model
                model = SIFT()
                # initalize feature_lengths with empty array
                feature_lengths = []
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
                        if hog_or_sift == 'sift':
                                # apped descriptors length
                                feature_lengths.append(len(descriptors))
                        
                        # insert the feature in feature list
                        features.append(descriptors)
        
        if hog_or_sift == 'sift':
                # Determine the minimum feature length
                min_feature_length = min(feature_lengths)
                # Truncate the descriptors based on the minimum feature length and flatten them
                features = [descriptors[:min_feature_length].flatten() for descriptors in features]
                #load min_feature_length into file
                min_feature_length_file_name = 'min_feature_length_sift'
                utils.writeListToFile([min_feature_length], output_dir, min_feature_length_file_name, 0)

        # convert features to np array
        features = np.array(features)

        # # Split into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # if len(X_train) == 0 or len(X_test) == 0:
        #         print("Not enough samples to split into training and testing sets.")
        #         return
        # Train SVM
        # svm_model = svm.train(X_train, y_train)
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


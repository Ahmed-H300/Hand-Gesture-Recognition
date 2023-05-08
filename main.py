import cv2
import os
from config.constants import *
from src.preProcess import PreprocessModel
from src.sift import SIFT


# TODO
def test():
        path_test = os.path.join(data_dir, 'test')
        pass

# TODO
def train():
        ################################
        # To be removed when the funciton is completed
        i = 0
        ################################
        # create the preprocces model
        preprocessModel = PreprocessModel()
        # create the sift model
        sift = SIFT()
        # set the train path
        path_train = os.path.join(data_dir, 'train')
        # set the original images path
        path_original = os.path.join(path_train, 'original')
        # loop through the original images
        for class_name in os.listdir(path_original):
                class_dir = os.path.join(path_original, class_name)
                for foldername in os.listdir(class_dir):
                    folder_path = os.path.join(class_dir,foldername)
                    for filename in os.listdir(folder_path):
                        # Load image
                        img_path = os.path.join(folder_path, filename)
                        img = cv2.imread(img_path)

                        # preprocess the image
                        Image = preprocessModel.preProcess(img)

                        # get the sift keypoints and discreptors
                        keypoints, descriptors = sift.compute(Image)

                        # TODO to be continued

                        ################################
                        # To be removed when the funciton is completed
                        # save the image
                        img_keypoints = sift.draw_descriptors(keypoints, Image)
                        # save the image with the keypoints on it
                        cv2.imwrite(os.path.join(output_dir,f'{i}.jpg'), img_keypoints)
                        i += 1
                        ################################

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


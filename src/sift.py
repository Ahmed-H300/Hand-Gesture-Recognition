import cv2
import os
import numpy as np


class SIFT:
    
    # Path to dataset folder
    data_dir = ''

    # Path to output folder
    output_dir = ''

    # SIFT object
    sift = None
    
    # to initialize the SIFT object and the directory structure
    def __init__(self, data_dir = os.path.join('data', 'processed'), output_dir = os.path.join('data', 'sift')):
        # Set path to dataset folder
        self.data_dir = data_dir
        # Set path to output folder
        self.output_dir = output_dir
        # Initialize SIFT object
        self.sift = cv2.SIFT_create()



    """
    This function computes SIFT descriptors for either the train, test, or both datasets, 
    based on the specified operation. 
    If showImage is set to True, the image will be displayed. The output is saved in the [output_dir] directory.

    Parameters:
    - operation: string, specifies which dataset to compute descriptors for ('train', 'test', or 'both')
    - showImage: bool, indicates whether or not to display the image

    Returns:
    None

    """
    def compute(self, operation = 'both', showImage = False):
        operation_array = []
        if operation == 'both':
            operation_array = ['train', 'test']
        else:
            operation_array.append(operation)
        # Loop through training images and compute SIFT descriptors
        for operation_type in operation_array:
            path = os.path.join(self.data_dir, operation_type)
            for class_name in os.listdir(path):
                class_dir = os.path.join(path, class_name)
                for foldername in os.listdir(class_dir):
                    folder_path = os.path.join(class_dir,foldername)
                    for filename in os.listdir(folder_path):
                        # Load image
                        img_path = os.path.join(folder_path, filename)
                        print(img_path)
                        img = cv2.imread(img_path)
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Detect keypoints and compute descriptors
                        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

                        # Save descriptors to file
                        out_path = os.path.join(self.output_dir, operation_type, class_name, foldername)
                        output_file = os.path.join(out_path, filename.split('.')[0] + '.npy')
                        np.save(output_file, descriptors)
                        if showImage:
                            # Marking the keypoint on the image using circles
                            img_keypoints =cv2.drawKeypoints(gray ,keypoints ,img ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                            cv2.imwrite(os.path.join(out_path, filename), img_keypoints)


if __name__ == '__main__':
    # run the sift
    print('''
    choose:
    1. train
    2. test
    3. both
    ''')
    choice = input()
    operation = 'both'    
    if choice == '1':
        operation = 'train'
    elif choice == '2':
        operation = 'test' 
    print('''
    choose:
    1. show image
    2. don't show image
    ''')
    choice = input()
    showImage = False
    if choice == '1':
        showImage = True

    sift = SIFT()
    sift.compute(operation=operation, showImage=showImage)
    
# Hand Gesture Recognition

This project is focused on Hand Gesture Recognition, utilizing machine learning techniques to classify hand gestures from image data. The pipeline of the project involves different modes of operation: Test mode, Train mode, Train and Test mode, and Grid Search mode. Each mode serves a specific purpose within the project.

## Getting Started

To begin, the project displays a welcome message to the user and prompts them to choose a mode of operation.

Modes available:

- Test mode
- Train mode
- Train and test mode
- Grid search mode

If an invalid choice is made, the program terminates.

## Test Mode

In the Test mode, the model is loaded, and the program iterates over the test data. It performs preprocessing on each image, makes predictions using the trained model, and appends the predicted labels to an array. Finally, the predicted labels are saved.

## Train Mode

The Train mode involves looping over the training data, located in sub-folders. For each sub-folder, the program performs preprocessing, extracts features, and appends them to a feature array. Once the feature array is prepared, the model is trained using these features. Finally, the trained model is saved.

## Train and Test Mode

In the Train and Test mode, the data set is split into training data and testing data. Approximately 80% of the data is used for training, while 20% is reserved for testing the trained model. Similar to the Train mode, preprocessing and feature extraction are performed on the training data. The model is then trained using the extracted features. After training, the model is saved. Subsequently, the model is tested on the 20% reserved testing data, and the accuracy of the model's predictions is calculated.

## Grid Search Mode

The Grid Search mode is used to perform hyperparameter tuning for the SVM classifier. The program defines a parameter grid containing different values for the regularization parameter, kernel coefficient, kernel type, and degree. The data set is split into training data and testing data, with an 80/20 ratio. Grid search is then executed to find the best combination of parameters. Once the best parameters are determined, the model is trained with these parameters using the training set, and the accuracy of the model is calculated on the testing set.

## Preprocessing Module

The preprocessing module follows the approach proposed by researchers who state that skin color follows a Gaussian distribution in the YCbCr color space. The preprocessing steps are as follows:

1. Resize the image to a width of 480 pixels, maintaining the aspect ratio of the original image, to reduce computational complexity.
2. Convert the resized image from the BGR color space to the YCbCr color space, excluding the illumination component to eliminate the effect of lighting on the skin color.
3. Reshape the image into a matrix, where each row represents a point and the two columns represent the chromium red and chromium blue values.
4. Fit a Gaussian Mixture Model (GMM) on the image using the reshaped matrix. The GMM predicts the labels of each pixel in the image, with parameters such as the number of clusters and initialization method.
5. To determine which cluster represents skin and which represents the background, a reference value called "typicalSkinColor" is inputted to the GMM, and the model predicts its label.
6. Perform morphological opening and closing operations on the mask image obtained from the GMM to remove noise and fill black holes.
7. Find the contour with the maximum area in the mask image, which represents the hand. Fill the contour and remove any other unattached skin-like colors from the background.
8. Apply K-means clustering with K=2 for further enhancement of the image. Repeat steps 5 and 6. Finally, return

 the image reshaped to 64x64 pixels.

## Feature Extraction/Selection Module

The feature extraction module applies Histogram of Oriented Gradients (HOG) on the preprocessed image. The HOG parameters include the number of orientations, pixels per cell, cells per block, visualization, and block normalization. HOG captures local edge patterns by computing gradient magnitudes and orientations in image cells, and the histograms of orientations are normalized and concatenated to form a feature vector.

Additionally, Local Binary Pattern (LBP) is applied to the image. LBP compares pixel intensities with neighboring pixels to capture local image patterns.

Both the HOG and LBP features are merged into a single feature vector.

## Model Selection/Training Module

The chosen classifier for this project is Support Vector Machines (SVM). The SVM model is applied with specific parameters, including the choice of the SVM kernel and its parameters.

To improve the performance, a BaggingClassifier is used with SVM as the base estimator. BaggingClassifier is an ensemble meta-estimator that fits multiple base classifiers on random subsets of the original dataset and aggregates their predictions to form a final prediction. This is done in parallel to enhance performance.

## Performance Analysis Module

The Performance Analysis module includes various utility methods within the Utils class:

- `calculateAccuracy`: Calculates the accuracy after performing testing in the main program.
- `startTimePoint`: Retrieves the current time at a specific point in the program.
- `getElapsedTimeInSeconds`: Calculates the time difference between the start and end points.
- `writeListToFile`: Writes a list to a specified file.
- `saveModel`: Saves the trained model.
- `getModel`: Loads a saved model for testing.
- `loadListFromFile`: Loads a list from a specific file.

## Enhancements and Future Work

Several enhancements and future work possibilities have been identified for this project:

- Expanding the dataset size by collecting more hand gesture images to improve the model's performance.
- Exploring different parameter tuning for HOG to find optimal settings.
- Trying different tuning parameters for the SVM classifier.
- Exploring advanced techniques such as deep learning with convolutional neural networks (CNNs) and dynamic K determination using deep scans.

import cv2
import os
import numpy as np
from config.constants import *
from src.preProcess import PreprocessModel
from src.sift import SIFT
from src.hog import HOG
from src.SVM_threads import SVM
from src.rf import RF
from src.performanceAnalysis import Utils
from sklearn.model_selection import train_test_split
from natsort import natsorted
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report



# use with features discreptors to bag the feature into fixed size
def bag_of_features(features, centres, k = 200):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

# Function test
def test():
	# list for labels
	labels = np.array([])
	# create the preprocces model
	preprocessModel = PreprocessModel()
	# create the utils module
	utils = Utils()
	# initialize times array with empty list
	times = []
	# create the feature extraction model
	#choose whether SIFT or HOG to RUN
	model = None
	if model_type == 'sift':
			# create SIFT Model
			model = SIFT()
	elif model_type == 'hog':
			# create HOG Model
			model = HOG()
	else:
			print('wrong choice for SIFT or HOG')
			exit()
	# load classifier train file
	classifier = None
	if classifier_type == 'svm':
		classifier = utils.getModel(output_dir, f'{classifier_type}_model')
	elif classifier_type == 'rf':
		classifier = utils.getModel(output_dir, f'{classifier_type}_model')
	else:
		print('wrong choice for SVM or RF')
		exit()
	# set the train path
	path_test = data_dir_test
	# loop through the original images
	images = os.listdir(path_test)
	images = natsorted(images)
	# get number of photos in the folder
	number_of_photos = len(os.listdir(path_test))
	photo_counter = 0
	# load centers in case of not hog
	centres = None
	if model_type != 'hog':
		# load centers
		centres = np.load(os.path.join(output_dir, 'centers.npy'))
	for filename in images:
		if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')): 
			# Load image
			img_path = os.path.join(path_test, filename)
			img = cv2.imread(img_path)
			# start timer
			######################################################
			utils.startTimePoint()
			# preprocess the image
			Image = preprocessModel.preProcess(img)
			# get the features
			descriptors = model.compute(Image)
			if model_type != 'hog':
				# convert features to np vstack
				descriptors = bag_of_features(descriptors, centres, k)

			if model_type != 'hog':
				# convert features to np vstack
				descriptors = np.vstack(descriptors)
				# predict with classifier
				prediction = classifier.predict(descriptors)
			else:
				# predict with classifier
				prediction = classifier.predict([descriptors])
			# stop timer
			######################################################
			elapsedTime = utils.getElapsedTimeInSeconds()
			times.append(elapsedTime)
			# append it to labels
			labels = np.append(labels, prediction)
			photo_counter += 1
			print(f"{photo_counter/number_of_photos:.2%} ", end='')
			print('\r', end='', flush=True)
    # get a new line          
	print('\n')
	
	# write labels in file output
	# note 0 means one decimal place
	utils.writeListToFile(labels.astype(np.float32), output_dir, results_file_name, 0)
	# write times
	utils.writeListToFile(np.array(times), output_dir, times_file_name, 3)
	
	# Training Ended
	print('Test Ended Successfully')
	# Exit Training
	print('Exit Testing...')

# built in train
def train_builtin():
	# list for labels
	labels = np.array([])
	# list for features
	features = []
	# create the preprocces model
	preprocessModel = PreprocessModel()
	# create the feature extraction model
	#choose whether SIFT or HOG to RUN
	model = None
	if model_type == 'sift':
		# create SIFT Model
		model = SIFT()
	elif model_type == 'hog':
		# create HOG Model
		model = HOG()
	else:
		print('wrong choice for SIFT or HOG')
		exit()
	# set the train path
	path_train = data_dir_train
	# loop through the original images
	for class_name in os.listdir(path_train):
		class_dir = os.path.join(path_train, class_name)
		for foldername in os.listdir(class_dir):
			folder_path = os.path.join(class_dir,foldername)
			# get number of photos in the folder
			number_of_photos = len(os.listdir(folder_path))
			photo_counter = 0
			for filename in os.listdir(folder_path):
				if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
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
					if descriptors is not None:
						features.append(descriptors)
					photo_counter += 1
					print(f"{photo_counter/number_of_photos:.2%} ", end='')
					print('\r', end='', flush=True)
			
			print(f'\n{folder_path} Finished Training.')
						
	
	if model_type != 'hog':
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		# convert features to np vstack
		features = np.vstack(features)
		# start clustering
		compactness, labels_sift, centres = cv2.kmeans(features, k, None, criteria, 10, flags)
		# save centers
		np.save(os.path.join(output_dir, 'centers'), centres)
		# empty the features
		features = []
		# loop through the original images
		for class_name in os.listdir(path_train):
			class_dir = os.path.join(path_train, class_name)
			for foldername in os.listdir(class_dir):
				folder_path = os.path.join(class_dir,foldername)
				# get number of photos in the folder
				number_of_photos = len(os.listdir(folder_path))
				photo_counter = 0
				for filename in os.listdir(folder_path):
					if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
						# Load image
						img_path = os.path.join(folder_path, filename)
						img = cv2.imread(img_path)
						# preprocess the image
						Image = preprocessModel.preProcess(img)
						# get the features
						descriptors = model.compute(Image)
						# insert the feature in feature list
						if descriptors is not None:
							descriptors = bag_of_features(descriptors, centres, k)
							features.append(descriptors)
						photo_counter += 1
						print(f"{photo_counter/number_of_photos:.2%} ", end='')
						print('\r', end='', flush=True)
				
				print(f'\n{folder_path} Finished Converting Descriptors.')
	
	if model_type == 'hog':
		# convert features to np array
		features = np.array(features)
	else:
		# convert features to np vstack
		features = np.vstack(features)

	# return features and labels
	return (features, labels)

# Train Function
def train():
	
	# load classifier train file
	classifier = None
	if classifier_type == 'svm':
		classifier = SVM()
	elif classifier_type == 'rf':
		classifier = RF()
	else:
		print('wrong choice for SVM or RF')
		exit()
	# create the utils module
	utils = Utils()
	# train
	features, labels = train_builtin()
	# Train classifier
	classifier_model = classifier.train(features, labels)
	# save the model
	utils.saveModel(classifier_model, output_dir, f'{classifier_type}_model')
	print('Model Saved Successfully')
	# Training Ended
	print('Training Ended Successfully')
	# Exit Training
	print('Exit Training...')

# Train and Test Function
def train_test():
	
	# train
	features, labels = train_builtin()

	# load classifier train file
	classifier = None
	if classifier_type == 'svm':
		classifier = SVM()
	elif classifier_type == 'rf':
		classifier = RF()
	else:
		print('wrong choice for SVM or RF')
		exit()
	# create the utils module
	utils = Utils()

	# Split into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
	if len(X_train) == 0 or len(X_test) == 0:
			print("Not enough samples to split into training and testing sets.")
			return
	# Train SVM
	classifier_model = classifier.train(X_train, y_train)
	# save the model
	utils.saveModel(classifier_model, output_dir, f'{classifier_type}_model')
	print('Model Saved Successfully')
	# Training Ended
	print('Training Ended Successfully')
	# Test the model
	print('Start Testing')
	predictions = classifier_model.predict(X_test)
	# calculte accuracy
	accuracy = utils.calculateAccuracy(predictions, y_test)
	# make it %100
	accuracy *= 100
	# print accuracy
	print("Accuracy {:.2f}".format(accuracy))
	# Training Ended
	print('Test Ended Successfully')
	# Exit Training
	print('Exit Train and test...')

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


def svm_gridSearch():

	# train
	features, labels = train_builtin()

	# Define the SVM parameters for grid search
	param_grid = {
    'C': [1, 10, 100],  # Regularization parameter
    'gamma': [0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Kernel type
	}

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

	# Create an SVM classifier
	svm = SVC()

	# Perform grid search to find the best hyperparameters
	grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
	grid_search.fit(X_train, y_train)

	# Get the best parameters and best score
	best_params = grid_search.best_params_
	best_score = grid_search.best_score_

	# Print the best parameters and best score
	print("Best Parameters: ", best_params)
	print("Best Score: ", best_score)

	# Fit the SVM classifier with the best parameters
	svm = SVC(**best_params)
	svm.fit(X_train, y_train)

	# Make predictions on the test data
	y_pred = svm.predict(X_test)

	# Evaluate the performance
	print(classification_report(y_test, y_pred))

	# save the model
	utils = Utils()
	utils.saveModel(svm, output_dir, f'{classifier_type}_model')	




# main function
def main():

	# print the welcome message
	print_welcome_message()

	# get the choice whether to run train or test
	userChoice = int(input("enter 1 to to test, 2 to train, 3 to test and train, 4 to preform svm grid search.\n"))

	# if the user choice is 1 then run test
	if userChoice == 1:
			print("Entering test mode...")
			test()
	# if the user choice is 2 then run train
	elif userChoice == 2:
			print("Entering train mode...")
			train()
	# if the user choice is 3 then run train and test
	elif userChoice == 3:
			print("Entering train and test mode...")
			train_test()
	# if the user choice is 4 then run grid search on svm
	elif userChoice == 4:
			print("Grid Search...")
			svm_gridSearch()			
	# if the user choice is not 1 or 2 then print wrong choice
	else:
			print("wrong choice")
	
	# print exit
	print('Exit')


if __name__ == '__main__':

	# enter main function
	main()


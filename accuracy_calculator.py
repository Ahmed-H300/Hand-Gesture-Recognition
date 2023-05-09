import numpy as np
from config.constants import *
from src.performanceAnalysis import Utils


if __name__ == '__main__':
    # define utils
    utils = Utils()
    # load prediced
    predicted = utils.loadListFromFile(output_dir, results_file_name) # load predicted

    # load actual
    actual = utils.loadListFromFile(output_dir, actual_file_name) # load actual

    # calculte accuracy
    accuracy = utils.calculateAccuracy(predicted, actual)

    # make it %100
    accuracy *= 100

    # print accuracy
    print("Accuracy {:.2f}".format(accuracy))

    # write accuracy to file
    utils.writeListToFile(np.array([accuracy]), output_dir, 'accuracy', 2)
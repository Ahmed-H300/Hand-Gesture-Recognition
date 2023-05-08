import numpy as np
import time
import joblib


class Utils:

    def calculateAccuracy(self, predicted, actual):
        return np.sum(predicted == actual) / np.size(predicted)

    def startTimePoint(self):
        self.StartTime = time.time()

    def getElapsedTimeInSeconds(self):
        return time.time() - self.StartTime

    def writeListToFile(self, list, path, filename, numOfDecimalPlaces):
        np.savetxt(path+'/'+filename+'.txt', list, fmt='%.'+str(numOfDecimalPlaces)+'f')

    def saveModel(self, model, path, filename):
        joblib.dump(model, path+'/'+filename+'.joblib')

    def getModel(self, path, filename):
        return joblib.load(path+'/'+filename+'.joblib')

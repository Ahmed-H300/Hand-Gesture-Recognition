from sklearn import svm


class SVM:

    def __init__(self):
        # creating a support vector machine model
        self.SVM_Model = svm.SVC(kernel='rbf')

    def train(self, ListOfImageFeatureList, lables):
        self.SVM_Model.fit(ListOfImageFeatureList, lables)
        return self.SVM_Model

    def predict(self, ListOfImageFeatureList):
        return self.SVM_Model.predict(ListOfImageFeatureList)
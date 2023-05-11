from sklearn import svm
from sklearn.ensemble import BaggingClassifier

class SVM:

    def __init__(self):
        # creating a support vector machine model
        self.SVM_Model = BaggingClassifier(estimator=svm.SVC(kernel='rbf'), n_estimators=10, random_state=0)

    def train(self, ListOfImageFeatureList, lables):
        self.SVM_Model.fit(ListOfImageFeatureList, lables)
        return self.SVM_Model

    def predict(self, ListOfImageFeatureList):
        return self.SVM_Model.predict(ListOfImageFeatureList)
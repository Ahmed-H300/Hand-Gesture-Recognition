from sklearn.ensemble import RandomForestClassifier


class RF:

    def __init__(self):
        # creating a support vector machine model
        self.RF_Model = RandomForestClassifier(n_estimators = 10000, random_state = 42)

    def train(self, ListOfImageFeatureList, lables):
        self.RF_Model.fit(ListOfImageFeatureList, lables)
        return self.RF_Model

    def predict(self, ListOfImageFeatureList):
        return self.RF_Model.predict(ListOfImageFeatureList)
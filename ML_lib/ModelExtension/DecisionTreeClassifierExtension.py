from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from IPython.core.display import display
from sklearn.externals import joblib
from sklearn import tree


class DecisionTreeClassifierExtension:

    def __init__ (self, X_train, y_train):
        self.model = DecisionTreeClassifier()
        self.X_train = X_train.select_dtypes(include=['float64', 'int'])
        self.y_train = y_train
        self.y_test_predict = None
        self.y_test = None
        self.path_model = None
        self.path_tree = None
        self.model.fit(self.X_train, self.y_train)   

    def predict(self, X_test, y_test):
        if self.y_test is not None:
            display(f"model was trained")
            return 
        self.y_test = y_test
        self.y_test_predict = self.model.predict(X_test)

    def get_accuracy_score(self):
        result_accuracy_score = accuracy_score(self.y_test, self.y_test_predict)
        display(f"DecisionTreeClassifier accuracy: {result_accuracy_score}")
    
    def save_model_path(self, path):
        if self.path_model is not None:
            display(f"model was saved, use save_model method to save again")
        self.path_model = path
        joblib.dump(self.model, path)
    
    def save_model(self):
        if self.path_model is None:
            display(f"no path, use method save_model_path")
        else:
            joblib.dump(self.model, self.path_model)

    
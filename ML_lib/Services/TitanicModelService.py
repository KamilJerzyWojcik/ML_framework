from IPython.core.display import display
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

class TitanicModelService:
    
    def __init__ (self):
        self.simple_models = [
        ('LogisticRegression', LogisticRegression()), # l1, l2 regularyzacja
        ('DecisionTreeClassifier', DecisionTreeClassifier()),
        ('RandomForestClassifier', RandomForestClassifier()),
        ('ExtraTreesClassifier', ExtraTreesClassifier()),
        ('SGDClassifier', SGDClassifier(n_iter_no_change=50, penalty=None, eta0=0.1)),
        ('Ridge', RidgeClassifier(alpha=1, solver="cholesky"))
        ]

        self.final_models = [
        ]

    def titanic_predicts_final_models(self, train_strat_num_titanic, test_strat_num_titanic):
        # rozdzial trenujacych danych na X i y
        X_train = train_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
        y_train = train_strat_num_titanic['Survived']

        for name, model in self.final_models:
            # trenowanie modelu
            model.fit(X_train, y_train)

            # rozdzial testujących danych na X i y
            X_test = test_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
            y_test = test_strat_num_titanic['Survived']

            # walidacja modelu
            accuracy = self.get_predict_accuracy(model, X_test, y_test)

            display(f"{name}: accuracy: {accuracy}")

    def titanic_simple_models_final_test(self, train_strat_num_titanic, test_strat_num_titanic):
        # rozdzial trenujacych danych na X i y
        X_train = train_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
        y_train = train_strat_num_titanic['Survived']

        for name, model in self.simple_models:
            # trenowanie modelu
            model.fit(X_train, y_train)

            # rozdzial testujących danych na X i y
            X_test = test_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
            y_test = test_strat_num_titanic['Survived']

            # walidacja modelu
            accuracy = self.get_predict_accuracy(model, X_test, y_test)

            display(f"{name}: accuracy: {accuracy}")
    
    def get_predict_accuracy(self, model, X_test, y_test):
        y_test_prediction = model.predict(X_test)
        return accuracy_score(y_test, y_test_prediction)

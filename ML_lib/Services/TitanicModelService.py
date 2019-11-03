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
        ('DecisionTreeClassifier', DecisionTreeClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('ExtraTreesClassifier', ExtraTreesClassifier()),
        ('RandomForestClassifier', RandomForestClassifier()),
        ('Ridge', RidgeClassifier(alpha=1, solver="cholesky")),
        ]

        self.final_models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(
                random_state=42,
                max_depth=None,
                splitter='random',
                min_samples_split=10,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=10,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0005,
                presort=False,
            )),
            ('ExtraTreesClassifier', ExtraTreesClassifier(
                random_state=42,
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='sqrt',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                warm_start=False
            )),
            ('LogisticRegression', LogisticRegression(
                random_state=42,
                penalty='l2',
                dual=False,
                tol=0.01,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=10.0,
                class_weight=None,
                solver='liblinear',
                max_iter=100,
                multi_class='ovr'
            )),
            ('RandomForestClassifier', RandomForestClassifier(
                random_state=42,
                n_estimators=10,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False
            )),
            ('Ridge', RidgeClassifier(
                random_state=42,
                alpha=10.0,
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                max_iter=10,
                solver='sparse_cg',
            )),
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

    def get_final_models(self):
        return self.final_models



# Best parameters
# DecisionTreeClassifierDf = pd.read_csv('DecisionTreeClassifier/Final_model_DecisionTreeClassifier_random_grid_1572808637.229387.csv')
# ExtraTreesClassifierDf = pd.read_csv('ExtraTreesClassifier/Final_model_ExtraTreesClassifier_1572806224.4869351.csv')
# LogisticRegresionClassifierDf = pd.read_csv('LogisticRegresionClassifier/final_model_LogisticRegresionClassifier_random_grid_1572800415.514078.csv')
# RandomForestClassifierDf = pd.read_csv('RandomForestClassifier/Final_model_RandomTreeClasifier_20191102.csv')
# RidgeClassifierDf = pd.read_csv('RidgeClassifier/Final_modelRidgeClassifier_1572803009.925738.csv')

# Best accuracy cv=3
# random tree        0.8455056179775281
# logistic regresion 0.808988764044943
# ridge              0.814606741573033
# extra tree .       0.832865
# decision tree .    0.8286516853932581
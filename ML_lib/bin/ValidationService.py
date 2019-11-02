from sklearn.model_selection import cross_validate

class ValidationService:

    def get_cross_validate_accuracy(self, model, X, y):
        return cross_validate(model, X, y, scoring='accuracy', cv=3, return_train_score=True)
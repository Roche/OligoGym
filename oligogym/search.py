from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from oligogym.models import SKLearnModel


class RandomSearch(SKLearnModel, BaseEstimator):
    def __init__(self, model=None, search=None):
        self.model = model
        self.search = search

    def random_search(
        self,
        param_distributions,
        scoring="r2",
        n_jobs=1,
        cv=10,
        **search_kwargs,
    ):
        
        self.search = RandomizedSearchCV(
            estimator=self.model.model,
            param_distributions=param_distributions,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            **search_kwargs,
        )

        return self

    def fit(self, X, y):
        self.search.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.search.predict(X)
    
    @property
    def best_estimator_(self):
        return self.search.best_estimator_
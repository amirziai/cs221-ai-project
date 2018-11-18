from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator
from numpy import ndarray
from typing import Callable, List


N_ESTIMATORS = 100


class ActiveLearning:
    def __init__(self,
                 x_training: ndarray,
                 y_training: ndarray,
                 x_pool: ndarray,
                 estimator: BaseEstimator = RandomForestClassifier(n_estimators=N_ESTIMATORS),
                 query_strategy: Callable[[BaseEstimator, ndarray], ndarray] = uncertainty_sampling):
        self.x_pool = x_pool
        self.learner = ActiveLearner(
            estimator=estimator,
            query_strategy=query_strategy,
            X_training=x_training, y_training=y_training,
        )

    def query(self) -> int:
        index, _ = self.learner.query(self.x_pool)
        return index

    def teach(self, x: ndarray, y: ndarray) -> None:
        self.learner.teach(x, y)


class ActiveLearningSimulator(ActiveLearning):
    def __init__(self,
                 x_training: ndarray,
                 y_training: ndarray,
                 x_dev: ndarray,
                 y_dev: ndarray,
                 x_test: ndarray,
                 y_test: ndarray,
                 x_pool: ndarray,
                 y_pool: ndarray,
                 estimator: BaseEstimator = RandomForestClassifier(n_estimators=N_ESTIMATORS),
                 query_strategy: Callable[[BaseEstimator, ndarray], ndarray] = uncertainty_sampling):
        super().__init__(x_training, y_training, x_pool, estimator, query_strategy)
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.x_test = x_test
        self.y_test = y_test
        self.y_pool = y_pool
        self.already_sampled_indices = []

    def simulate(self) -> List[float]:
        scores = []

        for _ in range(len(self.x_pool)):
            index = self.query()
            self.already_sampled_indices.append(index)
            self.teach(self.x_pool[index], self.y_pool[index])
            score = self.learner.score(self.x_dev, self.y_dev)
            scores.append(score)

        return scores

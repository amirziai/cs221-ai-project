import os
import pickle
from time import time
from typing import Dict, Iterable, Union, Optional, List, Callable, Tuple, NamedTuple

import numpy as np
import pandas as pd
import requests
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats.distributions import entropy
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised.label_propagation import BaseLabelPropagation, LabelSpreading

import config
import util

Pandas = Union[DataFrame, Series]
Number = Union[int, float]
ActiveSemiSup = Union[BaseEstimator, BaseLabelPropagation]
Stats = Dict[str, Union[str, Number]]


class ActiveLearningData(NamedTuple):
    x_train_start: DataFrame
    y_train_start: Series
    x_train_pool: DataFrame
    y_train_pool: Series
    x_dev: np.ndarray
    y_dev: Series


def random_sampling(_, x_pool):
    n_samples = len(x_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], x_pool[query_idx]


class NetworkIntrusionDetection:
    def __init__(self):
        # config
        self.label_col = config.label_col
        self.label_normal = config.label_normal
        self.results_folder = 'results'
        self.label_threshold = config.label_threshold
        self.random_seed = config.random_seed
        self.fill_na = config.fill_na
        self.size_train = config.size_train
        self.size_dev = config.size_dev
        self.size_test = 1 - (self.size_train + self.size_dev)
        self.baseline_random_n = config.baseline_random_n
        self.clf_n_estimator = config.clf_n_estimator
        self.active_learning_n_initial = config.active_learning_n_initial
        self.active_learning_budget = config.active_learning_budget
        self.active_learning_rf = RandomForestClassifier(n_estimators=self.clf_n_estimator, n_jobs=-1,
                                                         random_state=self.random_seed)
        self.active_learning_lr = LogisticRegression(solver='lbfgs', random_state=self.random_seed)
        self.active_learning_learners = [self.active_learning_rf, self.active_learning_lr]
        self.active_learning_strategies = [random_sampling, entropy_sampling]
        self.active_learning_log_intervals = {1, 10, 25, 50, 100}
        self.active_learning_print_every = 25
        self.semi_supervised_class = LabelSpreading
        self.semi_supervised_class_args = {'kernel': 'knn', 'max_iter': 5, 'n_jobs': -1}
        self.ensemble_weights = {'rf': 2, 'lr': 1, 'iforest': 1, 'lp': 1}

        np.random.seed(self.random_seed)

        # get the data from source
        self.df = self._get_data_from_source()
        # create a dataset per attack label of [attack label, normal] rows, remove labels w/count < label_threshold
        self.dfs_by_label = self._get_dataframes_by_label()
        # featurize and separate labels
        self.features = self._featurize()
        # split into train/dev/test
        self.splits = self._split()

    def _featurize(self) -> Dict[str, Dict[str, Pandas]]:
        return {
            label: {'x': self._get_x(self.dfs_by_label[label]), 'y': self._get_y(self.dfs_by_label[label])}
            for label in self.dfs_by_label
        }

    def _split(self) -> Dict[str, Dict[str, Dict[str, Pandas]]]:
        return {
            label: self._split_label(self.features[label]['x'], self.features[label]['y'])
            for label in self.features
        }

    def _split_label(self, x: DataFrame, y: Series) -> Dict[str, Pandas]:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=self.size_test,
                                                            random_state=self.random_seed,
                                                            stratify=y
                                                            )
        split_size = self.size_dev / (1 - self.size_test)
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                          test_size=split_size,
                                                          random_state=self.random_seed,
                                                          stratify=y_train
                                                          )
        return {
            'x_train': x_train, 'x_dev': x_dev, 'x_test': x_test,
            'y_train': y_train, 'y_dev': y_dev, 'y_test': y_test
        }

    def _get_data_from_source(self) -> DataFrame:
        headers_link = config.headers_link
        file_path = config.file_path
        contents = requests.get(headers_link).text
        headers = [x.split(':')[0] for x in contents.split('\n')[1:]]
        headers[-1] = self.label_col
        return pd.read_csv(file_path, header=None, names=headers)

    @staticmethod
    def _get_metrics(actual: Iterable, predicted: Iterable,
                     scores: Optional[Iterable[float]] = None) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        metrics = {
            'precision': precision_score(actual, predicted),
            'recall': recall_score(actual, predicted),
            'f1': f1_score(actual, predicted),
            'FP': fp,
            'FN': fn
        }
        metrics_threshold = {} if scores is None else {'roc auc': roc_auc_score(actual, scores),
                                                       'average precision': average_precision_score(actual, scores)}
        return util.merge_dicts(metrics, metrics_threshold)

    def _get_dataframes_by_label(self) -> Dict[str, DataFrame]:
        return {
            label: self.df[self.df[self.label_col].isin([self.label_normal, label])].reset_index(drop=True)
            for label, count in self.df[self.label_col].value_counts().items()
            if label != self.label_normal and count >= self.label_threshold
        }

    def _get_x(self, df: DataFrame) -> DataFrame:
        # detach the label
        x = df[df.columns.difference([self.label_col])]
        # one-hot encode and fill NaN
        categorical_cols = [col for col in x.columns if x.dtypes[col] == object]
        return (pd.get_dummies(x, columns=categorical_cols)
                .fillna(self.fill_na))

    def _get_y(self, df: DataFrame) -> Series:
        return df[self.label_col] != self.label_normal

    def _calculate_baseline_oracle(self, label: str) -> Dict[str, Union[str, float]]:
        p = len(self.splits[label]['y_train'][self.splits[label]['y_train'] == True]) / len(
            self.splits[label]['y_train'])
        out = {'label': label, 'prevalence': p}

        # oracle
        clf = RandomForestClassifier(n_estimators=self.clf_n_estimator, n_jobs=-1, random_state=self.random_seed)
        start = time()
        clf.fit(self.splits[label]['x_train'], self.splits[label]['y_train'])
        elapsed = time() - start
        predictions = clf.predict(self.splits[label]['x_dev'])
        scores = clf.predict_proba(self.splits[label]['x_dev'])[:, 1]
        oracle = self._get_metrics(self.splits[label]['y_dev'], predictions, scores)
        oracle = util.merge_dicts(oracle, {'train time (s)': elapsed})
        oracle = util.add_prefix_to_dict_keys(oracle, 'oracle_')
        out = util.merge_dicts(out, oracle)

        # baselines
        # random with same prevalence
        prevalence = self.splits[label]['y_train'].value_counts(normalize=True)
        categories = np.array(prevalence.index).astype(bool)
        n = len(self.splits[label]['x_dev'])
        baseline_random = pd.DataFrame([
            self._get_metrics(actual=self.splits[label]['y_dev'],
                              predicted=np.random.choice(categories, p=prevalence, size=n))
            for _ in range(self.baseline_random_n)
        ]).median().to_dict()
        baseline_random = util.add_prefix_to_dict_keys(baseline_random, 'baseline_random_')
        out = util.merge_dicts(out, baseline_random)

        # majority
        # majority_class = prevalence.index[0]
        # baseline_majority = self._get_metrics(actual=self.splits[label]['y_dev'],
        #                                       predicted=[majority_class] * n)
        # baseline_majority = util.add_prefix_to_dict_keys(baseline_majority, 'baseline_majority_')
        # out = util.merge_dicts(out, baseline_majority)

        # unsupervised
        iforest = IsolationForest(contamination=prevalence[True], behaviour='new',
                                  n_estimators=self.clf_n_estimator, random_state=self.random_seed)
        iforest.fit(self.splits[label]['x_train'])
        baseline_unsupervised = self._get_metrics(actual=self.splits[label]['y_dev'],
                                                  predicted=iforest.predict(self.splits[label]['x_dev']) == -1)
        baseline_unsupervised = util.add_prefix_to_dict_keys(baseline_unsupervised, 'baseline_unsupervised_')
        out = util.merge_dicts(out, baseline_unsupervised)

        return out

    def _active_learning_data_split(self, label: str) -> ActiveLearningData:
        x_train: DataFrame = self.splits[label]['x_train']
        y_train: Series = self.splits[label]['y_train']

        indices, rest = train_test_split(range(len(x_train)),
                                         test_size=1 - self.active_learning_n_initial / len(x_train),
                                         random_state=self.random_seed,
                                         stratify=y_train)
        assert len(indices) == self.active_learning_n_initial
        x_train_start: DataFrame = x_train.iloc[indices].reset_index(drop=True)
        y_train_start: Series = y_train.iloc[indices].reset_index(drop=True)
        assert y_train_start.nunique() == 2, f"The split for label {label} resulted in one classes"
        x_train_pool: DataFrame = x_train.iloc[rest].reset_index(drop=True)
        y_train_pool: Series = y_train.iloc[rest].reset_index(drop=True)
        y_dev: Series = self.splits[label]['y_dev']
        x_dev: np.ndarray = self.splits[label]['x_dev'].values
        return ActiveLearningData(x_train_start, y_train_start, x_train_pool, y_train_pool, x_dev, y_dev)

    @staticmethod
    def _check_directory_exists(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    @staticmethod
    def _pickle(obj: object, path: str) -> None:
        pickle.dump(obj, open(path, 'wb'))

    @staticmethod
    def _write_as_csv(df: DataFrame, path: str) -> None:
        df.to_csv(path, index=False)

    @staticmethod
    def _get_plotting_row(i: int,
                          metrics: Stats, elapsed_train: float, elapsed_query: float) -> Stats:
        return {'i': i + 1, 'f1': metrics['f1'], 'train time (s)': elapsed_train, 'query time (s)': elapsed_query}

    def _active_learning_update_metrics(self, active_learner: ActiveLearner, x_dev: np.ndarray, y_dev: Series,
                                        stats: Stats, data_for_plotting: List[Stats], i: int,
                                        elapsed_train: float,
                                        elapsed_query: float,
                                        labeled_indices: List[int],
                                        semi_sup: bool
                                        ) -> Tuple[Stats, List[Stats], List[int]]:
        predicted = active_learner.predict(x_dev)
        scores = None if semi_sup else active_learner.predict_proba(x_dev)[:, 1]
        metrics = self._get_metrics(actual=y_dev, predicted=predicted, scores=scores)

        data_for_plotting.append(self._get_plotting_row(i, metrics, elapsed_train, elapsed_query))
        metrics = util.add_prefix_to_dict_keys(metrics, f'sample_{i+1}_')
        if i + 1 in self.active_learning_log_intervals or i == -1:
            stats = util.merge_dicts(stats, metrics)
        return stats, data_for_plotting, labeled_indices

    @staticmethod
    def _get_active_learning_instance(x: DataFrame, y: Series, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return x.iloc[index, :].values, y.iloc[index].values

    @staticmethod
    def _get_label_propagation_max_entropy_index(lp: BaseLabelPropagation) -> int:
        entropies = entropy(lp.label_distributions_.T)
        return np.argsort(entropies)[-1]

    @staticmethod
    def _construct_semi_supervised_data(x_start: DataFrame, y_start: Series, x_pool: DataFrame, y_pool: Series,
                                        labeled_indices: List[int]) -> Tuple[DataFrame, Series]:
        y_pool_labeled = pd.Series([-1] * len(y_pool))
        y_pool_labeled[labeled_indices] = y_pool[labeled_indices]
        x_out = pd.concat([x_start, x_pool]).reset_index(drop=True)
        y_out = pd.concat([y_start, y_pool_labeled]).reset_index(drop=True)
        return x_out, y_out.astype(np.int)

    @staticmethod
    def _get_random_index(indices: List[int]) -> int:
        return np.random.choice(indices)

    def _get_output_path(self, label: str, learner: ActiveSemiSup, sampling_strategy: Callable):
        learner_name = learner.__class__.__name__
        sampling_strategy_name = sampling_strategy.__name__
        label_cleaned = label.replace('.', '')
        file_path = f'{label_cleaned}_{learner_name}_{sampling_strategy_name}'
        file_path_pkl = f'{self.results_folder}/{file_path}.pkl'
        file_path_csv = f'{self.results_folder}/{file_path}.csv'
        self._check_directory_exists(file_path_pkl)
        print(f'Label: {label}, learner: {learner_name}, sampling strategy: {sampling_strategy_name}')
        return file_path_pkl, file_path_csv, learner_name, sampling_strategy_name

    def _active_learning_initial_training(self, semi_sup: bool, out, data_for_plotting: List[Stats],
                                          learner: Optional[BaseEstimator], sampling_strategy: Callable,
                                          active_learning_data: ActiveLearningData,
                                          labeled_indices: List[int]) -> Tuple[ActiveSemiSup, Stats, List[Stats]]:
        if semi_sup:
            clf = self.semi_supervised_class(**self.semi_supervised_class_args)
            x, y = self._construct_semi_supervised_data(active_learning_data.x_train_start,
                                                        active_learning_data.y_train_start,
                                                        active_learning_data.x_train_pool,
                                                        active_learning_data.y_train_pool, labeled_indices)
            clf, elapsed_train = util.timer(clf.fit, **{'X': x, 'y': y})
        else:
            clf, elapsed_train = util.timer(ActiveLearner, **dict(estimator=learner,
                                                                  query_strategy=sampling_strategy,
                                                                  X_training=active_learning_data.x_train_start.values,
                                                                  y_training=active_learning_data.y_train_start.values))
        predicted, elapsed_query = util.timer(clf.predict, **{'X': active_learning_data.x_dev})
        predicted = clf.predict(active_learning_data.x_dev)
        # [:, 1] to get positive class probabilities, semi-sup probabilities can be NaN so skip
        scores = None if semi_sup else clf.predict_proba(active_learning_data.x_dev)[:, 1]
        metrics = self._get_metrics(actual=active_learning_data.y_dev, predicted=predicted, scores=scores)
        data_for_plotting.append(self._get_plotting_row(-1, metrics, elapsed_train, elapsed_query))
        metrics = util.add_prefix_to_dict_keys(metrics, 'initial_')
        out = util.merge_dicts(out, {'train time (s)': elapsed_train, 'query time (s)': elapsed_query})
        out = util.merge_dicts(out, metrics)
        return clf, out, data_for_plotting

    @staticmethod
    def _initialize_stats(label: str, learner_name: str, sampling_strategy_name: str) -> Stats:
        return {'label': label, 'learner': learner_name, 'sampling strategy': sampling_strategy_name}

    def _active_learning_single_query_semi_sup(self, clf: ActiveSemiSup, labeled_indices: List[int],
                                               active_learning_data: ActiveLearningData,
                                               sampling_strategy: Callable) -> Tuple[ActiveSemiSup, float, float]:
        # semi-supervised find either random unlabeled index or max entropy given the value of sampling strategy
        # then instantiate an entirely new classifier and train
        labeled_indices_set = set(labeled_indices)  # for faster search
        unlabeled_indices = [i for i in range(len(active_learning_data.y_train_pool)) if i not in labeled_indices_set]
        assert len(unlabeled_indices) > 0, "We're out of unlabeled instances, should not happen!"
        start = time()
        instance_index = (self._get_random_index(unlabeled_indices) if sampling_strategy == random_sampling else
                          self._get_label_propagation_max_entropy_index(clf))
        elapsed_query = time() - start
        labeled_indices += [instance_index]
        x, y = self._construct_semi_supervised_data(active_learning_data.x_train_start,
                                                    active_learning_data.y_train_start,
                                                    active_learning_data.x_train_pool,
                                                    active_learning_data.y_train_pool,
                                                    labeled_indices)
        clf = self.semi_supervised_class(**self.semi_supervised_class_args)
        # train
        start = time()
        clf.fit(x, y)
        elapsed_train = time() - start
        return clf, elapsed_train, elapsed_query

    def _active_learning_single_query_supervised(self, clf: ActiveSemiSup,
                                                 active_learning_data: ActiveLearningData
                                                 ) -> Tuple[ActiveSemiSup, float, float]:
        start = time()
        instance_index, _ = clf.query(active_learning_data.x_train_pool.values)
        elapsed_query = time() - start
        x_instance, y_instance = self._get_active_learning_instance(active_learning_data.x_train_pool,
                                                                    active_learning_data.y_train_pool,
                                                                    instance_index)
        start = time()
        clf.teach(x_instance, y_instance)
        elapsed_train = time() - start
        return clf, elapsed_train, elapsed_query

    def _active_learning_single_query(self, i: int, semi_sup: bool, clf: ActiveSemiSup, sampling_strategy: Callable,
                                      active_learning_data: ActiveLearningData, stats: Stats,
                                      data_for_plotting: List[Stats], labeled_indices: List[int]):
        if i % self.active_learning_print_every == 0:
            print(f'Query # {i + 1} to the analyst')

        if semi_sup:
            clf, elapsed_train, elapsed_query = self._active_learning_single_query_semi_sup(clf, labeled_indices,
                                                                                            active_learning_data,
                                                                                            sampling_strategy)
        else:
            # active learning query and teach
            clf, elapsed_train, elapsed_query = self._active_learning_single_query_supervised(clf, active_learning_data)

        return self._active_learning_update_metrics(clf, active_learning_data.x_dev, active_learning_data.y_dev, stats,
                                                    data_for_plotting, i,
                                                    elapsed_train, elapsed_query, labeled_indices, semi_sup)

    def _active_learning_for_learner_strategy(self, label: str, learner: Optional[BaseEstimator],
                                              sampling_strategy: Callable, active_learning_data: ActiveLearningData,
                                              semi_sup: bool = False) -> Stats:
        data_for_plotting = []
        file_path_pkl, file_path_csv, learner_name, sampling_strategy_name = self._get_output_path(label, learner,
                                                                                                   sampling_strategy)

        # used for label propagation
        labeled_indices = []

        if os.path.exists(file_path_pkl):
            print('Already exists in cache, returning...')
            return util.unpickle(file_path_pkl)

        # initialize stats
        stats = self._initialize_stats(label, learner_name, sampling_strategy_name)

        # initial training
        clf, stats, data_for_plotting = self._active_learning_initial_training(semi_sup,
                                                                               stats, data_for_plotting, learner,
                                                                               sampling_strategy, active_learning_data,
                                                                               labeled_indices)

        # actively learn one analyst query at a time
        for i in range(self.active_learning_budget):
            stats, data_for_plotting, labeled_indices = self._active_learning_single_query(i, semi_sup, clf,
                                                                                           sampling_strategy,
                                                                                           active_learning_data,
                                                                                           stats, data_for_plotting,
                                                                                           labeled_indices)

        # persist the results
        util.pickle_object(stats, file_path_pkl)
        util.write_as_csv(pd.DataFrame(data_for_plotting), file_path_csv)

        return stats

    def _active_learning(self, label: str) -> List[Stats]:
        return [
            self._active_learning_for_learner_strategy(label, learner, sampling_strategy,
                                                       self._active_learning_data_split(label), semi_sup=False)
            for learner in self.active_learning_learners
            for sampling_strategy in self.active_learning_strategies]

    def _semi_supervised(self, label: str) -> List[Stats]:
        return [
            self._active_learning_for_learner_strategy(label, LabelSpreading, sampling_strategy,
                                                       self._active_learning_data_split(label), semi_sup=True)
            for sampling_strategy in self.active_learning_strategies]

    def _ensemble_predictions(self, rf: ActiveLearner, lr: ActiveLearner, iforest: IsolationForest,
                              lp: LabelSpreading, active_learning_date: ActiveLearningData):
        x_dev = active_learning_date.x_dev
        threshold = sum(self.ensemble_weights.values()) / 2

        return np.vstack([
            rf.predict(x_dev) * self.ensemble_weights['rf'],
            lr.predict(x_dev) * self.ensemble_weights['lr'],
            (iforest.predict(x_dev) == -1) * self.ensemble_weights['iforest'],
            lp.predict(x_dev) * self.ensemble_weights['lp']
        ]).sum(axis=0) >= threshold

    def _generate_report(self, f: Callable):
        return pd.concat(pd.DataFrame(f(label)) for label in self.features)

    def report_active_learning(self) -> DataFrame:
        return self._generate_report(self._active_learning)

    def report_semi_supervised(self) -> DataFrame:
        return self._generate_report(self._semi_supervised)

    def _ensemble(self, label: str) -> Stats:
        active_learning_data = self._active_learning_data_split(label)
        stats = self._initialize_stats(label, 'ensemble', 'entropy_sampling')

        # supervised
        # active learners
        rf, _, _ = self._active_learning_initial_training(False, stats, [], self.active_learning_rf,
                                                          entropy_sampling, active_learning_data, [])
        lr, _, _ = self._active_learning_initial_training(False, stats, [], self.active_learning_lr,
                                                          entropy_sampling, active_learning_data, [])

        # semi-supervised: label propagation
        labeled_indices = []
        lp, _, _ = self._active_learning_initial_training(True, stats, [], None, entropy_sampling,
                                                          active_learning_data, labeled_indices)

        # unsupervised
        prevalence = len(active_learning_data.y_train_start[active_learning_data.y_train_start == True]) / len(
            active_learning_data.y_train_start)
        iforest = IsolationForest(contamination=prevalence, behaviour='new', n_estimators=self.clf_n_estimator)
        x = pd.concat([active_learning_data.x_train_pool, active_learning_data.x_train_pool]).reset_index(drop=True)
        iforest.fit(x)

        predictions = self._ensemble_predictions(rf, lr, iforest, lp, active_learning_data)
        metrics = self._get_metrics(active_learning_data.y_dev, predictions)
        data_for_plotting = [self._get_plotting_row(-1, metrics, 0, 0)]
        metrics = util.add_prefix_to_dict_keys(metrics, 'initial_')
        stats = util.merge_dicts(stats, metrics)

        for i in range(self.active_learning_budget):
            rf, _, _ = self._active_learning_single_query_supervised(rf, active_learning_data)
            lr, _, _ = self._active_learning_single_query_supervised(lr, active_learning_data)
            lp, _, _ = self._active_learning_single_query_semi_sup(lp, labeled_indices, active_learning_data,
                                                                   entropy_sampling)

            predictions = self._ensemble_predictions(rf, lr, iforest, lp, active_learning_data)
            metrics = self._get_metrics(active_learning_data.y_dev, predictions)

            data_for_plotting.append(self._get_plotting_row(i, metrics, 0, 0))
            if i + 1 in self.active_learning_log_intervals:
                metrics = util.add_prefix_to_dict_keys(metrics, f'sample_{i+1}_')
                stats = util.merge_dicts(stats, metrics)

        return stats

    def report_ensemble(self) -> DataFrame:
        return self._generate_report(self._ensemble)

    def report_baseline_oracle(self):
        return pd.DataFrame([self._calculate_baseline_oracle(label) for label in self.features]).set_index('label')

    def report_labels(self):
        return pd.DataFrame(
            [
                {
                    'label': label,
                    'records': len(self.dfs_by_label[label]),
                    'attacks': len(self.dfs_by_label[label][self.dfs_by_label[label][self.label_col] == label]),
                    'prevalence': len(
                        self.dfs_by_label[label][self.dfs_by_label[label][self.label_col] == label]) / len(
                        self.dfs_by_label[label]),
                    'prevalence (overall)': len(
                        self.dfs_by_label[label][self.dfs_by_label[label][self.label_col] == label]) / len(self.df)
                }
                for label in self.dfs_by_label
            ]
        ).set_index('label')

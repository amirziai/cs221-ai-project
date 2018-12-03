import os
import pickle
from time import time
from typing import Dict, Iterable, Union, Optional, List

import numpy as np
import pandas as pd
import requests
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split

import config
import util

Pandas = Union[DataFrame, Series]


def random_sampling(_, x_pool):
    n_samples = len(x_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], _


class KDD1999:
    def __init__(self):
        self.label_col = config.label_col
        self.label_normal = config.label_normal

        # config
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
        self.active_learning_log_at = config.active_learning_log_at

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
    def _get_metrics(actual: Iterable,
                     predicted: Iterable,
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
        iforest = IsolationForest(contamination=prevalence[True], behaviour='new', n_estimators=self.clf_n_estimator)
        iforest.fit(self.splits[label]['x_train'])
        baseline_unsupervised = self._get_metrics(actual=self.splits[label]['y_dev'],
                                                  predicted=iforest.predict(self.splits[label]['x_dev']) == -1)
        baseline_unsupervised = util.add_prefix_to_dict_keys(baseline_unsupervised, 'baseline_unsupervised_')
        out = util.merge_dicts(out, baseline_unsupervised)

        return out

    def _active_learning(self, label: str) -> List[Dict[str, Union[float, int]]]:
        x_train = self.splits[label]['x_train']
        y_train = self.splits[label]['y_train']

        indices, rest = train_test_split(range(len(x_train)),
                                         test_size=1 - self.active_learning_n_initial / len(x_train),
                                         random_state=self.random_seed,
                                         stratify=y_train)
        assert len(indices) == self.active_learning_n_initial
        x_train_start = x_train.iloc[indices].reset_index(drop=True)
        y_train_start = y_train.iloc[indices].reset_index(drop=True)
        assert y_train_start.nunique() == 2
        x_train_pool = x_train.iloc[rest].reset_index(drop=True)
        y_train_pool = y_train.iloc[rest].reset_index(drop=True)
        y_dev = self.splits[label]['y_dev']
        x_dev = self.splits[label]['x_dev'].values

        learners = [RandomForestClassifier(n_estimators=self.clf_n_estimator, n_jobs=-1),
                    LogisticRegression(solver='lbfgs')]
        sampling_strategies = [uncertainty_sampling, random_sampling, entropy_sampling]

        outs = []

        for learner in learners:
            for sampling_strategy in sampling_strategies:
                data_for_plotting = []
                learner_name = learner.__class__.__name__
                sampling_strategy_name = sampling_strategy.__name__
                label_cleaned = label.replace('.', '')
                file_path = f'{label_cleaned}_{learner_name}_{sampling_strategy_name}'
                file_path_pkl = f'{file_path}.pkl'
                file_path_csv = f'{file_path}.csv'
                print(f'Label: {label}, learner: {learner_name}, sampling strategy: {sampling_strategy_name}')

                if os.path.exists(file_path_pkl):
                    print('Loaded from cache')
                    out = pickle.load(open(file_path_pkl, 'rb'))
                else:
                    out = {'label': label, 'learner': learner_name, 'sampling strategy': sampling_strategy_name}
                    start = time()
                    active_learner = ActiveLearner(
                        estimator=learner,
                        query_strategy=sampling_strategy,
                        X_training=x_train_start.values, y_training=y_train_start.values,
                    )
                    elapsed = time() - start
                    out = util.merge_dicts(out, {'train time (s)': elapsed})
                    predicted = active_learner.predict(x_dev)
                    scores = active_learner.predict_proba(x_dev)[:, 1]
                    metrics = self._get_metrics(actual=y_dev, predicted=predicted, scores=scores)

                    # plotting data
                    data_for_plotting.append({'i': 0, 'f1': metrics['f1'], 'train time (s)': elapsed, 'query time (s)': 0})
                    metrics = util.add_prefix_to_dict_keys(metrics, 'initial_')
                    out = util.merge_dicts(out, metrics)

                    for i in range(self.active_learning_budget):
                        if i % 25 == 0:
                            print(f'round {i + 1}')
                        start = time()
                        idx, _ = active_learner.query(x_train_pool.values)
                        elapsed_query = time() - start
                        start = time()
                        active_learner.teach(x_train_pool.iloc[idx, :].values, y_train_pool.iloc[idx].values)
                        elapsed_train = time() - start
                        predicted = active_learner.predict(x_dev)
                        scores = active_learner.predict_proba(x_dev)[:, 1]
                        metrics = self._get_metrics(actual=y_dev, predicted=predicted, scores=scores)
                        data_for_plotting.append({'i': i + 1,
                                                  'f1': metrics['f1'],
                                                  'fp': metrics['FP'],
                                                  'fn': metrics['FN'],
                                                  'train time (s)': elapsed_train,
                                                  'query time (s)': elapsed_query,
                                                  })
                        metrics = util.add_prefix_to_dict_keys(metrics, f'sample_{i+1}_')
                        if i + 1 in {1, 10, 25, 50, 100}:
                            out = util.merge_dicts(out, metrics)

                    pickle.dump(out, open(file_path_pkl, 'wb'))
                    pd.DataFrame(data_for_plotting).to_csv(file_path_csv, index=False)

                outs.append(out)

        return outs

    def report_active_learning(self) -> DataFrame:
        return pd.concat(
            pd.DataFrame(self._active_learning(label))
            for label in self.features
        )

    def report_baseline_oracle(self):
        return pd.DataFrame([
            self._calculate_baseline_oracle(label)
            for label in self.features
        ]).set_index('label')

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

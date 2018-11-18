from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score


class Pipeline:
    def __init__(self) -> None:
        self.df_raw = None
        self.df = None
        self.x = None
        self.y = None
        self.label_header = None
        self.label_positive = None
        self.labels = None

        self.random_runs = 100
        self.metrics = {'f1': f1_score,
                        'ap': average_precision_score,
                        'precision': precision_score,
                        'recall': recall_score
                        }

    @staticmethod
    def get_xy(df: DataFrame, label_header: str, label_positive: str) -> Tuple[DataFrame, Series]:
        x = df[df.columns.difference([label_header])]
        y = df[label_header] == label_positive
        categorical_cols = [col for col in x.columns if df.dtypes[col] == object]
        x_ohe = pd.get_dummies(x, columns=categorical_cols)
        return x_ohe, y

    @staticmethod
    def get_subset(df: DataFrame, label_header: str, labels: Sequence[str]):
        df_subset = df[df[label_header].isin(labels)]
        return df_subset.reset_index(drop=True)

    def report_training_data(self):
        print('shape')
        print(self.df.shape)
        print('=' * 10)
        print('Labels')
        print(pd.DataFrame({'counts': self.df.label.value_counts(),
                            'pct': self.df.label.value_counts(normalize=True)}))
        print('=' * 10)

    def baseline(self):
        prevalence = self.y.value_counts(normalize=True)
        contamination = min(prevalence)
        isolation_forest = IsolationForest(contamination=contamination, behaviour='new')
        isolation_forest.fit(self.x)
        predicted = isolation_forest.predict(self.x) == 1
        assert abs(min(pd.Series(predicted).value_counts()) - min(self.y.value_counts())) <= 10
        matrix = pd.DataFrame({'predicted': predicted, 'actual': self.y}).groupby(['predicted', 'actual']).size()
        print('Confusion matrix')
        print(pd.DataFrame({'counts': matrix, 'pct': matrix / matrix.sum()}))
        print('=' * 10)
        print('Scores')
        scores = pd.DataFrame({
            metric: {'random': pd.Series([
                f_metric(self.y, np.random.choice([1, 0], p=prevalence, size=len(self.y)) == 1)
                for _ in range(self.random_runs)
            ]).median(),
                     'baseline': f_metric(self.y, predicted)}
            for metric, f_metric in self.metrics.items()
        }).transpose()
        scores['relative'] = scores.baseline / scores.random - 1
        print(scores)


class KDD1999(Pipeline):
    def __init__(self) -> None:
        super().__init__()
        # hardcoded values
        headers_link = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'
        path = 'resources/datasets/kddcup.data_10_percent'
        label_header = 'label'
        label_positive = 'normal.'
        labels = ['normal.', 'ipsweep.']

        contents = requests.get(headers_link).text
        headers = [x.split(':')[0] for x in contents.split('\n')[1:]]
        headers[-1] = label_header

        self.label_header = label_header
        self.label_positive = label_positive
        self.labels = labels
        self.df_raw = pd.read_csv(path, header=None, names=headers)
        self.df = self.get_subset(self.df_raw, self.label_header, self.labels)
        self.x, self.y = self.get_xy(self.df, self.label_header, self.label_positive)

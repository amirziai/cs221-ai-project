import unittest

import pandas as pd

import pipeline


class UnitTests(unittest.TestCase):
    def test_kdd(self):
        kdd = pipeline.KDD1999()
        expected = pd.Series({'normal.': 97278, 'ipsweep.': 1247})
        self.assertTrue(expected.equals(kdd.df.label.value_counts()))
        self.assertEqual((98525, 42), kdd.df.shape)
        self.assertEqual((494021, 42), kdd.df_raw.shape)
        self.assertEqual((98525, 82), kdd.x.shape)
        self.assertEqual((98525,), kdd.y.shape)
        self.assertEqual(['normal.', 'ipsweep.'], kdd.labels)


if __name__ == '__main__':
    unittest.main()

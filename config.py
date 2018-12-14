verbose = False

# data
headers_link = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'
file_path = 'resources/datasets/kddcup.data_10_percent'
label_col = 'label'
label_normal = 'normal.'
label_threshold = 100  # at least this many of the label to be included

# eval
baseline_random_n = 10
clf_n_estimator = 10

# pipeline
random_seed = 0
fill_na = 0
size_train = 0.8
size_dev = 0.1

# active learning
active_learning_n_initial = 1000
active_learning_budget = 100
active_learning_log_at = [0.01, 0.1, 0.25, 0.5, 1]
active_learning_log_intervals = {1, 10, 25, 50, 100}

# cosmetic
round_to = 2

# sanity checks
assert 0.5 < size_train + size_dev < 1, "check data split %"

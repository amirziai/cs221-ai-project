import pickle as pickle
from functools import reduce
from time import time
from typing import Dict, Any, Tuple, Callable

from pandas.core.frame import DataFrame


def merge_dicts(*args: Dict) -> Dict[Any, Any]:
    return reduce(lambda x, y: {**x, **y}, args)


def add_prefix_to_dict_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f'{prefix}{key}': d[key] for key in d}


def timer(f: Callable, **kwargs: Any) -> Tuple[Any, float]:
    start = time()
    out = f(**kwargs)
    return out, time() - start


def pickle_object(obj: object, path: str) -> None:
    pickle.dump(obj, open(path, 'wb'))


def unpickle(path: str) -> Any:
    return pickle.load(open(path, 'rb'))


def write_as_csv(df: DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

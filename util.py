from functools import reduce
from typing import Dict, Any


def merge_dicts(*args: Any) -> Dict[Any, Any]:
    return reduce(lambda x, y: {**x, **y}, args)


def add_prefix_to_dict_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f'{prefix}{key}': d[key] for key in d}

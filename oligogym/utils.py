import re
from collections import Counter


def merge_dicts(*dict_args):
    # Use the Counter class to easily sum values of matching keys
    result = Counter()
    for dictionary in dict_args:
        result += Counter(dictionary)
    return dict(result)


def count_overlapping(string, pattern):
    regex = "{}(?={})".format(re.escape(pattern[:1]), re.escape(pattern[1:]))
    # Consume iterator, get count with minimal memory usage
    return sum(1 for _ in re.finditer(regex, string))

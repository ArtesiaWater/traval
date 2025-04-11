# ruff: noqa: D100 D103
import numpy as np
import pandas as pd

import traval
from traval.ruleset import RuleSet


def func1(s):
    mask = s > 10
    s = pd.Series(index=s.index, data=0.0)
    s.loc[mask] = np.nan
    return s


def func2(s, val):
    mask = s < val
    s = pd.Series(index=s.index, data=0.0)
    s.loc[mask] = np.nan
    return s


def func3(name):
    d = {"test_series": 2}
    return d[name]


def func4(*args):
    for i, series in enumerate(args):
        if i == 0:
            result = series
        else:
            result = result.add(series)
    return result


def get_empty_ruleset():
    return RuleSet(name="test")


def get_filled_ruleset():
    rset = traval.RuleSet(name="test")
    rset.add_rule("gt10", func1, apply_to=0)
    rset.add_rule("less_than_value", func2, apply_to=1, kwargs={"val": 0})
    return rset


def test_init():
    _ = traval.RuleSet(name="test")


def test_add_rules():
    rset = traval.RuleSet(name="test")
    rset.add_rule("gt10", func1, apply_to=0)
    rset.add_rule("less_than_value", func2, apply_to=1, kwargs={"val": 0})


def test_update_rules():
    rset = traval.RuleSet(name="test")
    rset.add_rule("gt10", func1, apply_to=0)
    rset.add_rule("less_than_value", func2, apply_to=1, kwargs={"val": 0})
    rset.update_rule("less_than_value", func2, apply_to=1, kwargs={"val": func3})


def test_to_dataframe():
    rset = get_filled_ruleset()
    _ = rset.to_dataframe()


def test_applyself_static_kwargs():
    series = pd.Series(index=range(10), data=range(-5, 23, 3), name="test_series")
    rset = get_filled_ruleset()
    _, _ = rset(series)


def test_applyself_callable_kwargs():
    series = pd.Series(index=range(10), data=range(-5, 23, 3), name="test_series")
    rset = get_filled_ruleset()
    rset.update_rule("less_than_value", func2, apply_to=1, kwargs={"val": func3})
    _, _ = rset(series)


def test_applyself_combine():
    rset = traval.RuleSet(name="test")
    rset.add_rule("+1", lambda s: s + 1, apply_to=0)
    rset.add_rule("add 0+1", func4, apply_to=(0, 1))
    series = pd.Series(index=range(10), data=0.0, name="test_series")
    d, _ = rset(series)
    assert (d[len(d) - 1] == 1.0).all()


def test_del_rules():
    rset = get_filled_ruleset()
    rset.del_rule("gt10")
    assert len(rset.rules) == 1


def test_to_from_pickle():
    rset = get_filled_ruleset()
    rset.to_pickle("test.pkl")
    rset = RuleSet.from_pickle("test.pkl")
    import os

    os.remove("test.pkl")


def test_to_from_json():
    rset = get_filled_ruleset()
    rset.to_json("test.json")
    rset = RuleSet.from_json("test.json")
    import os

    os.remove("test.json")

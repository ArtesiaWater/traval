import os

import numpy as np
from traval import RuleSet, TravalParameters, rulelib


def get_ruleset1():
    rset = RuleSet("tester1")
    rset.add_rule(
        "gt10",
        rulelib.rule_ufunc_threshold,
        apply_to=0,
        kwargs={"ufunc": (np.greater,), "threshold": 10.0},
    )
    rset.add_rule(
        "lt0",
        rulelib.rule_ufunc_threshold,
        apply_to=0,
        kwargs={"ufunc": (np.less,), "threshold": 0.0},
    )
    return rset


def func(series, a, b, c, d):
    assert isinstance(a, str)
    assert isinstance(b, int)
    assert isinstance(c, float)
    assert callable(d)
    return series


def some_callable(*args):
    return 2


def get_ruleset2():
    rset = RuleSet("tester2")
    rset.add_rule(
        "many_kwargs",
        func,
        apply_to=0,
        kwargs={
            "a": "some_string",
            "b": 1,
            "c": 1.5,
            "d": some_callable,
        },
    )
    return rset


def test_tp_from_ruleset():
    rset = get_ruleset1()
    tp = TravalParameters.from_ruleset(rset)
    return tp


def test_tp_from_ruleset_w_locations():
    rset = get_ruleset1()
    tp = TravalParameters.from_ruleset(rset, locations=["loc1"])
    return tp


def test_tp_get_parameters_defaults():
    tp = test_tp_from_ruleset()
    _ = tp.get_parameters()  # return all defaults
    _ = tp.get_parameters(rulename="gt10")  # return all params for rule
    p3 = tp.get_parameters(rulename="gt10", parameter="threshold")  # value
    assert isinstance(p3, float)
    try:
        tp.get_parameters(location="non-existent-loc")
    except ValueError:
        pass

    try:
        tp.get_parameters(rulename="gt10", parameter="non-existent-param")
    except KeyError:
        pass
    return


def test_tp_get_parameters_location_specific():
    tp = test_tp_from_ruleset_w_locations()
    _ = tp.get_parameters()  # return all defaults
    _ = tp.get_parameters(location="loc1")  # return all for location
    # return loc params for rule
    _ = tp.get_parameters(location="loc1", rulename="gt10")
    p4 = tp.get_parameters(
        location="loc1", rulename="gt10", parameter="threshold"
    )  # value
    assert isinstance(p4, float)
    try:
        tp.get_parameters(location="non-existent-loc")
    except KeyError:
        pass

    try:
        tp.get_parameters(
            location="loc1", rulename="gt10", parameter="non-existent-param"
        )
    except KeyError:
        pass
    return


def test_tp_to_from_csv():
    rset = get_ruleset2()
    tp = TravalParameters.from_ruleset(rset, locations=["loc1"])
    tp.to_csv("test.csv")
    tp2 = TravalParameters.from_csv("test.csv")
    os.remove("test.csv")
    mask = tp.defaults["value"].apply(lambda s: tp._test_callable(s))
    assert (tp.defaults.loc[~mask].index == tp2.defaults.index).all()
    assert (tp.defaults.loc[~mask, "value"] == tp2.defaults.loc[~mask, "value"]).all()
    return


def test_tp_to_from_json():
    rset = get_ruleset2()
    tp = TravalParameters.from_ruleset(rset, locations=["loc1"])
    tp.to_json("test.json")
    tp2 = TravalParameters.from_json("test.json")
    os.remove("test.json")
    mask = tp.defaults["value"].apply(lambda s: tp._test_callable(s))
    assert (tp.defaults.loc[~mask].index == tp2.defaults.index).all()
    assert (tp.defaults.loc[~mask, "value"] == tp2.defaults.loc[~mask, "value"]).all()
    return


def test_tp_to_from_pickle():
    rset = get_ruleset2()
    tp = TravalParameters.from_ruleset(rset, locations=["loc1"])
    tp.to_pickle("test.pkl")
    tp2 = TravalParameters.from_pickle("test.pkl")
    os.remove("test.pkl")
    assert (tp.defaults.index == tp2.defaults.index).all()
    assert (tp.defaults["value"] == tp2.defaults["value"]).all()
    return

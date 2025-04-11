# ruff: noqa: D100 D103
import numpy as np
import pandas as pd
from test_002_ruleset import get_filled_ruleset

import traval


def get_detector():
    s = pd.Series(
        index=range(10), data=np.arange(-5, 23, 3, dtype=float), name="test_series"
    )
    return traval.Detector(s)


def get_detector_with_result():
    d = get_detector()
    rset = get_filled_ruleset()
    t = pd.Series(
        index=range(10), data=np.arange(-5, 23, 3, dtype=float), name="test_series"
    )
    t[t < 0] = np.nan
    t[t > 10] = np.nan
    d = get_detector()
    d.set_truth(t)
    d.apply_ruleset(rset)
    return d


def test_init_detector():
    s = pd.Series(
        index=range(10), data=np.arange(-5, 23, 3, dtype=float), name="test_series"
    )
    traval.Detector(s)


def test_repr():
    d = get_detector()
    d.__repr__()


def test_add_truth():
    t = pd.Series(
        index=range(10), data=np.arange(-5, 23, 3, dtype=float), name="test_series"
    )
    t[t < 0] = np.nan
    t[t > 10] = np.nan
    d = get_detector()
    d.set_truth(t)


def test_apply_ruleset():
    rset = get_filled_ruleset()
    t = pd.Series(
        index=range(10), data=np.arange(-5, 23, 3, dtype=float), name="test_series"
    )
    t[t < 0] = np.nan
    t[t > 10] = np.nan
    d = get_detector()
    d.set_truth(t)
    d.apply_ruleset(rset)


def test_reset():
    d = get_detector_with_result()
    d.reset()
    assert not hasattr(d, "ts_result")


def test_confusion_matrix():
    d = get_detector_with_result()
    _ = d.confusion_matrix()


def test_uniqueness():
    d = get_detector_with_result()
    _ = d.uniqueness()


def test_plot_overview():
    d = get_detector_with_result()
    _ = d.plot_overview()


def test_get_series():
    d = get_detector_with_result()
    _ = d.get_series(2, category="tp")


def test_get_corrections():
    d = get_detector_with_result()
    _ = d.get_corrections_dataframe()


def test_get_final_result():
    d = get_detector_with_result()
    _ = d.get_final_result()


def test_get_comment_series():
    d = get_detector_with_result()
    comments = d.get_comment_series()
    assert (comments == "gt10").sum() == 4
    assert (comments == "less_than_value").sum() == 2


def test_empty_comment_series():
    rset = traval.RuleSet(name="test_empty")
    rset.add_rule("no_op", lambda x: pd.Series(), apply_to=0)
    d = get_detector()
    d.apply_ruleset(rset)
    comments = d.get_comment_series()
    assert comments.empty

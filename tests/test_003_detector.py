import traval
import pandas as pd
import numpy as np

from test_002_ruleset import test_add_rules


def test_init_detector():
    s = pd.Series(index=range(10),
                  data=np.arange(-5, 23, 3, dtype=float),
                  name="test_series")
    d = traval.Detector(s)
    return d


def test_repr():
    d = test_init_detector()
    return d.__repr__


def test_add_truth():
    t = pd.Series(index=range(10),
                  data=np.arange(-5, 23, 3, dtype=float),
                  name="test_series")
    t[t < 0] = np.nan
    t[t > 10] = np.nan
    d = test_init_detector()
    d.set_truth(t)
    return d


def test_apply_ruleset():
    rset = test_add_rules()
    d = test_add_truth()
    d.apply_ruleset(rset)
    return d


def test_reset():
    d = test_apply_ruleset()
    d.reset()
    assert not hasattr(d, "ts_result")
    return

import numpy as np
import pandas as pd
import pytest
from traval import rulelib as rlib


def test_rule_ufunc_threshold_float():
    # rule_ufunc_threshold: float
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    c1 = rlib.rule_ufunc_threshold(s1, (np.greater_equal,), 5)
    assert c1.iloc[5:].isna().sum() == 5
    return c1


def test_rule_ufunc_threshold_series():
    # rule_ufunc_threshold: series
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    idx = date_range[:3].to_list() + date_range[-4:-1].to_list()
    thresh_series = pd.Series(index=idx, data=5.0)
    c2 = rlib.rule_ufunc_threshold(s1, (np.greater_equal,), thresh_series)
    assert c2.iloc[5:].isna().sum() == 5
    return c2


def test_rule_diff_ufunc_threshold():
    # rule_diff_ufunc_threshold
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s1.loc[date_range[4]] += 1
    c3 = rlib.rule_diff_ufunc_threshold(s1, (np.greater_equal,), 1.1)
    assert c3.iloc[4:5].isna().all()
    return c3


def test_rule_other_ufunc_threshold():
    # rule_other_ufunc_threshold
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    val = s1.copy()
    c4 = rlib.rule_other_ufunc_threshold(s1, val, (np.less,), 5)
    assert c4.iloc[:5].isna().sum() == 5
    return c4


def test_rule_max_gradient():
    # rule_max_gradient
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s1.loc[date_range[4]] += 1
    c5 = rlib.rule_max_gradient(s1, max_step=1.0, max_timestep="1D")
    assert c5.iloc[4:5].isna().all()
    return c5


def test_rule_spike_detection():
    # rule_spike_detection
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s1.iloc[4] += 3
    c6 = rlib.rule_spike_detection(s1, threshold=2, spike_tol=2)
    assert c6.iloc[4:5].isna().all()
    return c6


def test_offset_detection():
    # rule_offset_detection
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s1.iloc[3:7] += 10
    c7 = rlib.rule_offset_detection(s1, threshold=5, updown_diff=2.0)
    assert c7.iloc[3:7].isna().sum() == 4
    return c7


def test_rule_outside_n_sigma():
    # rule_outside_n_sigma
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    c8 = rlib.rule_outside_n_sigma(s1, n=1.0)
    assert c8.iloc[[0, 1, 8, 9]].isna().sum() == 4
    return c8


def test_rule_diff_outside_of_n_sigma():
    # rule_diff_outside_of_n_sigma
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s1.iloc[5:] += np.arange(5)
    c9 = rlib.rule_diff_outside_of_n_sigma(s1, 1.0)
    assert c9.iloc[6:].isna().sum() == 4
    return c9


def test_rule_outside_bandwidth():
    # rule_outside_bandwidth
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    lb = pd.Series(index=date_range[[0, -1]], data=[1, 2])
    ub = pd.Series(index=date_range[[0, -1]], data=[7, 8])
    c10 = rlib.rule_outside_bandwidth(s1, lb, ub)
    assert c10.iloc[[0, 1, 8, 9]].isna().sum() == 4
    return c10


def test_rule_shift_to_manual_obs():
    # rule_shift_to_manual_obs
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    h = pd.Series(index=date_range[[1, -1]], data=[2, 10])
    a = rlib.rule_shift_to_manual_obs(s1, h, max_dt="2D")
    assert (a.iloc[1:] == s1.iloc[1:] + 1).all()
    assert a.iloc[0] == s1.iloc[0]
    return a


def test_rule_combine_nan_or():
    # rule_combine_nan
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s2 = s1.copy()
    s1.iloc[0] = np.nan
    s2.iloc[-1] = np.nan
    c11 = rlib.rule_combine_nan_or(s1, s2)
    assert c11.iloc[[0, -1]].isna().sum() == 2
    return c11


def test_rule_combine_nan_and():
    # rule_combine_nan
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    s2 = s1.copy()
    s1.iloc[0:2] = np.nan
    s2.iloc[1:3] = np.nan
    c11 = rlib.rule_combine_nan_and(s1, s2)
    assert c11.isna().sum() == 2
    return c11


def test_rule_funcdict_to_nan():
    # rule_funcdict_to_nan
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    s1 = pd.Series(index=date_range, data=np.arange(10))
    fdict = {"lt_3": lambda s: s < 3.0, "gt_7": lambda s: s > 7.0}
    c12 = rlib.rule_funcdict_to_nan(s1, fdict)
    assert c12.iloc[[0, 1, 2, -2, -1]].isna().sum() == 5
    return c12


def test_rule_keep_comments():
    # rule_keep_comments
    date_range = pd.date_range("2020-01-01", freq="D", periods=10)
    raw = pd.Series(index=date_range, data=np.arange(10), dtype=float)
    comments = ["keep"] * 4 + [""] * 3 + ["discard"] * 3
    comment_series = pd.Series(index=raw.index, data=comments)
    val = raw.copy()
    val += 1.0
    val.loc[comment_series == "keep"] = np.nan
    f = rlib.rule_keep_comments(raw, ["keep"], comment_series, val)
    assert (f.loc[comment_series == "keep"] == 0).all()
    assert (f.loc[comment_series != "keep"] == 1).all()


@pytest.mark.skip
def test_rule_pastas_outside_pi():
    # rule_pastas_outside_pi
    # skip for now
    pass

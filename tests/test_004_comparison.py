import numpy as np
import pandas as pd
import traval


def test_series_comparison():
    idx1 = pd.date_range("2020-01-01", "2020-11-30", freq="D")
    idx2 = pd.date_range("2020-02-01", "2020-12-31", freq="D")

    s1 = pd.Series(index=idx1, data=1.0)
    s1.loc["2020-03-15":"2020-04-15"] = np.nan
    s2 = pd.Series(index=idx2, data=2.0)
    s2.loc["2020-04-01":"2020-04-30"] = np.nan

    sc = traval.SeriesComparison(s1, s2)
    return sc


def test_series_relative_comparison():
    testdf = pd.DataFrame(
        index=pd.date_range("2020-01-01", freq="D", periods=16),
        columns=["orig", "s1", "s2"],
        data=0.0,
    )

    # -9999 means obs is missing entirely
    missing = -9999.0

    # expected possibilities
    testdf.iloc[0] = 1.0, 1.0, 1.0  # 1   # kept_in_both
    testdf.iloc[1] = 1.0, 1.0, np.nan  # 2   # flagged_in_s2
    testdf.iloc[2] = 1.0, 1.0, missing  # 3   # flagged_in_s2
    testdf.iloc[3] = 1.0, np.nan, missing  # 4   # flagged_in_both
    testdf.iloc[4] = 1.0, np.nan, np.nan  # 5   # flagged_in_both
    testdf.iloc[5] = 1.0, missing, missing  # 6   # flagged_in_both

    # weirdness level 1
    testdf.iloc[6] = np.nan, 1.0, 1.0  # 7   # introduced_in_both
    testdf.iloc[7] = np.nan, 1.0, np.nan  # 8   # introduced_in_s1
    testdf.iloc[8] = np.nan, 1.0, missing  # 9   # introduced_in_s1
    testdf.iloc[9] = np.nan, missing, missing  # 10  # in_all_nan
    testdf.iloc[10] = np.nan, np.nan, np.nan  # 11  # in_all_nan

    # weirdness level 2
    testdf.iloc[11] = missing, 1.0, 1.0  # 12  # introduced_in_both
    testdf.iloc[12] = missing, 1.0, np.nan  # 13  # introduced_in_s1
    testdf.iloc[13] = missing, 1.0, missing  # 14  # introduced_in_s1

    # weirdness level 3 (for testing)
    testdf.iloc[14] = missing, np.nan, np.nan  # 15  # do not count...
    testdf.iloc[15] = missing, missing, missing  # 16  # do not count...

    orig = testdf.orig.loc[testdf.orig != missing]
    s1 = testdf.s1.loc[testdf.s1 != missing]
    s2 = testdf.s2.loc[testdf.s2 != missing]

    scr = traval.SeriesComparisonRelative(s1, s2, orig)

    checkresult = {
        "kept_in_both": 1,
        "flagged_in_s1": 0,
        "flagged_in_s2": 2,
        "flagged_in_both": 3,
        "in_all_nan": 2,
        "introduced_in_s1": 4,
        "introduced_in_s2": 0,
        "introduced_in_both": 2,
    }

    summary = scr.summary_base_comparison

    for k, v in checkresult.items():
        assert summary.loc[k] == v

    return scr


def test_relative_comparison_stats():
    base_idx = pd.date_range("2020-01-01", periods=110, freq="D")
    idx1 = pd.date_range("2020-01-01", periods=110, freq="D")
    idx2 = pd.date_range("2020-01-01", periods=110, freq="D")

    b = pd.Series(index=base_idx, data=1.0)  # raw data
    b.iloc[:10] = np.nan
    b.name = "raw"
    s1 = pd.Series(index=idx1, data=1.0)  # result algorithm
    s1.iloc[30:70] = np.nan
    s1.name = "result"
    s2 = pd.Series(index=idx2, data=1.0)  # truth
    s2.iloc[10:60] = np.nan
    s2.name = "truth"

    scr = traval.SeriesComparisonRelative(s1, s2, b)

    assert scr.bc.false_positive_rate + scr.bc.specificity == 1
    assert scr.bc.false_negative_rate + scr.bc.sensitivity == 1
    return scr


def test_confusion_matrix():
    cp = test_relative_comparison_stats()
    return cp.bc.confusion_matrix()

import numpy as np
import pandas as pd
from traval import SeriesComparison, SeriesComparisonRelative


def test_series_comparison_plot():
    base_idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    idx1 = pd.date_range("2020-01-01", "2020-11-30", freq="D")
    idx2 = pd.date_range("2020-02-01", "2020-12-31", freq="D")

    b = pd.Series(index=base_idx, data=1.0)
    b.iloc[:10] = np.nan

    s1 = pd.Series(index=idx1, data=1.0)
    s1.loc["2020-03-15":"2020-04-15"] = np.nan

    s2 = pd.Series(index=idx2, data=2.0)
    s2.loc["2020-04-01":"2020-04-30"] = np.nan
    s2.loc["2020-07"] = 1.0

    sc = SeriesComparison(s1, s2)

    ax = sc.plots.plot_series_comparison(
        mark_different=True, mark_identical=True, mark_unique=True
    )
    return ax


def test_relative_series_comparison_plot():
    base_idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    idx1 = pd.date_range("2020-01-01", "2020-11-30", freq="D")
    idx2 = pd.date_range("2020-02-01", "2020-12-31", freq="D")

    b = pd.Series(index=base_idx, data=1.0)
    b.iloc[:10] = np.nan

    s1 = pd.Series(index=idx1, data=1.0)
    s1.loc["2020-03-15":"2020-04-15"] = np.nan

    s2 = pd.Series(index=idx2, data=2.0)
    s2.loc["2020-04-01":"2020-04-30"] = np.nan
    s2.loc["2020-07"] = 1.0

    scr = SeriesComparisonRelative(s1, s2, b)

    ax = scr.plots.plot_relative_comparison(
        mark_unique=True, mark_different=True, mark_identical=True, mark_introduced=True
    )

    return ax

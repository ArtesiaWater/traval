import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ComparisonPlots:

    def __init__(self, cp):
        self.cp = cp

    def plot_series_comparison(self, mark_unique=True, mark_different=True,
                               mark_identical=True, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        else:
            fig = ax.figure

        plot_handles = []

        # Plot both series
        for s, c, ls in zip([self.cp.s1n, self.cp.s2n],
                            ["gray", "k"],
                            ["solid", "dashed"]):
            p1, = ax.plot(s.index, s, c=c, marker=None, ls=ls, label=s.name)
            plot_handles.append(p1)

            # Mark differences between both in red (do for both lines)
            if mark_different:
                s_diff = s.copy()
                not_diff = s_diff.index.difference(
                    self.cp.idx_in_both_different)
                s_diff.loc[not_diff] = np.nan
                p2, = ax.plot(s_diff.index, s_diff, lw=3, marker=None,
                              ls="solid", c="red", alpha=0.3,
                              label="different")
        # add to legend once
        if mark_different:
            plot_handles.append(p2)

        # Mark sections with identical measurements in green (do for one line)
        if mark_identical:
            s_identical = self.cp.s1n.copy()
            not_identical = s_identical.index.difference(
                self.cp.idx_in_both_identical)
            s_identical.loc[not_identical] = np.nan
            p5, = ax.plot(s_identical.index, s_identical,
                          marker=None, ls="solid", c="LimeGreen",
                          label="identical", lw=3, alpha=0.5)
            plot_handles.append(p5)

        # Mark unique observations with x's if they exist
        if mark_unique:
            if self.cp.idx_in_s1.size > 0:
                p3, = ax.plot(self.cp.idx_in_s1,
                              self.cp.s1.loc[self.cp.idx_in_s1],
                              marker="x", ms=5, ls="none", c="Orange",
                              label="only in series 1: {}".format(
                                  self.cp.s1n.name))
                plot_handles.append(p3)
            if self.cp.idx_in_s2.size > 0:
                p4, = ax.plot(self.cp.idx_in_s2,
                              self.cp.s2.loc[self.cp.idx_in_s2],
                              marker="x", ms=5, ls="none", c="Blue",
                              label="only in series 2: {}".format(
                                  self.cp.s2.name))
                plot_handles.append(p4)

        # Add legend and other plot stuff
        plot_labels = [i.get_label() for i in plot_handles]
        ax.legend(plot_handles, plot_labels, loc="best",
                  ncol=int(np.ceil(len(plot_handles) / 2.)))
        ax.grid(b=True)
        fig.tight_layout()
        return ax

    def plot_relative_comparison(self, mark_unique=True, mark_different=True,
                                 mark_identical=True, mark_introduced=False,
                                 ax=None):

        ax = self.plot_series_comparison(mark_unique=mark_unique,
                                         mark_different=mark_different,
                                         mark_identical=mark_identical,
                                         ax=ax)

        plot_handles, plot_labels = ax.get_legend_handles_labels()
        # remove duplicates
        for ilbl in plot_labels:
            if plot_labels.count(ilbl) > 1:
                idx = plot_labels.index(ilbl)
                plot_labels.remove(ilbl)
                plot_handles.remove(plot_handles[idx])

        # Add an base series (i.e. raw data to the plot)
        p0, = ax.plot(self.cp.basen.index, self.cp.basen, lw=0.5, c="k",
                      label="base series", ls="solid", zorder=2)
        # insert entry at beginning
        plot_handles.insert(0, p0)
        plot_labels.insert(0, p0.get_label())

        # mark flagged in both
        s_base = pd.Series(index=self.cp.basen.index, data=np.nan, dtype=float)
        s_base.loc[self.cp.idx_r_flagged_in_both] = \
            self.cp.basen.loc[self.cp.idx_r_flagged_in_both]
        p6, = ax.plot(s_base.index,
                      s_base, lw=0.5, c="DarkOrchid",
                      ls="none", marker="x", ms=5,
                      label="flagged in both")
        plot_handles.append(p6)
        plot_labels.append(p6.get_label())

        if mark_introduced:
            intro_idx = (self.cp.idx_r_introduced_in_s2
                         .union(self.cp.idx_r_introduced_in_both))
            if ((self.cp.idx_r_introduced_in_s1.size > 0) or
                    (intro_idx.size > 0)):
                ax.plot(self.cp.s1n.loc[self.cp.idx_r_introduced_in_s1].index,
                        self.cp.s1n.loc[self.cp.idx_r_introduced_in_s1],
                        c="Coral", ls="none", marker="x", ms=5,
                        label="introduced in s1/s2")
                p7, = ax.plot(self.cp.s1n.loc[intro_idx].index,
                              self.cp.s2n.loc[intro_idx],
                              c="Coral", ls="none", marker="x", ms=5,
                              label="introduced in s1/s2")
                plot_handles.append(p7)
                plot_labels.append(p7.get_label())

        ax.legend(plot_handles, plot_labels, loc="best",
                  ncol=int(np.ceil(len(plot_handles) / 3.)))

        return ax


def roc_plot(tpr, fpr, labels, ax=None, plot_diagonal=True, **kwargs):
    """Receiver operator characteristic plot.

    Plots the false positive rate (x-axis) versus the 
    true positive rate (y-axis). The 'tpr' and 'fpr' can be passed as:
    - values: outcome of a single error detection algorithm
    - arrays: outcomes of error detection algorithm in which a detection 
      parameter is varied. 
    - lists: for passing multiple results, entries can be values or
      arrays, as listed above. 

    Parameters
    ----------
    tpr : list or value or array
        true positive rate. If passed as a list loops through each 
        entry and plots it. Otherwise just plots the array or value.
    fpr : list or value or array
        false positive rate. If passed as a list loops through each 
        entry and plots it. Otherwise just plots the array or value.
    labels : list or str
        label for each tpr/fpr entry.
    ax : matplotlib.pyplot.Axes, optional
        axes to plot on, default is None, which creates new figure
    plot_diagonal : bool, optional
        whether to plot the diagonal (useful for combining multiple 
        ROC plots)
    **kwargs
        passed to ax.plot

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        axes instance

    """
    if not isinstance(tpr, list):
        tpr = [tpr]
    if not isinstance(fpr, list):
        fpr = [fpr]
    if not isinstance(labels, list):
        labels = [labels]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], ls="dashed", lw=1.0, c="k",
                label="random guess")

    for itpr, ifpr, ilbl in zip(tpr, fpr, labels):
        ax.plot(ifpr, itpr, marker="o", label=ilbl, ls="none", **kwargs)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(b=True)
    ax.legend(loc="lower right")
    ax.set_ylabel("True Positive Rate (sensitivity)")
    ax.set_xlabel("False Positive Rate (1-specificity)")
    ax.set_title("receiver operator characteristic plot")

    fig.tight_layout()

    return ax


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.interactive(True)

    from ts_comparison import SeriesComparison, SeriesComparisonRelative

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

    cpp = ComparisonPlots(sc)
    cpp.plot_series_comparison(
        mark_different=True, mark_identical=True, mark_unique=True)

    scr = SeriesComparisonRelative(s1, s2, b)

    cpp = ComparisonPlots(scr)
    cpp.plot_relative_comparison(
        mark_unique=True, mark_diff=True,
        mark_identical=True, mark_introduced=True)

    # the other test
    testdf = pd.DataFrame(index=pd.date_range(
        "2020-01-01", freq="D", periods=16), columns=["orig", "s1", "s2"],
        data=0.0)

    # -9999 means obs is missing entirely
    missing = -9999.

    # expected possibilities
    testdf.iloc[0] = 1.0, 1.0, 1.0               # 1   # kept_in_both
    testdf.iloc[1] = 1.0, 1.0, np.nan            # 2   # flagged_in_s2
    testdf.iloc[2] = 1.0, 1.0, missing           # 3   # flagged_in_s2
    testdf.iloc[3] = 1.0, np.nan, missing        # 4   # flagged_in_both
    testdf.iloc[4] = 1.0, np.nan, np.nan         # 5   # flagged_in_both
    testdf.iloc[5] = 1.0, missing, missing       # 6   # flagged_in_both

    # weirdness level 1
    testdf.iloc[6] = np.nan, 1.0, 1.0            # 7   # introduced_in_both
    testdf.iloc[7] = np.nan, 1.0, np.nan         # 8   # introduced_in_s1
    testdf.iloc[8] = np.nan, 1.0, missing        # 9   # introduced_in_s1
    testdf.iloc[9] = np.nan, missing, missing    # 10  # in_all_nan
    testdf.iloc[10] = np.nan, np.nan, np.nan     # 11  # in_all_nan

    # weirdness level 2
    testdf.iloc[11] = missing, 1.0, 1.0          # 12  # introduced_in_both
    testdf.iloc[12] = missing, 1.0, np.nan       # 13  # introduced_in_s1
    testdf.iloc[13] = missing, 1.0, missing      # 14  # introduced_in_s1

    # weirdness level 3 (for testing)
    testdf.iloc[14] = missing, np.nan, np.nan    # 15  # do not count...
    testdf.iloc[15] = missing, missing, missing  # 16  # do not count...

    orig = testdf.orig.loc[testdf.orig != missing]
    s1 = testdf.s1.loc[testdf.s1 != missing]
    s2 = testdf.s2.loc[testdf.s2 != missing]

    scr = SeriesComparisonRelative(s1, s2, orig)

    cpp = ComparisonPlots(scr)
    cpp.plot_relative_comparison(
        mark_unique=True, mark_different=True, mark_identical=True)

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

    scr = SeriesComparisonRelative(s1, s2, b)

    scr.plots.plot_relative_comparison(mark_unique=True, mark_introduced=True)

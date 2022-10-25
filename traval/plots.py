import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm


class ComparisonPlots:
    """Mix-in class for plots for comparing timeseries."""

    color_dict = {
        "only_in_s1": {"color": "orange"},
        "only_in_s2": {"color": "blue"},
        "identical": {"color": "LimeGreen", "alpha": 0.5},
        "different": {"color": "Red", "alpha": 0.3},
        "flagged_in_both": {"color": "DarkOrchid"},
        "introduced": {"color": "Coral"},
    }

    def __init__(self, cp):
        """Initialize comparison plots mix-in class.

        Parameters
        ----------
        cp : SeriesComparison
            traval comparison object
        """
        self.cp = cp

    def update_color_dict(self, key, color=None, alpha=None):
        """Update colors for plots.

        Parameters
        ----------
        key : str
            name of category to update, see
            `ComparisonPlots.color_dict.keys()` for options
        color : str, optional
            color name, by default None
        alpha : float, optional
            alpha value, by default None
        """
        d = self.color_dict[key]
        if color is not None:
            d.update({"color": color})
        if alpha is not None:
            d.update({"alpha": alpha})

    def reset_color_dict(self):
        """Reset color_dict to default values."""
        self.color_dict = {
            "only_in_s1": {"color": "orange"},
            "only_in_s2": {"color": "blue"},
            "identical": {"color": "LimeGreen", "alpha": 0.5},
            "different": {"color": "Red", "alpha": 0.3},
            "flagged_in_both": {"color": "DarkOrchid"},
            "introduced": {"color": "Coral"},
        }

    def plot_series_comparison(self, mark_unique=True, mark_different=True,
                               mark_identical=True, ax=None):
        """Plot comparison between two timeseries.

        Parameters
        ----------
        mark_unique : bool, optional
            mark unique values with colored X's, by default True
        mark_different : bool, optional
            highlight where timeseries differ with red, by default True
        mark_identical : bool, optional
            highlight where timeseries are identical with green,
            by default True
        ax : axis, optional
            axis object to plot on, by default None

        Returns
        -------
        ax : axis
            axis object
        """

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
                              ls="solid", label="different",
                              **self.color_dict["different"])
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
                          marker=None, ls="solid", label="identical", lw=3,
                          **self.color_dict["identical"])
            plot_handles.append(p5)

        # Mark unique observations with x's if they exist
        if mark_unique:
            if self.cp.idx_in_s1.size > 0:
                p3, = ax.plot(self.cp.idx_in_s1,
                              self.cp.s1.loc[self.cp.idx_in_s1],
                              marker="x", ms=5, ls="none",
                              **self.color_dict["only_in_s1"],
                              label="only in series 1: {}".format(
                                  self.cp.s1n.name))
                plot_handles.append(p3)
            if self.cp.idx_in_s2.size > 0:
                p4, = ax.plot(self.cp.idx_in_s2,
                              self.cp.s2.loc[self.cp.idx_in_s2],
                              marker="x", ms=5, ls="none",
                              **self.color_dict["only_in_s2"],
                              label="only in series 2: {}".format(
                                  self.cp.s2.name))
                plot_handles.append(p4)

        # Add legend and other plot stuff
        plot_labels = [i.get_label() for i in plot_handles]
        ax.legend(plot_handles, plot_labels, loc="best",
                  ncol=int(np.ceil(len(plot_handles) / 2.)))
        ax.grid(visible=True)
        fig.tight_layout()
        return ax

    def plot_relative_comparison(self, mark_unique=True, mark_different=True,
                                 mark_identical=True, mark_introduced=False,
                                 ax=None):
        """Plot comparison between two timeseries relative to base timeseries.

        Parameters
        ----------
        mark_unique : bool, optional
            mark unique observations with colored X's, by default True
        mark_different : bool, optional
            highlight where series are different in red, by default True
        mark_identical : bool, optional
            highlight where series are identical with green, by default True
        mark_introduced : bool, optional
            mark observations that are not in the base timeseries with X's,
            by default False
        ax : axis, optional
            axis to plot on, by default None

        Returns
        -------
        ax : axis
            axis handle
        """

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
        if self.cp.idx_r_flagged_in_both.size > 0:
            s_base = pd.Series(index=self.cp.basen.index,
                               data=np.nan, dtype=float)
            s_base.loc[self.cp.idx_r_flagged_in_both] = \
                self.cp.basen.loc[self.cp.idx_r_flagged_in_both]
            p6, = ax.plot(s_base.index,
                          s_base, lw=0.5,
                          **self.color_dict["flagged_in_both"],
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
                        ls="none", marker="x", ms=5,
                        **self.color_dict["introduced"],
                        label="introduced in s1/s2")
                p7, = ax.plot(self.cp.s1n.loc[intro_idx].index,
                              self.cp.s2n.loc[intro_idx],
                              ls="none", marker="x", ms=5,
                              **self.color_dict["introduced"],
                              label="introduced in s1/s2")
                plot_handles.append(p7)
                plot_labels.append(p7.get_label())

        ax.legend(plot_handles, plot_labels, loc="best",
                  ncol=int(np.ceil(len(plot_handles) / 3.)))

        return ax

    def plot_validation_result(self, ax=None):
        # Some plot settings
        ms_valid = 6  # markersize validation result
        mew = 1.25  # markeredgewidth validation result

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        else:
            fig = ax.figure

        plot_handles = []

        # Add an original series i.e. raw data to the plot
        p0, = ax.plot(self.cp.basen.index, self.cp.basen, lw=0.5, c="k",
                      marker=".", ms=3, label="base series", ls="solid")
        plot_handles.append(p0)

        # set marker colors
        c = pd.Series(index=self.cp.basen.index, data='')
        c.loc[self.cp.idx_r_flagged_in_both] = "Green"
        c.loc[self.cp.idx_r_flagged_in_s1] = "DarkOrange"
        c.loc[self.cp.idx_r_flagged_in_s2] = "Red"

        mask = c != ""

        s = self.cp.basen.loc[mask]
        c = c.loc[mask]

        sc = ax.scatter(s.index, s.values, c=c.values, s=ms_valid**2,
                        linewidths=mew, marker="o", edgecolor=c.values,
                        zorder=10)
        sc.set_facecolor("none")

        dummy1, = ax.plot([], [], c="Green", marker="o", mfc="none", mew=mew,
                          ls="none", ms=ms_valid, label="Correctly flagged (TP)")
        dummy2, = ax.plot([], [], c="DarkOrange", marker="o", mfc="none", mew=mew,
                          ls="none", ms=ms_valid, label="Incorrectly flagged (FP)")
        dummy3, = ax.plot([], [], c="Red", marker="o", mfc="none", mew=mew,
                          ls="none", ms=ms_valid, label="Wrongly kept (FN)")

        plot_handles += [dummy1, dummy2, dummy3]

        # Add legend and other plot stuff
        plot_labels = [i.get_label() for i in plot_handles]
        ax.legend(plot_handles, plot_labels, loc=(0, 1), markerscale=1.25,
                  ncol=len(plot_handles), frameon=False)
        ax.grid(visible=True)
        fig.tight_layout()

        return ax


def roc_plot(tpr, fpr, labels, colors=None, ax=None,
             plot_diagonal=True, colorbar_label=None, **kwargs):
    """Receiver operator characteristic plot.

    Plots the false positive rate (x-axis) versus the
    true positive rate (y-axis). The 'tpr' and 'fpr' can be passed as:
    -  values: outcome of a single error detection algorithm
    -  arrays: outcomes of error detection algorithm in which a detection
       parameter is varied.
    -  lists: for passing multiple results, entries can be values or
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
        passed to ax.scatter

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
        sc = ax.scatter(ifpr, itpr, s=6**2, c=colors,
                        marker="o", label=ilbl, **kwargs)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(visible=True)
    ax.legend(loc="lower right")
    ax.set_ylabel("True Positive Rate (sensitivity)")
    ax.set_xlabel("False Positive Rate (1-specificity)")
    ax.set_title("receiver operator characteristic plot")

    if colors is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(sc, cax=cax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    fig.tight_layout()
    return ax


def det_plot(fpr, fnr, labels, ax=None, **kwargs):
    """Detection Error Tradeoff plot.

    Adapted from scikitlearn `DetCurveDisplay`.

    Parameters
    ----------
    fpr : list or value or array
        false positive rate. If passed as a list loops through each
        entry and plots it. Otherwise just plots the array or value.
    fnr : list or value or array
        false negative rate. If passed as a list loops through each
        entry and plots it. Otherwise just plots the array or value.
    labels : list or str
        label for each fpr/fnr entry.
    ax : matplotlib.pyplot.Axes, optional
        axes handle to plot on, by default None, which
        creates a new figure

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        axes handle
    """

    if not isinstance(fpr, list):
        fpr = [fpr]
    if not isinstance(fnr, list):
        fnr = [fnr]
    if not isinstance(labels, list):
        labels = [labels]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.set_aspect("equal")

    for ifpr, ifnr, ilbl in zip(fpr, fnr, labels):
        ax.plot(norm.ppf(ifpr), norm.ppf(ifnr), marker="o",
                ls="none", label=ilbl, **kwargs)

    xlabel = "False Positive Rate"
    ylabel = "False Negative Rate"
    ax.set(xlabel=xlabel, ylabel=ylabel)

    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = norm.ppf(ticks)
    tick_labels = [
        '{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s)
        for s in ticks
    ]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-3, 3)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-3, 3)
    ax.grid(visible=True)

    ax.set_title("detection error tradeoff plot")

    # fig.tight_layout()

    return ax

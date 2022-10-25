from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ts_comparison import SeriesComparison, SeriesComparisonRelative
from .ts_utils import unique_nans_in_series


class Detector:
    """Detector object for applying error detection algorithms to timeseries.

    The Detector is used to apply error detection algorithms to a timeseries
    and optionally contains a 'truth' series, to which the error detection
    result can be compared. An example of a 'truth' series is a manually
    validated timeseries. Custom error detection algorithms can be defined
    using the RuleSet object.


    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        timeseries to check
    truth : pd.Series or pd.DataFrame, optional
        series that represents the 'truth', i.e. a benchmark to which
        the error detection result can be compared, by default None


    Examples
    --------

    Given a timeseries 'series' and some ruleset 'rset':

    >>> d = Detector(series)
    >>> d.apply_ruleset(rset)
    >>> d.plot_overview()


    See also
    --------
    traval.RuleSet : object for defining detection algorithms
    """

    def __init__(self, series, truth=None):
        """Initialize Detector object.

        Parameters
        ----------
        series : pd.Series or pd.DataFrame
            timeseries to check
        truth : pd.Series or pd.DataFrame, optional
            series that represents the 'truth', i.e. a benchmark to which
            the error detection result can be compared, by default None
        """
        # validate input series
        name = self._validate_input_series(series)
        if name is None:
            self.name = ""
        else:
            self.name = name
        self.series = series

        if truth is not None:
            self.set_truth(truth)
        else:
            self.truth = None

    def __repr__(self):
        """String representation of Detector object."""
        return f"Detector: <{self.name}>"

    @staticmethod
    def _validate_input_series(series):
        """Internal method for checking type and dtype of series.

        Parameters
        ----------
        series : object
            timeseries to check, must be pd.Series or pd.DataFrame. Datatype
            of series or first column of DataFrame must be float.

        Raises
        ------
        TypeError
            if series or dtype of series does not comply
        """

        # check pd.Series or pd.DataFrame
        if isinstance(series, pd.Series):
            dtype = series.dtypes
            name = series.name
        elif isinstance(series, pd.DataFrame):
            dtype = series.dtypes.values[0]
            name = series.columns[0]
        else:
            raise TypeError(
                "Series must be pandas.Series or pandas.DataFrame!")
        # check dtype (of first col)
        if not dtype in [float, np.float32]:
            raise TypeError("Series (or first column of DataFrame) must "
                            "have dtype float!")
        return name

    def reset(self):
        """Reset Detector object."""
        for attr in ["ruleset", "results",
                     "corrections", "comparisons"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def apply_ruleset(self, ruleset, compare=True):
        """Apply RuleSet to series.

        Parameters
        ----------
        ruleset : traval.RuleSet
            RuleSet object containing detection rules
        compare : bool or list of int, optional
            if True, compare all results to original series and store in
            dictionary under comparisons attribute, default is True. If False,
            do not store comparisons. If list of int, store only those step
            numbers as comparisons. Note: value of -1 refers to last step
            for convenience.


        See also
        --------
        traval.RuleSet : object for defining detection algorithms
        """
        self.ruleset = ruleset
        d, c = self.ruleset(self.series)

        # store corrections, results
        self.corrections = c
        self.results = d

        # if compare is not False do comparison
        if compare:
            self.comparisons = {}
            base = d[0].copy()

            # if compare is not list, get all step numbers
            if not isinstance(compare, list):
                compare = d.keys()

            # do comparison
            for k in compare:
                # if k is negative, convert to step number from end
                if k < 0:
                    k = len(d.keys()) + k
                # only do comparison for steps, not base series
                if k > 0:
                    s = d[k]
                    s.name = self.ruleset.get_step_name(k)
                    base.name = "base series"
                    if self.truth is None:
                        self.comparisons[k] = SeriesComparison(s, base)
                    else:
                        self.comparisons[k] = SeriesComparisonRelative(
                            s, self.truth, base)

    def set_truth(self, truth):
        """Set 'truth' series.

        Used for comparison with detection result.

        Parameters
        ----------
        truth : pd.Series or pd.DataFrame
            Series or DataFrame containing the "truth", i.e. a benchmark
            to compare the detection result to.
        """
        self._validate_input_series(truth)
        self.truth = truth

    def confusion_matrix(self, steps=None, truth=None):
        """Calculate confusion matrix stats for detection rules.

        Note: the calculated statistics per rule contain overlapping counts,
        i.e. multiple rules can mark the same observatin as suspect.

        Parameters
        ----------
        steps : int, list of int or None, optional
            steps for which to calculate confusion matrix statistics, by
            default None which uses all steps.
        truth : pd.Series or pd.DataFrame, optional
            series representing the "truth", i.e. a benchmark to which the
            resulting series is compared. By default None, which uses the
            stored truth series. Argument is included so a different truth
            can be passed.

        Returns
        -------
        df : pd.DataFrame
            dataframe containing confusion matrix data, i.e. counts of true
            positives, false positives, true negatives and false negatives.
        """
        # get list of step integers
        if isinstance(steps, int):
            steps = [steps]
        if not isinstance(steps, list):
            steps = self.results.keys()

        # use truth if provided, else use stored truth
        if truth is None:
            truth = self.truth

        # get rule names
        rulenames = [self.ruleset.get_step_name(i) for i in steps]

        df = pd.DataFrame(index=steps,
                          columns=["rule", "TP", "FP", "FN", "TN"])
        df.loc[:, "rule"] = rulenames
        base = self.results[0]
        base.name = "base series"

        # loop over steps
        for k in steps:
            # if k is negative, convert to step number from end
            if k < 0:
                k = len(self.results.keys()) + k
            # only do comparison for steps, not base series
            if k > 0:
                s = self.results[k]
                s.name = rulenames[k]
                cp = SeriesComparisonRelative(s, truth, base)

                # store stats
                df.loc[k, ["TP", "FP", "FN", "TN"]] = (cp.bc.tp,
                                                       cp.bc.fp,
                                                       cp.bc.fn,
                                                       cp.bc.tn)
        return df

    def uniqueness(self, truth=None):
        """Calculate unique contribution per rule to stats.

        Note: the calculated statistics per rule contain an undercount,
        i.e. when multiple rules mark the same observatin as suspect it is
        not contained in this result.

        Parameters
        ----------
        steps : int, list of int or None, optional
            steps for which to calculate confusion matrix statistics, by
            default None which uses all steps.
        truth : pd.Series or pd.DataFrame, optional
            series representing the "truth", i.e. a benchmark to which the
            resulting series is compared. By default None, which uses the
            stored truth series. Argument is included so a different truth
            can be passed.

        Returns
        -------
        df : pd.DataFrame
            dataframe containing confusion matrix data, i.e. unique counts
            of true positives, false positives, true negatives and
            false negatives.
        """
        steps = list(self.results.keys())[1:]

        # use truth if provided, else use stored truth
        if truth is None:
            truth = self.truth

        base = self.results[0]
        base.name = "base series"

        # last step, skip in comparison as this presumably contains all NaNs
        last_step = max(steps)
        steps.remove(last_step)

        # get rule names
        rulenames = [self.ruleset.get_step_name(i) for i in steps]

        df = pd.DataFrame(index=steps,
                          columns=["rule", "TP", "FP", "FN", "TN"])
        df.loc[:, "rule"] = rulenames

        for j, k in enumerate(steps):
            series_list = deepcopy(self.results)
            s = series_list.pop(k)
            series_list.pop(last_step)
            other_series = list(series_list.values())
            mask = unique_nans_in_series(s, *other_series)
            s.loc[~mask & s.isna()] = -9999.  # some random non-NaN number
            s.name = rulenames[j]
            cp = SeriesComparisonRelative(s, truth, base)

            # store stats
            df.loc[k, ["TP", "FP", "FN", "TN"]] = (cp.bc.tp,
                                                   cp.bc.fp,
                                                   cp.bc.fn,
                                                   cp.bc.tn)
        return df

    def stats_per_comment(self, step=None, truth=None):

        if step is None:
            step = list(self.results.keys())[-1]
        elif step < 0:
            step = len(self.results.keys()) + step

        if truth is None:
            truth = self.truth

        # get rule names
        rulename = self.ruleset.get_step_name(step)

        base = self.results[0]
        base.name = "base series"

        s = self.results[step]
        s.name = rulename

        cp = SeriesComparisonRelative(s, truth, base)
        stats = cp.compare_to_base_by_comment()

        cols = {
            "TP": 'flagged_in_both',
            "FP": 'flagged_in_s1',
            "FN": 'flagged_in_s2',
            "TN": 'kept_in_both'
        }
        df = stats.loc[cols.values(), :].transpose()
        df.index.name = rulename
        df.rename(columns=cols, inplace=True)
        return df

    def get_series(self, step, category=None):
        base = self.results[0]
        base.name = "base series"

        series = [base, self.results[step]]

        if self.truth is not None:
            truth = self.truth
            series.append(truth)

        df = pd.concat(series, axis=1)

        if category is not None:
            idx = self.get_indices(category=category, step=step)
            df = df.loc[idx]

        return df

    def get_indices(self, category, step, truth=None):

        s = self.results[step]
        base = self.results[0]
        base.name = "base series"

        if truth is None:
            truth = self.truth

        cp = SeriesComparisonRelative(s, truth, base)

        if category.lower() in ["tp", "true_positives"]:
            idx = cp.idx_r_flagged_in_both
        elif category.lower() in ["fp", "false_positives"]:
            idx = cp.idx_r_flagged_in_s1
        elif category.lower() in ["fn", "false_negatives"]:
            idx = cp.idx_r_flagged_in_s2
        elif category.lower() in ["tn", "true_negatives"]:
            idx = cp.idx_r_kept_in_both
        else:
            raise ValueError(f"Category '{category}' not recognized, must "
                             "be one of ('tp', 'fp', 'fn', 'tn')")

        return idx

    def get_comment_series(self, steps=None):

        # get list of step integers
        if isinstance(steps, int):
            if steps < 0:
                steps = [len(self.results.keys()) + steps]
            else:
                steps = [steps]
        if not isinstance(steps, list):
            steps = list(self.results.keys())[1:]

        # get rule names
        rulenames = [self.ruleset.get_step_name(i) for i in steps]

        # get corrections
        corr = self.get_corrections_dataframe()

        if corr.empty:
            corr = pd.DataFrame(index=self.series.index,
                                columns=rulenames, data=0.0)
        else:
            corr = corr.loc[:, rulenames]

        comments = []
        for col in corr.columns:
            s = corr[col].copy()
            s = s.replace(0.0, "").replace(np.nan, col)
            comments.append(s)

        comments = pd.concat(comments, axis=1).apply(
            lambda s: ",".join(s[s != ""]), axis=1)
        comments = comments.replace(np.nan, "")
        comments.name = "comment"

        return comments

    def get_results_dataframe(self):
        """Get results as DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            results with flagged values set to NaN per applied rule.
        """
        df = pd.concat(self.results.values(), axis=1)
        df.columns = ["base series"] + list(self.ruleset.rules.keys())
        return df

    def get_final_result(self):
        """Get final timeseries with flagged values set to NaN.

        Returns
        -------
        series : pandas.Series
            Timeseries produced by final step in RuleSet with flagged
            values set to NaN.
        """
        key = len(self.results.keys()) - 1
        s = self.results[key]
        s.name = self.name
        return s

    def get_corrections_dataframe(self):
        """Get DataFrame containing corrections.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing corrections. NaN means value is flagged
            as suspicious, 0.0 means no correction.
        """
        clist = []
        for s in self.corrections.values():
            if isinstance(s, np.ndarray):
                s = pd.Series(dtype=float)
            clist.append(s.fillna(-9999))

        # corrections are nan, 0.0 means nothing is changed
        df = (pd.concat(clist, axis=1)
              .isna()
              .astype(float)
              .replace(0.0, np.nan)
              .replace(1.0, 0.0))
        df.columns = list(self.ruleset.rules.keys())
        return df

    def get_corrections_comparison(self, truth=None):

        if truth is None and self.truth is not None:
            truth = self.truth
        else:
            raise ValueError("Supply a time series for 'truth'!")

        comments_traval = self.get_comment_series()
        comments_traval.name = "traval_comment"

        mask_truth_corrections = truth.iloc[:, 0].isna()
        comments_truth = truth.loc[mask_truth_corrections]

        k = list(self.comparisons.keys())[-1]
        comparison = self.comparisons[k].comparison_series()
        translate = {
            -1: "Value modified",
            0: "Flagged in both",
            1: "Only flagged in 'truth' series",
            2: "Only flagged in 'traval' series",
            -9999: "NaN in both"
        }
        comparison = comparison.apply(lambda v: translate[v])
        comparison.name = "comparison_label"

        raw_index = (comments_traval.index
                     .union(comments_truth.index))

        truth.columns = ["truth_series", "truth_comment"]

        traval_series = self.get_final_result()
        traval_series.name = "traval_series"

        df = pd.concat([
            self.series.loc[raw_index.intersection(self.series.index)],
            traval_series.loc[raw_index.intersection(traval_series.index)],
            comments_traval,
            truth.loc[raw_index.intersection(truth.index)],
            comparison.loc[raw_index.intersection(comparison.index)]
        ], axis=1)

        return df

    def plot_overview(self, mark_suspects=True, **kwargs):
        """Plot timeseries with flagged values per applied rule.

        Parameters
        ----------
        mark_suspects : bool, optional
            mark suspect values with red X, by default True

        Returns
        -------
        ax : list of matplotlib.pyplot.Axes
            axes objects
        """
        resultsdf = self.get_results_dataframe()

        if "figsize" in kwargs:
            figsize = kwargs.pop("figsize")
        else:
            figsize = (12, 5)

        fig, axes = plt.subplots(len(self.corrections) + 1, 1,
                                 sharex=True, sharey=True, figsize=figsize,
                                 **kwargs)

        for iax, icol in zip(axes, resultsdf):
            iax.plot(resultsdf.index, resultsdf[icol], label=icol)

            if mark_suspects:
                if icol != resultsdf.columns[0]:
                    corr = self.corrections[resultsdf.columns.get_loc(icol)]
                    if isinstance(corr, pd.Series):
                        iax.plot(corr.index,
                                 resultsdf.loc[corr.index].iloc[:, 0],
                                 marker="x", c="C3", ls="none",
                                 label="flagged")

            iax.legend(loc="upper left", ncol=2)
            iax.grid(b=True)

        fig.tight_layout()
        return axes

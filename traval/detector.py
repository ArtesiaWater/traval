
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .ts_comparison import SeriesComparison, SeriesComparisonRelative


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
    >>> d.plot()


    See also
    --------
    RuleSet : object for defining detection algorithms

    """

    def __init__(self, series, truth=None):
        """Initialize Detector object

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
        return f"Detector: <{self.name}>"

    @staticmethod
    def _validate_input_series(series):
        """internal method for checking type and dtype of series

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
            dtype = series.dtypes[0]
            name = series.columns[0]
        else:
            raise TypeError(
                "Series must be pandas.Series or pandas.DataFrame!")
        # check dtype (of first col)
        if dtype != float:
            raise TypeError("Series (or first column of DataFrame) must "
                            "have dtype float!")
        return name

    def reset(self):
        """Reset Detector object
        """
        for attr in ["ruleset", "results",
                     "corrections", "comparisons"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def apply_ruleset(self, ruleset, compare=True):
        """Apply RuleSet to series

        Parameters
        ----------
        ruleset : RuleSet
            RuleSet object containing detection rules
        compare : bool or list of int, optional
            if True, compare all results to original series and store in
            dictionary under comparisons attribute, default is True. If False,
            do not store comparisons. If list of int, store only those step 
            numbers as comparisons. Note: value of -1 refers to last step 
            for convenience.


        See also
        --------
        RuleSet : object for defining detection algorithms

        """
        self.ruleset = ruleset
        d, c = self.ruleset(self.series)

        # store corrections, results
        self.corrections = c
        self.results = d

        # if compare is not False do comparison
        if compare:
            self.comparisons = {}
            base = d[0]
            base.name = "base series"
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
                    if self.truth is None:
                        self.comparisons[k] = SeriesComparison(s, base)
                    else:
                        self.comparisons[k] = SeriesComparisonRelative(
                            s, self.truth, base)

    def set_truth(self, truth):
        """set 'truth' series. 

        Used for comparison with detection result.

        Parameters
        ----------
        truth : pd.Series or pd.DataFrame
            Series or DataFrame containing the "truth", i.e. a benchmark
            to compare the detection result to.

        """
        self._validate_input_series(truth)
        self.truth = truth

    def confusion_matrix(self, series=None, truth=None):
        """calculate confusion matrix.

        Parameters
        ----------
        series : pd.Series or pd.DataFrame, optional
            resulting series, by default None which uses the final
            result after applying ruleset. Argument is included so the
            confusion matrix can also be calculated for intermediate results.
        truth : pd.Series or pd.DataFrame, optional
            series representing the "truth", i.e. a benchmark to which the
            resulting series is compared. By default None, which uses the
            stored truth series. Argument is included so a different truth 
            can be passed.

        Returns
        -------
        confusion_matrix : pd.DataFrame
            confusion matrix containing counts of true positives, 
            false positives, true negatives and false negatives.

        """
        pass

    def get_results_dataframe(self):
        df = pd.concat([s for s in self.results.values()],
                       axis=1)
        df.columns = ["base series"] + list(self.ruleset.rules.keys())
        return df

    def get_final_result(self):
        key = len(self.results.keys()) - 1
        s = self.results[key]
        s.name = self.name
        return s

    def get_corrections_dataframe(self):
        clist = []
        for s in self.corrections.values():
            if isinstance(s, np.ndarray):
                s = pd.Series()
            clist.append(s.fillna(1.0))
        # corrections are nan, 0.0 means nothing is changed
        df = (pd.concat(clist, axis=1)
              .isna()
              .replace(False, np.nan)
              .replace(True, 0.0))
        df.columns = list(self.ruleset.rules.keys())
        return df

    def plot_overview(self, mark_suspects=True):
        resultsdf = self.get_results_dataframe()

        _, axes = plt.subplots(len(self.corrections) + 1, 1,
                               figsize=(12, 5), dpi=100, sharex=True,
                               sharey=True)

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

        return axes

import numpy as np
import pandas as pd

from .binary_classifier import BinaryClassifier
from .plots import ComparisonPlots


class DateTimeIndexComparison:
    """Helper class for comparing two DateTimeIndexes."""

    def __init__(self, idx1, idx2):
        """Create object for comparing DateTimeIndex objects.

        Parameters
        ----------
        idx1 : DateTimeIndex
            DateTimeIndex #1
        idx2 : DateTimeIndex
            DateTimeIndex #2
        """
        self.idx1 = idx1
        self.idx2 = idx2

    def idx_in_both(self):
        """Index members in both DateTimeIndexes.

        Returns
        -------
        DateTimeIndex
            index with entries in both
        """
        return self.idx1.intersection(self.idx2)

    def idx_in_idx1(self):
        """Index members only in Index #1.

        Returns
        -------
        DateTimeIndex
            index with entries only in index #1
        """
        return self.idx1.difference(self.idx2)

    def idx_in_idx2(self):
        """Index members only in Index #2.

        Returns
        -------
        DateTimeIndex
            index with entries only in index #2
        """
        return self.idx2.difference(self.idx1)


class SeriesComparison:
    """Object for comparing two timeseries.

    Comparison yields the following categories:

    -  in_both_identical: in both series and difference <= than diff_threshold
    -  in_both_different: in both series and difference > than diff_threshold
    -  in_s1: only in series #1
    -  in_s2: only in series #2
    -  in_both_nan: NaN in both


    Parameters
    ----------
    s1 : pd.Series or pd.DataFrame
        first series to compare
    s2 : pd.Series or pd.DataFrame
        second series to compare
    diff_threshold : float, optional
        value beyond which a difference is considered significant, by
        default 0.0. Two values whose difference is smaller than threshold
        are considered identical.
    """

    def __init__(self, s1, s2, names=None, diff_threshold=0.0):
        """Compare two timeseries.

        Parameters
        ----------
        s1 : pd.Series or pd.DataFrame
            first series to compare
        s2 : pd.Series or pd.DataFrame
            second series to compare
        names : list of str, optional
            list of names of timeseries, by default None, which
            uses series name, or dataframe column name
        diff_threshold : float, optional
            value beyond which a difference is considered significant, by
            default 0.0. Two values whose difference is smaller than threshold
            are considered identical.
        """
        self.diff_threshold = diff_threshold

        # With NaNs
        self.s1n, self.c1n = self._parse_series(s1)
        self.s2n, self.c2n = self._parse_series(s2)
        # identify where both NaN
        self._compare_indices_with_nans()

        if names is not None:
            self.s1n.name = names[0]
            self.s2n.name = names[1]
        if self.s1n.name is None:
            self.s1n.name = "series1"
        if self.s2n.name is None:
            self.s2n.name = "series2"

        # Without NaNs
        self.s1 = self.s1n.loc[~self.s1n.isna()]
        self.s2 = self.s2n.loc[~self.s2n.isna()]
        self._compare_indices_without_nans()

        # Compare values (subset of idx_in_both)
        d_idx, i_idx = self._compare_series_values()
        self.idx_in_both_identical = i_idx
        self.idx_in_both_different = d_idx

        # Summarize comparison
        self.summary = self._summarize_series_comparison()

        # add plotting
        self.plots = ComparisonPlots(self)

    @staticmethod
    def _parse_series(series):
        """Internal method to parse timeseries input.

        Parameters
        ----------
        series : pd.Series or pd.DataFrame
            series or dataframe to parse, in the case of dataframe,
            column 1 must hold the data and column 2 the comments

        Returns
        -------
        series, comments :  pd.Series, pd.Series
            returns timeseries and comment series. Comment series is empty
            series if no comments are included in input

        Raises
        ------
        TypeError
            if series is not of type pd.Series or pd.DataFrame
        """
        if isinstance(series, pd.DataFrame):
            if len(series.columns) == 2:
                return series.iloc[:, 0], series.iloc[:, 1]
            elif len(series.columns) == 1:
                return series.iloc[:, 0], pd.Series(dtype='object')
            else:
                raise ValueError("Cannot interpret DataFrame columns. "
                                 "Pass DataFrame with first column containing "
                                 "values, and optional second column with "
                                 "comments.")
        elif isinstance(series, pd.Series):
            return series, pd.Series(dtype='object')
        else:
            raise TypeError("Provide pandas.Series or pandas.DataFrame!")

    def _compare_indices_with_nans(self):
        """Internal method for identifying indices with NaNs in both series."""
        nanmask1 = self.s1n.isna()
        nanmask2 = self.s2n.isna()

        idxcomp_nan = DateTimeIndexComparison(self.s1n.loc[nanmask1].index,
                                              self.s2n.loc[nanmask2].index)

        # weird categories NaNs in s1 but missing in s2
        self.idx_in_s1_nan = idxcomp_nan.idx_in_idx1()
        # weird categories NaNs in s2 but missing in s1
        self.idx_in_s2_nan = idxcomp_nan.idx_in_idx2()

        # in_both_nan
        self.idx_in_both_nan = idxcomp_nan.idx_in_both()

    def _compare_indices_without_nans(self):
        """Internal method for identifying overlapping and unique indices."""
        idxcomp = DateTimeIndexComparison(self.s1.index, self.s2.index)

        self.idx_in_both = idxcomp.idx_in_both()
        self.idx_in_s1 = idxcomp.idx_in_idx1()
        self.idx_in_s2 = idxcomp.idx_in_idx2()

        # add where s1 is NaN and s2 is missing to in_both_nan
        nan_missing = self.idx_in_s1_nan.difference(self.idx_in_s2)
        # add where s1 is NaN and s2 is missing to in_both_nan
        missing_nan = self.idx_in_s2_nan.difference(self.idx_in_s1)

        self.idx_in_both_nan = self.idx_in_both_nan.union(
            nan_missing).union(missing_nan)

    def _compare_series_values(self):
        """Internal method for identifying different identical values.

        Returns
        -------
        different_idx, identical_idx : DateTimeIndex, DateTimeIndex
            DateTimeIndexes containing index members where values are
            different and identical, respectively.
        """
        diff = self.s1.loc[self.idx_in_both] - self.s2.loc[self.idx_in_both]
        different_idx = self.idx_in_both[diff.abs() > self.diff_threshold]
        identical_idx = self.idx_in_both[diff.abs() <= self.diff_threshold]
        return different_idx, identical_idx

    def _summarize_series_comparison(self):
        """Internal method for summarizing comparison.

        Returns
        -------
        summary: pandas.Series
            Series summarizing the series comparison, containing counts
            per category
        """
        categories = ["in_both_identical",
                      "in_both_different",
                      "in_s1",
                      "in_s2",
                      "in_both_nan"]
        summary = pd.Series(index=categories, name="N_obs",
                            dtype=int)
        for cat in categories:
            summary.loc[cat] = getattr(self, "idx_" + cat).size

        return summary

    def compare_by_comment(self):
        """Compare series per comment.

        Returns
        -------
        comparison : pd.Series
            series containing the possible comparison outcomes, but split
            into categories, one for each unique comment. Comments must
            be passed via series2.

        Raises
        ------
        ValueError
            if no comment series is found
        """

        if self.c2n.empty:
            raise ValueError("No comment series!")

        categories = ["in_both_identical",
                      "in_both_different",
                      "in_s1",
                      "in_s2",
                      "in_both_nan"]

        unique_comments = self.c2n.unique()
        summary = pd.DataFrame(index=categories,
                               columns=unique_comments,
                               dtype=int)
        for cat in categories:
            cat_idx = getattr(self, "idx_" + cat)
            vc = self.c2n.reindex(cat_idx).dropna().value_counts()
            summary.loc[cat, vc.index] = vc

        return summary.sort_index(axis=1)

    def comparison_series(self):
        """Create series that indicates what happend to a value.

        Series index is the union of s1 and s2 with a value indicating
        the status of the comparison:

            - -1: value is modified
            - 0: value stays the same
            - 1: value only in series 1
            - 2: value only in series 2
            - -9999: value is NaN in both series

        Returns
        -------
        s : pd.Series
            series containing status of value from comparison
        """
        s = pd.Series(index=self.s1n.index.union(self.s2n.index), data=np.nan)
        s.loc[self.idx_in_both_identical] = -1.
        s.loc[self.idx_in_both_different] = 0.
        s.loc[self.idx_in_s1] = 1.
        s.loc[self.idx_in_s2] = 2.
        s.loc[self.idx_in_both_nan] = -9999.
        return s

    def _check_idx_comparison(self, return_missing=False):
        """Internal method for verifying comparison.

        Used for debugging during development. Ensures the counts match total
        length of union series indices. Returns False if there are any missing
        members, otherwise returns True.

        Parameters
        ----------
        return_missing : bool
            return missing members of index if there are any

        Returns
        -------
        check : bool
            True if there are no missing members, otherwise returns False
        missing : DateTimeIndex
            only returned if return_missing=True, contains missing members
            from comparison.
        """
        seriesindexunion = self.s1.index.union(self.s2.index)
        missing = (self.idx_in_both_nan
                   .union(self.idx_in_both)
                   .union(self.idx_in_s1)
                   .union(self.idx_in_s2)
                   .symmetric_difference(seriesindexunion))

        idxlen = (self.idx_in_both_nan.size + self.idx_in_both.size +
                  self.idx_in_s1.size + self.idx_in_s2.size)

        if return_missing:
            return missing
        else:
            if idxlen - seriesindexunion.size != 0:
                return False
            else:
                return True


class SeriesComparisonRelative(SeriesComparison):
    """Object for comparing two timeseries relative to a third timeseries.

    Extends the SeriesComparison object to include a comparison between
    two timeseries and a third base timeseries. This is used for example, when
    comparing the results of two error detection outcomes to the original
    raw timeseries.

    Comparison yields both the results from SeriesComparison as well as the
    following categories for the relative comparison to the base timeseries:

    -  kept_in_both: both timeseries and the base timeseries contain values
    -  flagged_in_s1: value is NaN/missing in series #1
    -  flagged_in_s2: value is NaN/missing in series #2
    -  flagged_in_both: value is NaN/missing in both series #1 and series #2
    -  in_all_nan: value is NaN in all timeseries (series #1, #2 and base)
    -  introduced_in_s1: value is NaN/missing in base but has value in series #1
    -  introduced_in_s2: value is NaN/missing in base but has value in series #2
    -  introduced_in_both: value is NaN/missing in base but has value in both
       timeseries

    Parameters
    ----------
    s1 : pd.Series or pd.DataFrame
        first series to compare
    truth : pd.Series or pd.DataFrame
        second series to compare, if a "truth" timeseries is available
        pass it as the second timeseries. Stored in object as 's2'.
    base : pd.Series or pd.DataFrame
        timeseries to compare other two series with
    diff_threshold : float, optional
        value beyond which a difference is considered significant, by
        default 0.0. Two values whose difference is smaller than threshold
        are considered identical.


    See also
    --------
    SeriesComparison : Comparison of two timeseries relative to each other
    """

    def __init__(self, s1, truth, base, diff_threshold=0.0):
        """Compare two timeseries relative to a base timeseries.

        Parameters
        ----------
        s1 : pd.Series or pd.DataFrame
            first series to compare
        truth : pd.Series or pd.DataFrame
            second series to compare, if a "truth" timeseries is available
            pass it as the second timeseries. Stored in object as 's2'.
        base : pd.Series or pd.DataFrame
            timeseries to compare other two series with
        diff_threshold : float, optional
            value beyond which a difference is considered significant, by
            default 0.0. Two values whose difference is smaller than threshold
            are considered identical.
        """

        # Do the original comparison between s1 and s2
        super().__init__(
            s1, truth, diff_threshold=diff_threshold)

        # With NaNs
        self.basen = base
        # Without NaNs
        self.base = self.basen.loc[~self.basen.isna()]

        # do comparison
        self._compare_series_to_base()

        # summarize results
        self.summary_series_comparison = self.summary
        delattr(self, "summary")
        self.summary_base_comparison = self._summarize_comparison_to_base()

        # do binary classification
        self.bc = BinaryClassifier.from_series_comparison_relative(self)

    def _compare_series_to_base(self):
        """Internal method for comparing two timseries to base timeseries."""

        # where Nans in base timeseries
        nanmask = self.basen.isna()

        # prepare some indices
        s1s2_union = self.s1.index.union(self.s2.index)
        s1s2_intersect = self.s1.index.intersection(self.s2.index)
        only_in_s1 = self.s1.index.difference(self.s2.index)
        only_in_s2 = self.s2.index.difference(self.s1.index)

        # identify the differences
        self.idx_r_flagged_in_both = self.base.index.difference(s1s2_union)
        self.idx_r_flagged_in_s1 = self.base.index.intersection(only_in_s2)
        self.idx_r_flagged_in_s2 = self.base.index.intersection(only_in_s1)
        self.idx_r_kept_in_both = self.base.index.intersection(s1s2_intersect)

        # the generally more unexpected differences
        # contains where all NaN and where s1 and s2 missing
        self.idx_r_in_all_nan = self.basen.loc[nanmask].index.difference(
            s1s2_union)
        # self.idx_r_in_all_nan = self.basen.loc[nanmask].index.intersection(
        #     self.idx_in_both_nan)  # only where all are NaN
        # counts for both NaNs and missing in base timeseries
        self.idx_r_introduced_in_s1 = (self.basen.loc[nanmask].index
                                       .intersection(only_in_s1)
                                       .union(only_in_s1.difference(
                                           self.basen.index)))
        self.idx_r_introduced_in_s2 = (self.basen.loc[nanmask].index
                                       .intersection(only_in_s2)
                                       .union(only_in_s2.difference(
                                           self.basen.index)))
        self.idx_r_introduced_in_both = (self.basen.loc[nanmask].index
                                         .intersection(s1s2_intersect)
                                         .union(s1s2_intersect.difference(
                                             self.basen.index)))

    def _summarize_comparison_to_base(self):
        """Internal method for summarizing comparison with base timeseries.

        Returns
        -------
        summary : pandas.Series
            Series summarizing the series comparison relative to base
            timeseries, containing counts per category
        """
        categories = ['kept_in_both',
                      'flagged_in_s1',
                      'flagged_in_s2',
                      'flagged_in_both',
                      'in_all_nan',
                      'introduced_in_s1',
                      'introduced_in_s2',
                      'introduced_in_both']
        summary = pd.Series(index=categories, name="N_obs",
                            dtype=int)
        for cat in categories:
            summary.loc[cat] = getattr(self, "idx_r_" + cat).size

        return summary

    def compare_to_base_by_comment(self):
        """Compare two series to base series per comment.

        Returns
        -------
        comparison : pd.Series
            Series containing the number of observations in each possible
            comparison category, but split per unique comment. Comments must
            be provided via 'truth' series (series2).

        Raises
        ------
        ValueError
            if no comment series is available.
        """

        if self.c2n.empty:
            raise ValueError("No comment series!")

        categories = ['kept_in_both',
                      'flagged_in_s1',
                      'flagged_in_s2',
                      'flagged_in_both',
                      'in_all_nan',
                      'introduced_in_s1',
                      'introduced_in_s2',
                      'introduced_in_both']

        unique_comments = self.c2n.unique()
        summary = pd.DataFrame(index=categories,
                               columns=unique_comments,
                               dtype=int)
        for cat in categories:
            cat_idx = getattr(self, "idx_r_" + cat)
            vc = self.c2n.reindex(cat_idx).dropna().value_counts()
            summary.loc[cat, vc.index] = vc

        return summary.sort_index(axis=1)

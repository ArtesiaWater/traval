import os
from operator import or_

import numpy as np
import pandas as pd

from .ts_utils import (
    CorrectionCode,
    diff_with_gap_awareness,
    get_empty_corrections_df,
    interpolate_series_to_new_index,
    mask_corrections_above_below,
    mask_corrections_above_threshold,
    mask_corrections_below_threshold,
    mask_corrections_equal_value,
    mask_corrections_no_comparison_value,
    mask_corrections_not_equal_value,
    resample_short_series_to_long_series,
    smooth_lower_bound,
    smooth_upper_bound,
    spike_finder,
)


def _ufunc_corrections(series, ufunc, threshold, mask):
    if "greater" in ufunc.__name__:
        return mask_corrections_above_threshold(series, threshold, mask)
    elif "less" in ufunc.__name__:
        return mask_corrections_below_threshold(series, threshold, mask)
    elif ufunc.__name__ == "equal":
        return mask_corrections_equal_value(series, threshold, mask)
    else:
        return mask_corrections_not_equal_value(series, threshold, mask)


def rule_funcdict(series, funcdict):
    """Detection rule, flag values with dictionary of functions.

    Use dictionary of functions to identify suspect values and set
    them to NaN.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    funcdict : dict
        dictionary with function names as keys and functions/methods as
        values. Each function is applied to each value in the time series
        using `series.apply(func)`. Suspect values are those where
        the function evaluates to True.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values (according to the provided functions)
        are set to np.nan.
    """
    for i, func in enumerate(funcdict.values()):
        if i == 0:
            mask = series.apply(func)
        else:
            mask = or_(mask, series.apply(func))
    return mask_corrections_no_comparison_value(series, mask)


def rule_max_gradient(series, max_step=0.5, max_timestep="1D"):
    """Detection rule, flag values when maximum gradient exceeded.

    Flag values when maximum gradient between two observations is exceeded.
    Use negative max_step to flag values with negative gradient.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    max_step : float, optional
        max jump between two observations within given timestep,
        by default 0.5
    timestep : str, optional
        maximum timestep to consider, by default "1D". The gradient is not
        calculated for values that lie farther apart.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    conversion = pd.Timedelta(max_timestep) / pd.Timedelta("1s")
    grad = (
        series.diff() / series.index.to_series().diff().dt.total_seconds() * conversion
    )
    if max_step > 0.0:
        mask = grad > max_step
        return mask_corrections_above_threshold(series, max_step, mask)
    else:
        mask = grad < -max_step
        return mask_corrections_below_threshold(series, max_step, mask)


def rule_hardmax(series, threshold, offset=0.0):
    """Detection rule, flag values greater than threshold value."""
    return rule_ufunc_threshold(series, (np.greater,), threshold, offset=offset)


def rule_hardmin(series, threshold, offset=0.0):
    """Detection rule, flag values lower than threshold value."""
    return rule_ufunc_threshold(series, (np.less,), threshold, offset=offset)


def rule_ufunc_threshold(series, ufunc, threshold, offset=0.0):
    """Detection rule, flag values based on operator and threshold value.

    Set values to Nan based on operator function and threshold value.
    The argument ufunc is a tuple containing an operator function
    (i.e. '>', '<', '>=', '<='). These are passed using their named
    equivalents, e.g. in numpy: np.greater, np.less, np.greater_equal,
    np.less_equal. This function essentially does the following:
    ufunc(series, threshold).

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float or pd.Series
        value or time series to compare series with
    offset : float, optional
        value that is added to the threshold, e.g. if some extra tolerance is
        allowable. Default value is 0.0.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    if isinstance(threshold, pd.Series):
        full_threshold_series = resample_short_series_to_long_series(threshold, series)
        threshold = full_threshold_series.add(offset)
        mask = ufunc(series, full_threshold_series.add(offset))
    else:
        threshold = threshold + offset
        mask = ufunc(series, threshold)
    return _ufunc_corrections(series, ufunc, threshold, mask)


def rule_diff_ufunc_threshold(series, ufunc, threshold, max_gap="7D"):
    """Detection rule, flag values based on diff, operator and threshold.

    Calculate diff of series and identify suspect observations based on
    comparison with threshold value.

    The argument ufunc is a tuple containing a function, e.g. an operator
    function (i.e. '>', '<', '>=', '<='). These are passed using their named
    equivalents, e.g. in numpy: np.greater, np.less, np.greater_equal,
    np.less_equal. This function essentially does the following:
    ufunc(series, threshold_series). The argument is passed as a tuple to
    bypass RuleSet logic.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float
        value to compare diff of time series to
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)
    mask = ufunc(diff, threshold)
    return _ufunc_corrections(series, ufunc, threshold, mask)


def rule_other_ufunc_threshold(series, other, ufunc, threshold):
    """Detection rule, flag values based on other series and threshold.

    Correct values based on comparison of another time series with a threshold value.

    The argument ufunc is a tuple containing an operator function (i.e. '>',
    '<', '>=', '<='). These are passed using their named equivalents, e.g. in
    numpy: np.greater, np.less, np.greater_equal, np.less_equal. This function
    essentially does the following: ufunc(series, threshold_series). The
    argument is passed as a tuple to bypass RuleSet logic.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified, only used
        to test if index of other overlaps
    other : pd.Series
        other time series based on which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float
        value to compare time series to

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    mask = ufunc(other, threshold)
    shared_idx = series.index.intersection(other.loc[mask].index)
    other_values = other.reindex(series.index).loc[series.index]
    return _ufunc_corrections(other_values, ufunc, threshold, shared_idx)


def rule_spike_detection(series, threshold=0.15, spike_tol=0.15, max_gap="7D"):
    """Detection rule, identify spikes in time series and set to NaN.

    Spikes are sudden jumps in the value of a time series that last 1 timestep.
    They can be both negative or positive.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    threshold : float, optional
        the minimum size of the jump to qualify as a spike, by default 0.15
    spike_tol : float, optional
        offset between value of time series before spike and after spike,
        by default 0.15. After a spike, the value of the time series is usually
        close to but not identical to the value that preceded the spike. Use
        this parameter to control how close the value has to be.
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    upspikes, downspikes = spike_finder(
        series, threshold=threshold, spike_tol=spike_tol, max_gap=max_gap
    )
    mask = upspikes.index.union(downspikes.index)
    return mask_corrections_no_comparison_value(series, mask)


def rule_offset_detection(
    series,
    threshold=0.15,
    updown_diff=0.1,
    max_gap="7D",
    search_method="time",
    return_df=False,
):
    """Detection rule, detect periods with an offset error.

    This rule looks for jumps in both positive and negative direction that
    are larger than a particular threshold. It then tries to match jumps
    in upward direction to one in downward direction of a similar size. If
    this is possible, all observations between two matching but oppposite
    jumps are set to NaN.

    Parameters
    ----------
    series : pd.Series
        time series in which to look for offset errors
    threshold : float, optional
        minimum jump to consider as offset error, by default 0.35
    updown_diff : float, optional
        the maximum difference between two opposite jumps to consider them
        matching, by default 0.1
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".
    search_method : str
        method for seeking matching opposite jumps. Options are "match" or "time".
        Method "match" looks for the jump closest in magnitude to the current jump.
        Method "time" looks for the next jump in time that meets the updown_diff
        criterium.
    return_df : bool, optional
        return the dataframe containing the potential offsets,
        by default False

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    verbose = False

    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)

    diff_up = diff.copy()
    diff_up.loc[diff < 0.0] = np.nan
    diff_down = diff.copy()
    diff_down.loc[diff > 0.0] = np.nan

    mask = diff_up >= threshold
    sd_up = diff_up.loc[mask]

    mask = diff_down <= -threshold
    sd_down = diff_down.loc[mask]

    jump_df = pd.concat([sd_up, sd_down], sort=True, axis=0)
    jump_df.sort_index(inplace=True)

    if (jump_df.index.size % 2) != 0:
        print("Warning! Uneven no. of down and up jumps found.")

    periods = []
    df = pd.DataFrame(columns=["start", "end", "dh1", "dh2", "diff"])

    j = 0
    if jump_df.index.size > 1:
        for i in jump_df.index:
            if i not in periods:
                dh = jump_df.loc[i]
                if search_method == "match":
                    idiff = jump_df.loc[jump_df.index.difference(periods + [i])] + dh
                    index_best_match = idiff.abs().idxmin()
                    if np.abs(idiff.loc[index_best_match]) <= updown_diff:
                        periods += [i, index_best_match]
                        df.loc[j] = [
                            i,
                            index_best_match,
                            dh,
                            jump_df.loc[index_best_match],
                            idiff.loc[index_best_match],
                        ]
                        j += 1
                elif search_method == "time":
                    idiff = jump_df.loc[jump_df.index.difference(periods + [i])] + dh
                    mask = np.abs(idiff) <= updown_diff
                    first = (
                        jump_df.loc[jump_df.index.difference(periods + [i])]
                        .loc[mask]
                        .iloc[0:1]
                    )
                    if first.empty:
                        if verbose:
                            print("No matching jump found.")
                        continue
                    periods += [i, first.index[0]]
                    df.loc[j] = [
                        i,
                        first.index[0],
                        dh,
                        first.iloc[0],
                        idiff.loc[first.index[0]],
                    ]
                    j += 1
    else:
        if not jump_df.empty:
            df.loc[j] = [
                jump_df.index[0],
                np.nan,
                jump_df.iloc[0],
                np.nan,
                np.nan,
            ]
            periods = [jump_df.index[0], series.index[-1]]

    # manually compute corrections dataframe
    corrections = pd.DataFrame(
        index=series.index,
        data={
            "correction_code": np.zeros(series.size, dtype=float),
            "series_values": np.full(series.size, np.nan),
            "comparison_values": np.full(series.size, np.nan),
        },
    )
    for j in range(0, len(periods), 2):
        corrections.loc[
            periods[j] : periods[j + 1] - pd.Timedelta(seconds=30), "correction_code"
        ] = 99
    if return_df:
        return corrections, df, jump_df
    else:
        return corrections


def rule_outside_n_sigma(series, n=2.0):
    """Detection rule, set values outside of n * standard deviation to NaN.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    n : float, optional
        number of standard deviations to use, by default 2

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.

    """
    threshold_above = series.mean() + n * series.std()
    mask_above = series > threshold_above
    threshold_below = series.mean() - n * series.std()
    mask_below = series < threshold_below

    return mask_corrections_above_below(
        series,
        mask_above,
        threshold_above,
        mask_below,
        threshold_below,
    )


def rule_diff_outside_of_n_sigma(series, n=2.0, max_gap="7D"):
    """Detection rule, calculate diff of series and identify suspect.

    observations based on values outside of n * standard deviation of the
    difference.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    n : float, optional
        number of standard deviations to use, by default 2
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)
    nsigma = n * diff.std()
    mask = diff.abs() > nsigma
    return mask_corrections_above_threshold(diff, nsigma, mask)


def rule_outside_bandwidth(series, lowerbound, upperbound):
    """Detection rule, set suspect values to NaN if outside bandwidth.

    Parameters
    ----------
    series : pd.Series
        time series in which suspect values are identified
    lowerbound : pd.Series
        time series containing the lower bound, if bound values are less
        frequent than series, bound is interpolated to series.index
    upperbound : pd.Series
        time series containing the upper bound, if bound values are less
        frequent than series, bound is interpolated to series.index

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    if series.index.symmetric_difference(lowerbound.index).size > 0:
        lowerbound = interpolate_series_to_new_index(lowerbound, series.index)
    if series.index.symmetric_difference(upperbound.index).size > 0:
        upperbound = interpolate_series_to_new_index(upperbound, series.index)

    mask_above = series > upperbound
    mask_below = series < lowerbound
    return mask_corrections_above_below(
        series, mask_above, upperbound, mask_below, lowerbound
    )


def rule_pastas_outside_pi(
    series,
    ml,
    ci=0.95,
    min_ci=None,
    smoothfreq=None,
    tmin=None,
    tmax=None,
    savedir=None,
    verbose=False,
):
    """Detection rule, flag values based on pastas model prediction interval.

    Flag suspect outside prediction interval calculated by pastas time series
    model. Uses a pastas.Model and a confidence interval as input.

    Parameters
    ----------
    series : pd.Series
        time series to identify suspect observations in
    ml : pastas.Model
        time series model for series
    ci : float, optional
        confidence interval for calculating bandwidth, by default 0.95.
        Higher confidence interval means that bandwidth is wider and more
        observations will fall within the bounds.
    min_ci : float, optional
        value indicating minimum distance between upper and lower
        bounds, if ci does not meet this requirement, this value is added
        to the bounds. This can be used to prevent extremely narrow prediction
        intervals. Default is None.
    smoothfreq : str, optional
        str indicating no. of periods and frequency str (i.e. "1D") for
        smoothing upper and lower bounds only used if smoothbounds=True,
        default is None.
    tmin : str or pd.Timestamp, optional
        set tmin for model simulation
    tmax : str or pd.Timestamp, optional
        set tmax for model simulation
    savedir : str, optional
        save calculated prediction interval to folder as pickle file.

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Suspect values are set to np.nan.
    """
    # no model
    if ml is None:
        if verbose:
            print("Warning: No Pastas model found!")
        corrections = get_empty_corrections_df(series)
        corrections.columns = ["sim", "series_values", "comparison_values"]
    # no solver
    elif ml.solver is None:
        if verbose:
            print("Warning: Model has no attribute solver!")
        corrections = get_empty_corrections_df(series)
        corrections.columns = ["sim", "series_values", "comparison_values"]
    # calculate pi
    else:
        if tmin is None:
            tmin = series.first_valid_index()
        if tmax is None:
            tmax = series.last_valid_index()

        # calculate prediction interval
        pi = ml.solver.prediction_interval(alpha=(1 - ci), tmin=tmin, tmax=tmax)

        # prediction interval empty
        if pi.empty:
            if verbose:
                print(
                    "Warning: calculated prediction interval with "
                    "Pastas model is empty!"
                )
            corrections = get_empty_corrections_df(series)
            corrections.columns = ["sim", "series_values", "comparison_values"]
        else:
            lower = pi.iloc[:, 0]
            upper = pi.iloc[:, 1]

            # apply time-shift smoothing
            if smoothfreq is not None:
                upper = smooth_upper_bound(upper, smoothfreq=smoothfreq)
                lower = smooth_lower_bound(lower, smoothfreq=smoothfreq)

            # apply minimum ci if passed
            if min_ci is not None:
                # check if mean of current interval is smaller than ci
                if (upper - lower).mean() < min_ci:
                    # adjust bounds with half of min_ci each
                    upper = upper + min_ci / 2.0
                    lower = lower - min_ci / 2.0

            corrections = rule_outside_bandwidth(series, lower, upper)
            corrections.columns = [
                "correction_code",
                "series_values",
                "comparison_values",
            ]
            corrections.index.name = f"sim (r^2={ml.stats.rsq():.3f})"

            if savedir:
                savedir.mkdir(exist_ok=True)
                pi.to_pickle(os.path.join(savedir, f"pi_{ml.name}.pkl"))
    return corrections


def rule_pastas_percentile_pi(
    series, ml, alpha=0.1, tmin=None, tmax=None, verbose=False
):
    # no model
    if ml is None:
        if verbose:
            print("Warning: No Pastas model found!")
        corrections = get_empty_corrections_df(series)
        corrections.columns = ["sim", "series_values", "comparison_values"]
    # no solver
    elif ml.solver is None:
        if verbose:
            print("Warning: Model has no solver attribute!")
        corrections = get_empty_corrections_df(series)
        corrections.columns = ["sim", "series_values", "comparison_values"]
    # calculate realizations

    # TODO: work in progress


def rule_keep_comments(series, keep_comments, comment_series):
    """Filter rule, modify time series to keep data with certain comments.

    This rule was invented to extract time series only containing certain
    types of errors, based on labeled data. For example, to get only erroneous
    observations caused by sensors above the groundwater level:

    -  series: the raw time series
    -  keep_comments: list of comments to keep, e.g. ['dry sensor']
    -  comment_series: time series containing the comments for erroneous obs

    Parameters
    ----------
    series : pd.Series
        time series to filter
    keep_comments : list of str
        list of comments to keep
    comment_series : pd.Series
        time series containing comments, should have same index as series

    Returns
    -------
    corrections : pd.DataFrame
        dataframe containing correction code 99 where comment is in keep_comments
        and 0 otherwise.
    """
    c = get_empty_corrections_df(series)
    c["comparison_values"] = ""
    for comment in keep_comments:
        mask = comment_series.str.contains(comment)
        c.loc[mask, "correction_code"] = CorrectionCode.UNKNOWN_COMPARISON_VALUE
        c.loc[mask, "series_values"] = series.loc[mask]
        c.loc[mask, "comparison_values"] = comment

    return c


def rule_compare_to_manual_obs(
    series, manual_obs, threshold=0.05, method="linear", max_dt="1D"
):
    # check if time between manual obs and sensor obs
    # are further apart than max_dt:
    nearest = series.index.get_indexer(manual_obs.index, method="nearest")
    mask = np.abs((series.index[nearest] - manual_obs.index).total_seconds()) <= (
        pd.Timedelta(max_dt) / pd.Timedelta("1s")
    )

    # interpolate raw obs to manual obs times
    s_obs = (
        series.reindex(series.index.join(manual_obs.index, how="outer"))
        .interpolate(method="time")
        .loc[manual_obs.index]
    )

    # calculate diff (manual - sensor, i.e. positive value means
    # manual observation is higher)
    diff = -(s_obs - manual_obs)

    # use only diff where mask is True (= time between obs < max_dt)
    diff = diff.loc[mask]

    # interpolate w/ method
    if method == "linear":
        diff_full_index = (
            diff.reindex(series.index.join(diff.index, how="outer"), method=None)
            .interpolate(method="linear")
            .fillna(0.0)
        )
    else:
        diff_full_index = diff.reindex(series.index, method=method).fillna(0.0)

    mask_above = diff_full_index.loc[series.index] > threshold
    mask_below = diff_full_index.loc[series.index] < -threshold

    return mask_corrections_above_below(
        diff_full_index.loc[series.index],
        mask_above,
        threshold,
        mask_below,
        -threshold,
    )


def rule_shift_to_manual_obs(
    series, hseries, method="linear", max_dt="1D", reset_dates=None
):
    """Adjustment rule, for shifting time series onto manual observations.

    Used for shifting time series based on sensor observations onto manual
    verification measurements. By default uses linear interpolation between
    two manual verification observations.

    Parameters
    ----------
    series : pd.Series
        time series to adjust
    hseries : pd.Series
        time series containing manual observations
    method : str, optional
        method to use for interpolating between two manual observations,
        by default "linear". Other options are those that are accepted by
        series.reindex(): 'bfill', 'ffill', 'nearest'.
    max_dt : str, optional
        maximum amount of time between manual observation and value in
        series, by default "1D"
    reset_dates : list, optional
        list of dates  (as str or pd.Timestamp) on which to reset the
        adjustments to 0.0, by default None. Useful for resetting the
        adjustments when the sensor is replaced, for example.

    Returns
    -------
    adjusted_series :  pd.Series
        time series containing adjustments to shift series onto manual
        observations.
    """
    # check if time between manual obs and sensor obs
    # are further apart than max_dt:
    nearest = series.index.get_indexer(hseries.index, method="nearest")
    mask = np.abs((series.index[nearest] - hseries.index).total_seconds()) <= (
        pd.Timedelta(max_dt) / pd.Timedelta("1s")
    )

    # interpolate raw obs to manual obs times
    s_obs = (
        series.reindex(series.index.join(hseries.index, how="outer"))
        .interpolate(method="time")
        .loc[hseries.index]
    )

    # calculate diff (manual - sensor, i.e. positive value means
    # manual observation is higher)
    diff = -(s_obs - hseries)

    # use only diff where mask is True (= time between obs < max_dt)
    diff = diff.loc[mask]

    if reset_dates is not None:
        for i in reset_dates:
            diff.loc[i] = 0.0

    # interpolate w/ method
    if method == "linear":
        diff_full_index = (
            diff.reindex(series.index.join(diff.index, how="outer"), method=None)
            .interpolate(method="linear")
            .fillna(0.0)
        )
    else:
        diff_full_index = diff.reindex(series.index, method=method).fillna(0.0)

    adjusted_series = series + diff_full_index

    return adjusted_series


def rule_combine_nan_or(*args):
    """Combination rule, combine NaN values for any number of time series.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (s1.isna() OR s2.isna())

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Contains NaNs where any of the input series
        values is NaN.
    """
    for i, series in enumerate(args):
        if i == 0:
            result = series.copy()
        else:
            result.loc[series.isna()] = np.nan
    return result


def rule_combine_corrections_or(*args):
    """Combination rule, combine corrections for any number of time series.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (corr_s1 OR corr_s2)

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Contains corrections where all of the input series
        values contain corrections.
    """
    for i, series in enumerate(args):
        if i == 0:
            c = get_empty_corrections_df(series)
        c.loc[series["correction_code"] != 0, "correction_code"] = 99
    return c


def rule_combine_nan_and(*args):
    """Combination rule, combine NaN values for any number of time series.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (s1.isna() AND s2.isna())

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Contains NaNs where any of the input series
        values is NaN.
    """
    for i, series in enumerate(args):
        if i == 0:
            mask = series.isna()
        else:
            mask = mask & series.isna()
    result = args[0].copy()
    result.loc[mask] = np.nan
    return result


def rule_combine_corrections_and(*args):
    """Combination rule, combine corrections for any number of time series.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (corr_s1 AND corr_s2)

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Contains corrections where all of the input series
        values contain corrections.
    """
    for i, series in enumerate(args):
        if i == 0:
            mask = series["correction_code"] != 0
        else:
            mask = mask & (series["correction_code"] != 0)
    c = get_empty_corrections_df(args[0])
    c.loc[mask, "correction_code"] = 99
    return c


def rule_flat_signal(
    series,
    window,
    min_obs,
    std_threshold=7.5e-3,
    qbelow=None,
    qabove=None,
    hbelow=None,
    habove=None,
):
    """Detection rule, flag values based on dead signal in rolling window.

    Flag values when variation in signal within a window falls below a
    certain threshold value. Optionally provide quantiles below or above
    which to look for dead/flat signals.

    Parameters
    ----------
    series : pd.Series
        time series to analyse
    window : int
        number of days in window
    min_obs : int
        minimum number of observations in window to calculate
        standard deviation
    std_threshold : float, optional
        standard deviation threshold value, by default 7.5e-3
    qbelow : float, optional
        quantile value between 0 and 1, signifying an upper
        limit. Only search for flat signals below this limit.
        By default None.
    qabove : float, optional
        quantile value between 0 and 1, signifying a lower
        limit. Only search for flat signals above this limit.
        By default None.
    hbelow : float, optional
        absolute value in units of time series signifying an upper limit.
        Only search for flat signals below this limit. By default None.
    habove : float, optional
        absolute value in units of time series signifying a lower limit.
        Only search for flat signals above this limit. By default None.

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input time series containing
        corrections. Contains NaNs where the signal is considered flat
        or dead.
    """
    s = series.dropna()
    stdfilt = s.dropna().rolling(f"{int(window)}D", min_periods=min_obs).std()
    stdmask = stdfilt < std_threshold

    if qabove is None and qbelow is not None:
        quantilemask = s < s.quantile(qbelow)
    elif qabove is not None and qbelow is None:
        quantilemask = s > s.quantile(qabove)
    elif qabove is not None and qbelow is not None:
        quantilemask = (s > s.quantile(qabove)) | (s < s.quantile(qbelow))
    else:
        quantilemask = pd.Series(index=s.index, data=True, dtype=bool)

    if habove is None and hbelow is not None:
        levelmask = s < hbelow
    elif habove is not None and hbelow is None:
        levelmask = s > habove
    elif habove is not None and hbelow is not None:
        levelmask = (s > habove) | (s < hbelow)
    else:
        levelmask = pd.Series(index=s.index, data=True, dtype=bool)

    mask = stdmask & quantilemask & levelmask
    mask = mask.reindex(series.index, fill_value=False)

    return mask_corrections_no_comparison_value(series, mask)

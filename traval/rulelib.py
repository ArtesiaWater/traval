import os
from operator import or_

import numpy as np
import pandas as pd

from .ts_utils import (
    diff_with_gap_awareness,
    interpolate_series_to_new_index,
    mask_corrections_as_nan,
    resample_short_series_to_long_series,
    smooth_lower_bound,
    smooth_upper_bound,
    spike_finder,
)


def rule_funcdict_to_nan(series, funcdict):
    """Detection rule, flag values with dictionary of functions.

    Use dictionary of functions to identify suspect values and set
    them to NaN.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    funcdict : dict
        dictionary with function names as keys and functions/methods as
        values. Each function is applied to each value in the timeseries
        using `series.apply(func)`. Suspect values are those where
        the function evaluates to True.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values (according to the provided functions)
        are set to np.nan.
    """
    for i, func in enumerate(funcdict.values()):
        if i == 0:
            mask = series.apply(func)
        else:
            mask = or_(mask, series.apply(func))
    return mask_corrections_as_nan(series, mask)


def rule_max_gradient(series, max_step=0.5, max_timestep="1D"):
    """Detection rule, flag values when maximum gradient exceeded.

    Set values tot NaN when maximum gradient between two
    observations is exceeded.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    max_step : float, optional
        max jump between two observations within given timestep,
        by default 0.5
    timestep : str, optional
        maximum timestep to consider, by default "1D". The gradient is not
        calculated for values that lie farther apart.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    conversion = pd.Timedelta(max_timestep) / pd.Timedelta("1S")
    grad = (
        series.diff()
        / series.index.to_series().diff().dt.total_seconds()
        * conversion
    )
    mask = grad.abs() > max_step
    return mask_corrections_as_nan(series, mask)


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
        timeseries in which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float or pd.Series
        value or timeseries to compare series with
    offset : float, optional
        value that is added to the threshold, e.g. if some extra tolerance is
        allowable. Default value is 0.0.

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    if isinstance(threshold, pd.Series):
        full_threshold_series = resample_short_series_to_long_series(
            threshold, series
        )
        mask = ufunc(series, full_threshold_series.add(offset))
    else:
        mask = ufunc(series, threshold + offset)
    return mask_corrections_as_nan(series, mask)


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
        timeseries in which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float
        value to compare diff of timeseries to
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)
    mask = ufunc(diff.abs(), threshold)
    return mask_corrections_as_nan(series, mask)


def rule_other_ufunc_threshold(series, other, ufunc, threshold):
    """Detection rule, flag values based on other series and threshold.

    Set values to Nan based on comparison of another timeseries with a
    threshold value.

    The argument ufunc is a tuple containing an operator function (i.e. '>',
    '<', '>=', '<='). These are passed using their named equivalents, e.g. in
    numpy: np.greater, np.less, np.greater_equal, np.less_equal. This function
    essentially does the following: ufunc(series, threshold_series). The
    argument is passed as a tuple to bypass RuleSet logic.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified, only used
        to test if index of other overlaps
    other : pd.Series
        other timeseries based on which suspect values are identified
    ufunc : tuple
        tuple containing ufunc (i.e. (numpy.greater_equal,) ). The function
        must be callable according to `ufunc(series, threshold)`. The function
        is passed as a tuple to bypass RuleSet logic.
    threshold : float
        value to compare timeseries to

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    ufunc = ufunc[0]
    mask = ufunc(other, threshold)
    shared_idx = series.index.intersection(other.loc[mask].index)
    return mask_corrections_as_nan(series, shared_idx)


def rule_spike_detection(series, threshold=0.15, spike_tol=0.15, max_gap="7D"):
    """Detection rule, identify spikes in timeseries and set to NaN.

    Spikes are sudden jumps in the value of a timeseries that last 1 timestep.
    They can be both negative or positive.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    threshold : float, optional
        the minimum size of the jump to qualify as a spike, by default 0.15
    spike_tol : float, optional
        offset between value of timeseries before spike and after spike,
        by default 0.15. After a spike, the value of the timeseries is usually
        close to but not identical to the value that preceded the spike. Use
        this parameter to control how close the value has to be.
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    upspikes, downspikes = spike_finder(
        series, threshold=threshold, spike_tol=spike_tol, max_gap=max_gap
    )
    mask = upspikes.index.union(downspikes.index)
    return mask_corrections_as_nan(series, mask)


def rule_offset_detection(
    series, threshold=0.15, updown_diff=0.1, max_gap="7D", return_df=False
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
        timeseries in which to look for offset errors
    threshold : float, optional
        minimum jump to consider as offset error, by default 0.35
    updown_diff : float, optional
        the maximum difference between two opposite jumps to consider them
        matching, by default 0.1
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".
    return_df : bool, optional
        return the dataframe containing the potential offsets,
        by default False

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """

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

    if (len(jump_df.index) % 2) != 0:
        print("Warning! Uneven no. of down and up jumps found.")

    periods = []
    df = pd.DataFrame(columns=["start", "end", "dh1", "dh2", "diff"])

    j = 0
    for i in jump_df.index:
        if i not in periods:
            dh = jump_df.loc[i]
            idiff = jump_df.loc[jump_df.index.difference([i])] + dh
            index_best_match = idiff.abs().idxmin()
            if idiff.loc[index_best_match] <= updown_diff:
                periods += [i, index_best_match]
                df.loc[j] = [
                    i,
                    index_best_match,
                    dh,
                    jump_df.loc[index_best_match],
                    idiff.loc[index_best_match],
                ]
                j += 1

    corrections = pd.Series(
        index=series.index, data=np.zeros(series.index.size), fastpath=True
    )
    for j in range(0, len(periods), 2):
        corrections.loc[
            periods[j] : periods[j + 1] - pd.Timedelta(seconds=30)
        ] = np.nan
    if return_df:
        return corrections, df
    else:
        return corrections


def rule_outside_n_sigma(series, n=2.0):
    """Detection rule, set values outside of n * standard deviation to NaN

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    n : float, optional
        number of standard deviations to use, by default 2

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.

    """

    mask = (series > series.mean() + n * series.std()) | (
        series < series.mean() - n * series.std()
    )
    return mask_corrections_as_nan(series, mask)


def rule_diff_outside_of_n_sigma(series, n, max_gap="7D"):
    """Detection rule, calculate diff of series and identify suspect.

    observations based on values outside of n * standard deviation of the
    difference.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    n : float, optional
        number of standard deviations to use, by default 2
    max_gap : str, optional
        only considers observations within this maximum gap
        between measurements to calculate diff, by default "7D".

    Returns
    -------
    corrections: pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """

    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)
    nsigma = n * diff.std()
    mask = (diff.abs() - diff.mean()) > nsigma
    return mask_corrections_as_nan(series, mask)


def rule_outside_bandwidth(series, lowerbound, upperbound):
    """Detection rule, set suspect values to NaN if outside bandwidth.

    Parameters
    ----------
    series : pd.Series
        timeseries in which suspect values are identified
    lowerbound : pd.Series
        timeseries containing the lower bound, if bound values are less
        frequent than series, bound is interpolated to series.index
    upperbound : pd.Series
        timeseries containing the upper bound, if bound values are less
        frequent than series, bound is interpolated to series.index

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    if series.index.symmetric_difference(lowerbound.index).size > 0:
        lowerbound = interpolate_series_to_new_index(lowerbound, series.index)
    if series.index.symmetric_difference(upperbound.index).size > 0:
        upperbound = interpolate_series_to_new_index(upperbound, series.index)

    mask = (series > upperbound) | (series < lowerbound)
    return mask_corrections_as_nan(series, mask)


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

    Flag suspect outside prediction interval calculated by pastas timeseries
    model. Uses a pastas.Model and a confidence interval as input.

    Parameters
    ----------
    series : pd.Series
        timeseries to identify suspect observations in
    ml : pastas.Model
        timeseries model for series
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
        a series with same index as the input timeseries containing
        corrections. Suspect values are set to np.nan.
    """
    # no model
    if ml is None:
        if verbose:
            print("Warning: No Pastas model found!")
        corrections = mask_corrections_as_nan(
            series, pd.Series(index=series.index, data=False)
        )
        corrections.name = "sim"
    # no fit
    elif ml.fit is None:
        if verbose:
            print("Warning: Pastas model fit attribute is None!")
        corrections = mask_corrections_as_nan(
            series, pd.Series(index=series.index, data=False)
        )
        corrections.name = "sim"
    # calculate pi
    else:
        if tmin is not None:
            ml.settings["tmin"] = tmin
        if tmax is not None:
            ml.settings["tmax"] = tmax

        # calculate prediction interval
        pi = ml.fit.prediction_interval(alpha=(1 - ci))

        # prediction interval empty
        if pi.empty:
            if verbose:
                print(
                    "Warning: calculated prediction interval with "
                    "Pastas model is empty!"
                )
            corrections = mask_corrections_as_nan(
                series, pd.Series(index=series.index, data=False)
            )
            corrections.name = "sim"
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
            corrections.name = "sim (r^2={0:.3f})".format(ml.stats.rsq())

            if savedir:
                pi.to_pickle(os.path.join(savedir, f"pi_{ml.name}.pkl"))
    return corrections


def rule_pastas_percentile_pi(
    series, ml, alpha=0.1, tmin=None, tmax=None, verbose=False
):
    # no model
    if ml is None:
        if verbose:
            print("Warning: No Pastas model found!")
        corrections = mask_corrections_as_nan(
            series, pd.Series(index=series.index, data=False)
        )
        corrections.name = "sim"
    # no fit
    elif ml.fit is None:
        if verbose:
            print("Warning: Pastas model fit attribute is None!")
        corrections = mask_corrections_as_nan(
            series, pd.Series(index=series.index, data=False)
        )
        corrections.name = "sim"
    # calculate realizations

    # TODO: work in progress


def rule_keep_comments(series, keep_comments, comment_series, other_series):
    """Filter rule, modify timeseries to keep data with certain comments.

    This rule was invented to extract timeseries only containing certain
    types of errors, based on labeled data. For example, to get only erroneous
    observations caused by sensors above the groundwater level:

    -  series: the raw timeseries
    -  keep_comments: list of comments to keep, e.g. ['dry sensor']
    -  comment_series: timeseries containing the comments for erroneous obs
    -  other_series: the validated timeseries where the commented observations
       were removed (set to NaN).

    Parameters
    ----------
    series : pd.Series
        timeseries to filter
    keep_comments : list of str
        list of comments to keep
    comment_series : pd.Series
        timeseries containing comments, should have same index as series
    other_series : pd.Series
        timeseries containing corrected/adjusted values corresponding
        to the commmented entries.

    Returns
    -------
    corrections : pd.Series
        timeseries containing NaN values where comment is in keep_comments
        and 0 otherwise.
    """
    new_series = series.copy()
    for c in keep_comments:
        mask = comment_series.str.startswith(c)
        new_series.where(mask, other=other_series, inplace=True)

    corrections = new_series - series
    corrections.name = "_".join(keep_comments)

    return corrections


def rule_shift_to_manual_obs(
    series, hseries, method="linear", max_dt="1D", reset_dates=None
):
    """Adjustment rule, for shifting timeseries onto manual observations.

    Used for shifting timeseries based on sensor observations onto manual
    verification measurements. By default uses linear interpolation between
    two manual verification observations.

    Parameters
    ----------
    series : pd.Series
        timeseries to adjust
    hseries : pd.Series
        timeseries containing manual observations
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
        timeseries containing adjustments to shift series onto manual
        observations.
    """
    # check if time between manual obs and sensor obs
    # are further apart than max_dt:
    nearest = hseries.index.map(
        lambda t: series.index.get_loc(t, method="nearest")
    )
    mask = np.abs((series.index[nearest] - hseries.index).total_seconds()) <= (
        pd.Timedelta(max_dt) / pd.Timedelta("1S")
    )

    # interpolate raw obs to manual obs times
    s_obs = (
        series.reindex(series.index.join(hseries.index, how="outer"))
        .interpolate(method="time")
        .loc[hseries.index]
    )

    # calculate diff
    diff = s_obs - hseries

    # use only diff where mask is True (= time between obs < max_dt)
    diff = diff.loc[mask]

    if reset_dates is not None:
        for i in reset_dates:
            diff.loc[i] = 0.0

    # interpolate w/ method
    if method == "linear":
        diff_full_index = (
            diff.reindex(
                series.index.join(diff.index, how="outer"), method=None
            )
            .interpolate(method="linear")
            .fillna(0.0)
        )
    else:
        diff_full_index = diff.reindex(series.index, method=method).fillna(0.0)

    adjusted_series = series - diff_full_index

    return adjusted_series


def rule_combine_nan_or(*args):
    """Combination rule, combine NaN values for any number of timeseries.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (s1.isna() OR s2.isna())

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input timeseries containing
        corrections. Contains NaNs where any of the input series
        values is NaN.
    """
    for i, series in enumerate(args):
        if i == 0:
            result = series.copy()
        else:
            result.loc[series.isna()] = np.nan
    return result


def rule_combine_nan_and(*args):
    """Combination rule, combine NaN values for any number of timeseries.

    Used for combining intermediate results in branching algorithm trees to
    create one final result, i.e. (s1.isna() AND s2.isna())

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input timeseries containing
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
        timeseries to analyse
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
        absolute value in units of timeseries signifying an upper limit.
        Only search for flat signals below this limit. By default None.
    habove : float, optional
        absolute value in units of timeseries signifying a lower limit.
        Only search for flat signals above this limit. By default None.

    Returns
    -------
    corrections : pd.Series
        a series with same index as the input timeseries containing
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

    return mask_corrections_as_nan(series, mask)

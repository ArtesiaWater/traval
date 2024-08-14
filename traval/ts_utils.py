from enum import IntEnum

import numpy as np
import pandas as pd


class CorrectionCode(IntEnum):
    """Codes and labels for labeling error detection results."""

    NO_CORRECTION = 0
    BELOW_THRESHOLD = -2
    NOT_EQUAL_VALUE = -1
    EQUAL_VALUE = 1
    ABOVE_THRESHOLD = 2
    MODIFIED_VALUE = 4
    UNKNOWN_COMPARISON_VALUE = 99


def get_empty_corrections_df(series):
    """Method to get corrections empty dataframe.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    """
    c = pd.DataFrame(
        index=series.index,
        data={
            "correction_code": CorrectionCode.NO_CORRECTION,
            "series_values": np.full(series.size, np.nan),
            "comparison_values": np.full(series.size, np.nan),
        },
    )
    return c


def _mask_corrections(series, values, mask, correction_code):
    c = get_empty_corrections_df(series)
    c.loc[mask, "series_values"] = series
    if values is not None:
        if isinstance(values, pd.Series):
            c.loc[mask, "comparison_values"] = values.loc[mask]
        else:
            c.loc[mask, "comparison_values"] = values
    c.loc[mask, "correction_code"] = correction_code
    return c


def mask_corrections_above_below(
    series,
    mask_above,
    threshold_above,
    mask_below,
    threshold_below,
):
    """Get corrections where above threshold.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    threshold_above : pd.Series
        time series with values to compare with
    mask_above : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    threshold_below : pd.Series
        time series with values to compare with
    mask_below : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    c_above = mask_corrections_above_threshold(series, threshold_above, mask_above)
    c_below = mask_corrections_below_threshold(series, threshold_below, mask_below)
    return c_above.add(c_below, fill_value=0)


def mask_corrections_above_threshold(series, threshold, mask):
    """Get corrections where below threshold.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    threshold : pd.Series
        time series with values to compare with
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(series, threshold, mask, CorrectionCode.ABOVE_THRESHOLD)


def mask_corrections_below_threshold(series, threshold, mask):
    """Get corrections where below threshold.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    threshold : pd.Series
        time series with values to compare with
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(series, threshold, mask, CorrectionCode.BELOW_THRESHOLD)


def mask_corrections_equal_value(series, values, mask):
    """Get corrections where equal to value.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    values : pd.Series
        time series with values to compare with
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(series, values, mask, CorrectionCode.EQUAL_VALUE)


def mask_corrections_modified_value(series, values, mask):
    """Get corrections where value was modified.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    values : pd.Series
        time series with values to compare with
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(series, values, mask, CorrectionCode.MODIFIED_VALUE)


def mask_corrections_not_equal_value(series, values, mask):
    """Get corrections where not equal to value.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    values : pd.Series
        time series with values to compare with
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(series, values, mask, CorrectionCode.NOT_EQUAL_VALUE)


def mask_corrections_no_comparison_value(series, mask):
    """Get corrections where equal to value.

    Parameters
    ----------
    series : pd.Series
        time series to apply corrections to
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)
    """
    return _mask_corrections(
        series, None, mask, CorrectionCode.UNKNOWN_COMPARISON_VALUE
    )


def corrections_as_nan(corrections):
    """Convert correction code series to NaNs.

    Excludes codes 0 and 4, which are used to indicate no correction and a modification
    of the value, respectively.

    Parameters
    ----------
    corrections : pd.Series or pd.DataFrame
        series or dataframe with correction code

    Returns
    -------
    c : pd.Series
        return corrections series with nans where value is corrected
    """
    if isinstance(corrections, pd.DataFrame):
        corrections = corrections["correction_code"]
    c = pd.Series(index=corrections.index, data=0.0)
    # set values where correction code is *not* 0 or 4 to NaN
    # (meaning a correction was applied)
    c.loc[(corrections != 0) | (corrections != 4)] = np.nan
    return c


def corrections_as_float(corrections):
    """Convert correction code series to NaNs.

    Excludes codes 0 and 4, which are used to indicate no correction and a modification
    of the value, respectively.

    Parameters
    ----------
    corrections : pd.DataFrame
        dataframe with correction code and original + modified values

    Returns
    -------
    c : pd.Series
        return corrections series with floats where value is modified
    """
    c = pd.Series(index=corrections.index, data=0.0)
    # set values where correction code is 4 to difference between original and modified
    mask = corrections["correction_code"] == 4
    c.loc[mask] = (
        corrections.loc[mask, "comparison_values"]
        - corrections.loc[mask, "series_values"]
    )
    return c


def resample_short_series_to_long_series(short_series, long_series):
    """Resample a short time series to index from a longer time series.

    First uses 'ffill' then 'bfill' to fill new series.

    Parameters
    ----------
    short_series : pd.Series
        short time series
    long_series : pd.Series
        long time series

    Returns
    -------
    new_series : pd.Series
        series with index from long_series and data from short_series
    """
    new_series = pd.Series(index=long_series.index, dtype=float)

    for i, idatetime in enumerate(short_series.index):
        mask = long_series.index >= idatetime
        if mask.sum() == 0:
            continue
        first_date_after = long_series.loc[mask].index[0]
        new_series.loc[first_date_after] = short_series.iloc[i]

    new_series = new_series.ffill().bfill()
    return new_series


def diff_with_gap_awareness(series, max_gap="7D"):
    """Get diff of time series with a limit on gap between two values.

    Parameters
    ----------
    series : pd.Series
        time series to calculate diff for
    max_gap : str, optional
        maximum period between two observations for calculating diff, otherwise
        set value to NaN, by default "7D"

    Returns
    -------
    diff : pd.Series
        time series with diff, with NaNs whenever two values are farther apart
        than max_gap.
    """
    diff = series.diff()
    # identify gaps and set diff value after gap to nan
    dt = series.index[1:] - series.index[:-1]
    mask = np.r_[np.array([False]), dt > pd.Timedelta(max_gap)]
    for idate in series.index[mask]:
        diff.loc[idate] = np.nan
    return diff


def spike_finder(series, threshold=0.15, spike_tol=0.15, max_gap="7D"):
    """Find spikes in time series.

    Spikes are sudden jumps in the value of a time series that last 1 timestep.
    They can be both negative or positive.

    Parameters
    ----------
    series : pd.Series
        time series to find spikes in
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
    upspikes, downspikes : pandas.DateTimeIndex
        pandas DateTimeIndex objects containing timestamps of upward and
        downward spikes.
    """
    # identify gaps and set diff value after gap to nan
    diff = diff_with_gap_awareness(series, max_gap=max_gap)

    diff_up = diff.copy()
    diff_up.loc[diff < 0.0] = np.nan
    diff_down = diff.copy()
    diff_down.loc[diff > 0.0] = np.nan

    # Find spikes:
    # find up and down spike moments and mark when change in
    # head after spike is less than spike_tol
    spike_up = (diff_up.iloc[1:-1] + diff_down.values[2:]).abs()
    spike_up.loc[spike_up > spike_tol] = np.nan
    spike_down = (diff_down.iloc[1:-1] + diff_up.values[2:]).abs()
    spike_down.loc[spike_down > spike_tol] = np.nan

    # Mask spikes to only include large ones
    # use spike moments from above and check whether
    # jump in head is larger than threshold.
    upspikes = diff.loc[spike_up.dropna().index].where(lambda s: s > threshold).dropna()
    downspikes = (
        diff.loc[spike_down.dropna().index].where(lambda s: s < -threshold).dropna()
    )
    return upspikes, downspikes


def bandwidth_moving_avg_n_sigma(series, window, n):
    """Calculate bandwidth around time series based moving average + n * std.

    Parameters
    ----------
    series : pd.Series
        series to calculate bandwidth for
    window : int
        number of observations to consider for moving average
    n : float
        number of standard deviations from moving average for bandwidth

    Returns
    -------
    bandwidth : pd.DataFrame
        dataframe with 2 columns, with lower and upper bandwidth
    """
    avg = series.rolling(window).mean()
    nstd = series.std() * n
    bandwidth = pd.DataFrame(index=series.index)
    bandwidth["lower_{}_sigma".format(n)] = avg - nstd
    bandwidth["upper_{}_sigma".format(n)] = avg + nstd
    return bandwidth


def interpolate_series_to_new_index(series, new_index):
    """Interpolate time series to new DateTimeIndex.

    Parameters
    ----------
    series : pd.Series
        original series
    new_index : DateTimeIndex
        new index to interpolate series to

    Returns
    -------
    si : pd.Series
        new series with new index, with interpolated values
    """
    # interpolate to new index
    s_interp = np.interp(
        new_index, series.index.asi8, series.values, left=np.nan, right=np.nan
    )
    si = pd.Series(index=new_index, data=s_interp, dtype=float)
    return si


def unique_nans_in_series(series, *args):
    """Get mask where NaNs in series are unique compared to other series.

    Parameters
    ----------
    series : pd.Series
        identify unique NaNs in series
    *args
        any number of pandas.Series

    Returns
    -------
    mask : pd.Series
        mask with value True where NaN is unique to series
    """
    mask = series.isna()

    for s in args:
        if not isinstance(s, pd.Series):
            raise ValueError("Only supports pandas Series")

        mask = mask & ~s.isna()

    return mask


def create_synthetic_raw_time_series(raw_series, truth_series, comments):
    """Create synthetic raw time series.

    Updates 'truth_series' (where values are labelled with a comment)
    with values from raw_series. Used for removing unlabeled changes between
    a raw and validated time series.

    Parameters
    ----------
    raw_series : pd.Series
        time series with raw data
    truth_series : pd.Series
        time series with validated data
    comments : pd.Series
        time series with comments. Index must be same as 'truth_series'.
        When value does not have a comment it must be an empty string: ''.

    Returns
    -------
    s : pd.Series
        synthetic raw time series, same as truth_series but updated with
        raw_series where value has been commented.
    """
    if truth_series.index.symmetric_difference(comments.index).size > 0:
        raise ValueError("'truth_series' and 'comments' must have same index!")

    # get intersection of index (both need to have data)
    idx_in_both = raw_series.dropna().index.intersection(truth_series.index)

    # get obs with comments
    mask_comments = comments.loc[idx_in_both] != ""

    # create synthetic raw series
    synth_raw = truth_series.loc[idx_in_both].copy()
    synth_raw.loc[mask_comments] = raw_series.loc[idx_in_both].loc[mask_comments]

    return synth_raw


def shift_series_forward_backward(s, freqstr="1D"):
    n = int(freqstr[:-1]) if freqstr[:-1].isnumeric() else 1
    freq = freqstr[-1] if freqstr[:-1].isalpha() else "D"
    shift_forward = s.shift(periods=n, freq=freq)
    shift_backward = s.shift(periods=-n, freq=freq)
    return pd.concat([shift_backward, s, shift_forward], axis=1)


def smooth_upper_bound(b, smoothfreq="1D"):
    smoother = shift_series_forward_backward(b, freqstr=smoothfreq)
    smoother.iloc[:, 0] = smoother.iloc[:, 0].interpolate(method="linear")
    smoother.iloc[:, 2] = smoother.iloc[:, 2].interpolate(method="linear")
    return smoother.max(axis=1).loc[smoother.iloc[:, 1].dropna().index]


def smooth_lower_bound(b, smoothfreq="1D"):
    smoother = shift_series_forward_backward(b, freqstr=smoothfreq)
    smoother.iloc[:, 0] = smoother.iloc[:, 0].interpolate(method="linear")
    smoother.iloc[:, 2] = smoother.iloc[:, 2].interpolate(method="linear")
    return smoother.min(axis=1).loc[smoother.iloc[:, 1].dropna().index]


def get_correction_status_name(corrections):
    """Get correction status name from correction codes.

    Parameters
    ----------
    correction_code : pd.DataFrame or pd.Series
        dataframe or series containing corrections codes

    Returns
    -------
    pd.DataFrame or pd.Series
        dataframe or series filled with correction status name
    """
    return corrections.fillna(0).map(lambda c: CorrectionCode(c).name)

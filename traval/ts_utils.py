import numpy as np
import pandas as pd


def mask_corrections_as_nan(series, mask):
    """Get corrections series with NaNs where mask == True.

    Parameters
    ----------
    series : pd.Series
        timeseries to provide corrections for
    mask : DateTimeIndex or boolean np.array
        DateTimeIndex containing timestamps where value should be set to NaN,
        or boolean array with same length as series set to True where
        value should be set to NaN. (Uses pandas .loc[mask] to set values.)

    Returns
    -------
    c : pd.Series
        return corrections series
    """
    c = pd.Series(
        index=series.index,
        data=np.zeros(series.index.size),
        fastpath=True,
        dtype=float,
    )
    c.loc[mask] = np.nan
    return c


def resample_short_series_to_long_series(short_series, long_series):
    """Resample a short timeseries to index from a longer timeseries.

    First uses 'ffill' then 'bfill' to fill new series.

    Parameters
    ----------
    short_series : pd.Series
        short timeseries
    long_series : pd.Series
        long timeseries

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

    new_series = new_series.fillna(method="ffill").fillna(method="bfill")
    return new_series


def diff_with_gap_awareness(series, max_gap="7D"):
    """Get diff of timeseries with a limit on gap between two values.

    Parameters
    ----------
    series : pd.Series
        timeseries to calculate diff for
    max_gap : str, optional
        maximum period between two observations for calculating diff, otherwise
        set value to NaN, by default "7D"

    Returns
    -------
    diff : pd.Series
        timeseries with diff, with NaNs whenever two values are farther apart
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
    """Find spikes in timeseries.

    Spikes are sudden jumps in the value of a timeseries that last 1 timestep.
    They can be both negative or positive.

    Parameters
    ----------
    series : pd.Series
        timeseries to find spikes in
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
    upspikes = (
        diff.loc[spike_up.dropna().index]
        .where(lambda s: s > threshold)
        .dropna()
    )
    downspikes = (
        diff.loc[spike_down.dropna().index]
        .where(lambda s: s < -threshold)
        .dropna()
    )
    return upspikes, downspikes


def bandwidth_moving_avg_n_sigma(series, window, n):
    """Calculate bandwidth around timeseries based moving average + n * std.

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
    """Interpolate timeseries to new DateTimeIndex.

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
    si = pd.Series(index=new_index, data=s_interp, dtype=float, fastpath=True)
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


def create_synthetic_raw_timeseries(raw_series, truth_series, comments):
    """Create synthetic raw timeseries.

    Updates 'truth_series' (where values are labelled with a comment)
    with values from raw_series. Used for removing unlabeled changes between
    a raw and validated timeseries.

    Parameters
    ----------
    raw_series : pd.Series
        timeseries with raw data
    truth_series : pd.Series
        timeseries with validated data
    comments : pd.Series
        timeseries with comments. Index must be same as 'truth_series'.
        When value does not have a comment it must be an empty string: ''.

    Returns
    -------
    s : pd.Series
        synthetic raw timeseries, same as truth_series but updated with
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
    synth_raw.loc[mask_comments] = raw_series.loc[idx_in_both].loc[
        mask_comments
    ]

    return synth_raw


def shift_series_forward_backward(s, freqstr="1D"):
    n = int(freqstr[:-1]) if freqstr[:-1].isnumeric() else 1
    freq = freqstr[-1] if freqstr[:-1].isalpha() else "D"
    shift_forward = s.shift(periods=n, freq=freq)
    shift_backward = s.shift(periods=-n, freq=freq)
    return pd.concat([shift_backward, s, shift_forward], axis=1)


def smooth_upper_bound(b, smoothfreq="1D"):
    smoother = shift_series_forward_backward(b, freqstr=smoothfreq)
    return smoother.max(axis=1)


def smooth_lower_bound(b, smoothfreq="1D"):
    smoother = shift_series_forward_backward(b, freqstr=smoothfreq)
    return smoother.min(axis=1)

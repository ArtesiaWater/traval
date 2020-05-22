import numpy as np
import pandas as pd


def mask_corrections_as_nan(series, mask):
    """internal method, get corrections series with NaNs where mask == True

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
    c = pd.Series(index=series.index, data=np.zeros(series.index.size),
                  fastpath=True, dtype=float)
    c.loc[mask] = np.nan
    return c


def resample_short_series_to_long_series(short_series, long_series):
    new_series = pd.Series(index=long_series.index, dtype=float)

    for i, idatetime in enumerate(short_series.index):
        mask = long_series.index >= idatetime
        if mask.sum() == 0:
            continue
        first_date_after = long_series.loc[mask].index[0]
        new_series.loc[first_date_after] = short_series.iloc[i]

    new_series = (new_series
                  .fillna(method="ffill")
                  .fillna(method="bfill"))
    return new_series


def diff_with_gap_awareness(series, max_gap="7D"):
    diff = series.diff()
    # identify gaps and set diff value after gap to nan
    dt = series.index[1:] - series.index[:-1]
    mask = np.r_[np.array([False]), dt > pd.Timedelta(max_gap)]
    for idate in series.index[mask]:
        diff.loc[idate] = np.nan
    return diff


def spike_finder(series, threshold=0.15, spike_tol=0.15, max_gap="7D"):

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
    upspikes = diff.loc[spike_up.dropna().index].where(
        lambda s: s > threshold).dropna()
    downspikes = diff.loc[spike_down.dropna().index].where(
        lambda s: s < -threshold).dropna()
    return upspikes, downspikes


def bandwidth_moving_avg_n_sigma(series, window, n):
    """Calculate bandwidth around timeseries based moving average and 
    n * standard deviation.

    Parameters
    ----------
    series : pd.Series
        [description]
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
    """Interpolate timeseries to new DateTimeIndex

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
    s_interp = np.interp(new_index, series.index.asi8, series.values)
    si = pd.Series(index=new_index, data=s_interp,
                   dtype=float, fastpath=True)
    return si

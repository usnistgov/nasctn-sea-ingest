import pandas as pd
from frozendict import frozendict
import numpy as np


def _iso_to_datetime(s, tz=None):
    """returns a timezone aware datetime from a metadata timestamp string"""

    if tz is None:
        return pd.Timestamp(np.datetime64(s[:-1])).tz_localize("utc")
    else:
        return pd.Timestamp(np.datetime64(s[:-1])).tz_localize("utc").tz_convert(tz)


class _CachedDataFrameInitializer:
    """introspection on dtypes slows down new pandas DataFrame objects.

    cache them for repeated dataframe generation.
    """

    def __init__(self):
        self.cache = {}

    def DataFrame(self, values, cache_key, index=None, columns=None, dtype=None, **kws):
        dtype = self.cache.get((cache_key, "dtype"), None)

        # if columns is None:
        #     columns = self.cache.get((cache_key, 'columns'), None)

        df = pd.DataFrame(values, index=index, columns=columns, dtype=dtype, **kws)

        self.cache[(cache_key, "dtype")] = df.dtypes.values
        # self.cache[(cache_key, 'columns')] = df.columns

        return df


cpd = _CachedDataFrameInitializer()


def localize_timestamps(dfs, tz=None):
    if tz is None:
        tz = dfs["sensor_metadata"]["timezone"]

    for key, df in dict(dfs).items():
        if isinstance(df, pd.DataFrame) and "datetime" in df.index.names:
            # first, the row timestamps
            index = df.index

            i = index.names.index("datetime")

            levels = list(index.levels)
            levels[i] = levels[i].tz_convert(tz)

            index = pd.MultiIndex(
                codes=index.codes, levels=levels, names=df.index.names, sortorder=0
            )

            df = pd.DataFrame(df.values, index=index, columns=df.columns)

        if isinstance(df, pd.DataFrame) and "metadata" in key:
            # update metadata timestamps
            dtypes = df.dtypes
            new_dtypes = {}
            for col in df.columns:
                if isinstance(dtypes[col], pd.core.dtypes.dtypes.DatetimeTZDtype):
                    new_dtypes[col] = pd.core.dtypes.dtypes.DatetimeTZDtype("ns", tz)
            df = df.astype(new_dtypes)

        dfs[key] = df


def _flatten_dict(d) -> frozendict:
    ret = {}
    for k, v in dict(d).items():
        if isinstance(v, (dict, frozendict)):
            ret.update({k + "_" + ksub: vsub for ksub, vsub in v.items()})
        else:
            ret[k] = v

    return frozendict(ret)


def trace(dfs: dict, type: str = None, *columns: str, **inds) -> pd.DataFrame:
    """indexing shortcut for dictionaries of SEA pd.DataFrame data tables.

    Args:
        dfs: a dictionary of pandas DataFrame objects
        type: table name key (e.g., 'pfp', 'psd', 'channel_metadata', etc.)
        columns: if specified, the sequence of columns to select (otherwise all)
        inds: the index value to select, keyed on the label name
    """
    if isinstance(dfs, dict):
        if type is None:
            raise ValueError(
                'when "dfs" is a dictionary of dataframes, must pass a string key to select a dataframe'
            )
        ret = dfs[type]
    else:
        ret = dfs

    if len(inds) > 0:
        ret = ret.xs(
            key=tuple(inds.values()), level=tuple(inds.keys()), drop_level=True
        )

    if len(columns) > 1:
        ret = ret[list(columns)]
    if len(columns) == 1:
        # this is actually a bit dicey, as it forces returning a series.
        # you might deliberately want to pass in a length-1 list to force
        # dataframe?
        ret = ret[columns[0]]

    return ret


def _cartesian_multiindex(i1: pd.MultiIndex, i2: pd.MultiIndex) -> pd.MultiIndex:
    """combine two MultiIndex objects as a cartesian product.

    The result is equivalent to the index of a DataFrame produced by concatenating
    a DataFrame with index i2 at each row in i1, but this is faster

    """
    return pd.MultiIndex(
        codes=(
            np.repeat(i1.codes, len(i2), axis=1).tolist()
            + np.tile(i2.codes, len(i1)).tolist()
        ),
        levels=i1.levels + i2.levels,
        names=i1.names + i2.names,
        verify_integrity=False
        # dtype=list(i1.dtypes.values)+list(i2.dtypes.values)
    )

import sys
import typing
import zipfile  # from zipfile import ZipFile, ZipInfo
from functools import reduce
from operator import add
from pathlib import Path
from traceback import format_tb

import dask
import msgspec
import numpy as np
import pandas as pd
from dask import dataframe
from eliot import log_call, log_message, start_action
from frozendict import frozendict
from natsort import natsorted

from .dask_ops import multiindex_to_index
from .seamf import _iso_to_datetime, localize_timestamps, read_seamf, read_seamf_meta


class QuackZipInfo(msgspec.Struct):
    """duck-typed ZipInfo clone for fast IPC"""

    orig_filename: str
    filename: str
    date_time: tuple
    compress_type: int
    _compresslevel: typing.Union[int, None]
    comment: bytes
    extra: bytes
    create_system: int
    create_version: int
    extract_version: int
    reserved: int
    flag_bits: int
    volume: int
    internal_attr: int
    external_attr: int
    header_offset: int
    CRC: int
    compress_size: int
    file_size: int
    _raw_time: int

    __repr__ = zipfile.ZipInfo.__repr__
    FileHeader = zipfile.ZipInfo.FileHeader
    _encodeFilenameFlags = zipfile.ZipInfo._encodeFilenameFlags
    _decodeExtra = zipfile.ZipInfo._decodeExtra
    is_dir = zipfile.ZipInfo.is_dir

    @classmethod
    def from_zipinfo(cls, zinfo):
        kws = {k: getattr(zinfo, k) for k in zinfo.__slots__ if k != "_end_offset"}
        return cls(**kws)

    @classmethod
    def from_filelist(cls, filelist):
        return [cls.from_zipinfo(zinfo) for zinfo in filelist]


class ZipFileCache(msgspec.Struct):
    filename: str
    _comment: bytes
    start_dir: int
    filelist: typing.List[QuackZipInfo]
    NameToInfo: typing.Dict[str, QuackZipInfo]


class CachedZipFile(zipfile.ZipFile):
    _encoder = msgspec.msgpack.Encoder()
    _decoder = msgspec.msgpack.Decoder(type=ZipFileCache)

    def __init__(self, cache, *args, **kws):
        self.cache = cache
        super().__init__(*args, **kws)

    def _RealGetContents(self):
        """monkeypatched: not so 'real' any more :)"""
        cache = self._decoder.decode(self.cache)
        for k in cache.__annotations__.keys():
            v = getattr(cache, k)
            setattr(self, k, v)

    def getinfo(self, name):
        # may be specific to python>=3.11
        return self.NameToInfo[name]


class MultiProcessingZipFile:
    """Serializable ZipFile wrapper for fast IPC.

    It remains closed unless a context is opened. While open, the
    object is not serializable.
    """

    def __init__(self, *args, **kws):
        self._zfile = None

        self.cache = kws.pop("cache", None)

        if self.cache is None:
            # do a one-time cache using the file scan done by ZipFile .__init__
            zfile = zipfile.ZipFile(*args, **kws)
            filelist = QuackZipInfo.from_filelist(zfile.filelist)
            NameToInfo = dict(zip(zfile.NameToInfo.keys(), filelist))

            self.cache = CachedZipFile._encoder.encode(
                dict(
                    filename=zfile.filename,
                    _comment=zfile._comment,
                    start_dir=zfile.start_dir,
                    filelist=filelist,
                    NameToInfo=NameToInfo,
                )
            )

            self._names = [f.filename for f in filelist]
            zfile.close()

        # reuse these on entry
        self.args = tuple(args)
        self.kws = dict(kws)

    def namelist(self, sort=True):
        if self._zfile is None:
            raise IOError("need to open context first")
        ret = self._names
        if sort:
            ret = natsorted(ret)
        return ret

    @classmethod
    def from_zipfile(cls, zfile: zipfile.ZipFile):
        kws = dict(
            file=zfile.filename,
            mode=zfile.mode,
            compression=zfile.compression,
            compresslevel=zfile.compresslevel,
            strict_timestamps=zfile._strict_timestamps,
            allowZip64=zfile._allowZip64,
        )
        return cls(**kws)

    @classmethod
    def from_mpzipfile(cls, zfile):
        return cls(cache=zfile.cache, *zfile.args, **zfile.kws)

    def open(self, fn):
        if self._zfile is None:
            raise IOError("open a zipfile context first")

        return self._zfile.open(fn)

    def __enter__(self):
        self._zfile = CachedZipFile(self.cache, *self.args, **self.kws)
        return self

    def __exit__(self, type, value, traceback):
        self._zfile, _ = None, self._zfile.__exit__(type, value, traceback)


def concat_dicts(*dicts):
    """concats dictionaries of dataframes by key"""
    if len(dicts) == 1:
        return dicts[0]
    elif len(dicts) == 0:
        return None

    ret = {}

    for k in dicts[0].keys():
        if isinstance(dicts[0][k], pd.DataFrame):
            ret[k] = pd.concat([d[k] for d in dicts])
        elif isinstance(dicts[0][k], (dict, frozendict)):
            ret[k] = dicts[0][k]

    return ret


@log_call(include_result=False)
def read_seamf_zipfile_as_delayed(
    data_path,
    partition_func: typing.Callable = None,
    limit_count: int = None,
    partition_size: int = 40,
    dataframe_info: bool = False,
    tz=None,
    localize=False,
) -> typing.List[dask.delayed]:
    """scan the zip file archive(s) at `data_path` and return a list of dask.delayed objects.

    Each object corresponds to a data partition comprising `partition_size` files.

    Calling `dask.compute()` directly on the list of returned objects will load the underlying data.
    In this case, the return value would be a dictionary of dataframes for each partition, containing
    the data loaded from the partition's files.

    If specified, `partition_func` should accept a dictionary of pandas.DataFrame objects as its only
    argument, and return an adjusted dictionary. It may adjust the passed dictionary in-place, though
    only its returned value is used.

    Arguments:
        data_path: path to a zipfile to load, or a tuple or list of zipfiles to aggregate
        partition_func: callable used to process partition data returned by `read_seamf_zipfile`
        partition_size: the number of SEAMF files to pack in each partition
        limit_count: `None` to load all files, otherwise a limit to the number of files to load

    """
    if isinstance(data_path, (list, tuple)):
        # support aggregating across zipfiles
        kws = locals()
        kws.pop(data_path)
        rets = [read_seamf_zipfile_as_delayed(p, **kws) for p in data_path]

        if dataframe_info:
            # merge the dataframe info returned by each call
            partition_data = []
            df_info = {}

            for this_data, this_df_info in rets:
                partition_data += this_data

                for k, v in this_df_info.items():
                    df_info.setdefault(k, {}).setdefault("meta", []).extend(v["meta"])
                    df_info[k].setdefault("divisions", []).extend(v["divisions"])

            return partition_data, df_info
        else:
            return reduce(add, rets)

    # the first open is expensive, because it involves enumerating all of the files in the zip archive.
    # we do that once for the scan; everything else is cached
    zfile = MultiProcessingZipFile(data_path)
    with zfile:
        filelist = [n for n in zfile.namelist() if n.endswith(".sigmf")][:limit_count]
    file_blocks = np.split(filelist, range(0, len(filelist), partition_size))[1:]

    @dask.delayed
    def read_partition(files: list, errors="log"):
        ret = read_seamf_zipfile(zfile, tz=tz, allow=files, errors=errors)
        if localize:
            localize_timestamps(ret)
        if partition_func is not None and ret is not None:
            ret = partition_func(ret)
            if not isinstance(ret, dict):
                raise ValueError("partition_func must return a dict")
        return ret

    # generate the tuple of delayed objects for reading each partition
    partition_data = tuple(read_partition(block) for block in file_blocks)

    if dataframe_info:
        # first: load the first file in order to sketch out the dask dataframe structure
        last_ex = None
        for filename in file_blocks[0]:
            try:
                stub_data = read_partition([filename], errors="raise").compute(
                    scheduler="synchronous"
                )
            except BaseException as ex:
                last_ex = ex
                continue
            else:
                if stub_data is not None:
                    break
        else:
            raise ValueError(
                "couldn't read any files from the first partition"
            ) from last_ex

        # these work as a-priori column stubs that dask DataFrames use to speed up concat operations
        meta_map = {
            name: multiindex_to_index(df.iloc[:0])
            for name, df in stub_data.items()
            if isinstance(df, pd.DataFrame)
        }

        # the index value boundaries between data file partitions
        divisions = _read_seamf_zipfile_divisions(
            zfile, partition_size, filelist, tz=tz
        )

        df_info = {
            k: dict(meta=meta_map[k], divisions=divisions) for k in meta_map.keys()
        }

        return partition_data, df_info
    else:
        return partition_data


def read_seamf_zipfile_as_ddf(
    data_path,
    partition_func=None,
    limit_count: int = None,
    partition_size=200,
    tz=None,
    localize=False,
) -> typing.Dict[str, dask.dataframe.DataFrame]:
    """scans the file(s) specified by data_path, returning a dictionary of dask DataFrame objects for setting up operations.

    The returned dask dataframes are repartitioned to 1-day blocks.

    Args:
        see `zipfile_delayed`
    """

    # passthrough first before other variables enter the local namespace
    partition_data, df_info = read_seamf_zipfile_as_delayed(
        **locals(), dataframe_info=True
    )

    @dask.delayed
    def select_data_product(partition_data: dict, key: str) -> pd.DataFrame:
        # select the data product, and make sure that it has a single-level index,
        # since dask dataframes do not support multiindexing
        return multiindex_to_index(partition_data[key])

    ddfs = {}
    for key in df_info.keys():  # 'sweep_metadata',
        if key == "sweep_metadata":
            # TODO: achieve timestamp indexing in seamf.py so we can get rid of this
            # hard-coded special case
            continue

        # make delayed objects for this specific key
        delayed_list = [
            select_data_product(d, key) for d in partition_data if d is not None
        ]

        ddfs[key] = dataframe.from_delayed(delayed_list, **df_info[key]).repartition(
            freq="1D"
        )

    return ddfs


def read_seamf_zipfile(
    zipfile_or_path, allow: list = None, errors="raise", tz=None, localize=False
) -> typing.Dict[str, pd.DataFrame]:
    """reads SEA-SigMF sensor data file(s) from the specified archive.

    Args:
        allow: `None` to read all files, an integer to read a fixed number of sweeps, or an explicit list of file names
        errors: 'raise' to raise exceptions, or 'log' to swallow each exception in `read_seamf` and write to log
        tz: timezone e.g. "America/New_York", which needs to be specified for metadata v3 and older

    Returns:
        a dictionary of pandas dataframes, keyed by name of the data product or metadata
    """

    kws = locals()

    def single_read(zfile: MultiProcessingZipFile, fn):
        try:
            with zfile.open(fn) as fd:
                return read_seamf(fd, tz=tz)

        except BaseException as ex:
            if errors == "raise":
                raise
            elif errors == "log":
                return sys.exc_info()

    if isinstance(zipfile_or_path, zipfile.ZipFile):
        # TODO: this will probably fail if zipfile_or_path was opened as anything
        # besides a path string
        zfile = MultiProcessingZipFile.from_zipfile(zipfile_or_path)
    elif isinstance(zipfile_or_path, (str, Path)):
        zfile = MultiProcessingZipFile(zipfile_or_path)
    elif isinstance(zipfile_or_path, MultiProcessingZipFile):
        zfile = MultiProcessingZipFile.from_mpzipfile(zipfile_or_path)
    else:
        raise TypeError(
            f'unsupported type {type(zipfile_or_path)} in argument "zipfile_or_path"'
        )

    if errors not in ("raise", "log"):
        raise ValueError('errors argument must be one of "raise" or "log"')

    with zfile:
        if allow is None:
            allow = [n for n in zfile.namelist() if n.endswith(".sigmf")]
        elif isinstance(allow, int):
            allow = [n for n in zfile.namelist() if n.endswith(".sigmf")][1 : 1 + allow]

        log_info = dict(
            name=zfile._zfile.filename,
            file_first=allow[0] if len(allow) > 0 else None,
            file_last=allow[-1] if len(allow) > 0 else None,
            file_count=len(allow),
        )

        with start_action(action_type="read_seamf_zipfile", **log_info):
            ret = [single_read(zfile, filename) for filename in allow]

            if errors == "log":
                ret = [r for r in ret if isinstance(r, dict)]
                exceptions = [
                    (r, fn) for r, fn in zip(ret, allow) if isinstance(r, tuple)
                ]

                if len(ret) == 0:
                    raise ValueError("no valid data in the partition")

                for (extype, exc, tb), fn in exceptions:
                    log_message(
                        "error",
                        filename=fn,
                        exception=extype.__name__,
                        exception_str=", ".join(exc.args),
                        traceback=format_tb(tb),
                    )

    ret = concat_dicts(*ret)

    if localize:
        localize_timestamps(ret)

    return ret


def restore_multiindex(dfs: typing.Dict[str, pd.DataFrame]):
    for name in dfs.keys():
        if isinstance(dfs[name].columns.values[-1], str):
            if dfs[name].columns[0] == "frequency":
                dfs[name].set_index("frequency", append=True, inplace=True)
            continue

        ind_names = [n for n in dfs[name].columns if isinstance(n, str)]
        dfs[name].set_index(ind_names, append=True, inplace=True)


def _read_seamf_zipfile_divisions(
    zipfile_or_path, partition_size: int, file_list: list, tz=None
) -> list:
    """reads SEA-SigMF sensor data file(s) from the specified archive

    If allow is None, then all files are read. For other arguments, see `read_sea_sigmf`.
    """

    def single_read(zfile, fn):
        with zfile.open(fn) as fd:
            return read_seamf_meta(fd, tz=tz)

    if isinstance(zipfile_or_path, zipfile.ZipFile):
        # TODO: this will probably fail if zipfile_or_path was opened as anything
        # besides a path string
        zfile = MultiProcessingZipFile.from_zipfile(zipfile_or_path)
    elif isinstance(zipfile_or_path, (str, Path)):
        zfile = MultiProcessingZipFile(zipfile_or_path)
    elif isinstance(zipfile_or_path, MultiProcessingZipFile):
        zfile = MultiProcessingZipFile.from_mpzipfile(zipfile_or_path)
    else:
        raise TypeError(
            f'unsupported type {type(zipfile_or_path)} in argument "zipfile_or_path"'
        )

    with zfile:
        division_list = file_list[::partition_size]
        # if len(division_list) % partition_size != 0:
        #     division_list.append(file_list[-1])

        dicts = [single_read(zfile, filename) for filename in division_list]

    starts = [_iso_to_datetime(d.captures[0]["core:datetime"]) for d in dicts]

    starts.append(_iso_to_datetime(dicts[-1].captures[-1]["core:datetime"]))

    return starts

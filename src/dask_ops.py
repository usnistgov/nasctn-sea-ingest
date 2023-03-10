import dask
from ziparchive import read_seamf_zipfile, MultiProcessingZipFile, _read_seamf_zipfile_divisions
import numpy as np
from dask import dataframe
import eliot
from functools import reduce
from operator import add

from multiprocessing.queues import Queue, Empty
import multiprocessing
import threading
import pandas as pd
import typing

class QueuedLogWriter(Queue):
    """ multiprocessing queue duck-typed as a stream object for distributed logging """
    def __init__(self, filename: str, file_mode = 'ab'):
        self._run = True

        if not file_mode.endswith('b'):
            raise ValueError('open file in binary mode ("wb" or "ab")')
        
        self._file_args = filename, file_mode
        super().__init__(ctx=multiprocessing.get_context())
        threading.Thread(target=self.poll_to_file, name='monitor').start() 

    def write(self, item):
        """ this is what's called in other processes """
        self.put(item)

    def flush(self):
        pass

    def close(self):
        self._run = False

    def poll_to_file(self):
        """ this runs in the parent/main process, which polls """
        with open(*self._file_args) as fd:
            while self._run:
                try:
                    fd.write(self.get(timeout=0.5))
                except Empty:
                    pass

    def __del__(self):
        self.close()

def log_to_json(file_name_or_queue, mode='wb'):
    """ set up logging via eliot that queued from worker processes """
    if not isinstance(file_name_or_queue, QueuedLogWriter):
        queue = QueuedLogWriter(file_name_or_queue, mode)

    eliot.to_file(queue)

def multiindex_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """ shift index levels into columns, leaving only the first.
     
    This is necessary when packing a dask.dataframe, which doesn't yet support multilevel indexing.
    """
    return df.reset_index(df.index.names[1:])

def trace(dfs: dict, type: str, *columns: str, **inds) -> pd.DataFrame:
    """ indexing shortcut for dictionaries of SEA pd.DataFrame data tables in  .

    Args:
        dfs: a dictionary of pandas DataFrame objects
        type: table name key (e.g., 'pfp', 'psd', 'channel_metadata', etc.)
        columns: if specified, the sequence of columns to select (otherwise all)
        inds: the index value to select, keyed on the label name
    """
    ret = dfs[type]

    if len(inds) > 0:
        ret = ret.xs(
            key=tuple(inds.values()),
            level=tuple(inds.keys()),
            drop_level=True
        )

    if len(columns) > 1:
        ret = ret[list(columns)]
    if len(columns) == 1:
        # this is actually a bit dicey, as it forces returning a series.
        # you might deliberately want to pass in a length-1 list to force
        # dataframe?
        ret = ret[columns[0]]

    return ret

@eliot.log_call(include_result=False)
def zipfile_delayed(data_path, partition_func: typing.Callable=None, limit_count: int=None, partition_size:int=40, dataframe_info:bool=False) -> typing.List[dask.delayed]:
    """ scan the zip file archive(s) at `data_path` and return a list of dask.delayed objects.

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
        rets = [zipfile_delayed(p, **kws) for p in data_path]

        if dataframe_info:
            # merge the dataframe info returned by each call
            partition_data = []
            df_info = {}

            for this_data, this_df_info in rets:
                partition_data += this_data

                for k,v in this_df_info.items():
                    df_info.setdefault(k, {}).setdefault('meta', []).extend(v['meta'])
                    df_info[k].setdefault('divisions', []).extend(v['divisions'])

            return partition_data, df_info
        else:
            return reduce(add, rets)

    # the first open is expensive, because it involves enumerating all of the files in the zip archive.
    # we do that once for the scan; everything else is cached
    zfile = MultiProcessingZipFile(data_path)
    with zfile:
        filelist = [n for n in zfile.namelist() if n.endswith('.sigmf')][:limit_count]
    file_blocks = np.split(filelist, range(0,len(filelist), partition_size))[1:]

    @dask.delayed
    def read_partition(files: list, errors='log'):
        ret = read_seamf_zipfile(zfile, allowlist=files, errors=errors)
        if partition_func is not None and ret is not None:
            ret = partition_func(ret)
            if not isinstance(ret, dict):
                raise ValueError('partition_func must return a dict')
        return ret

    # generate the tuple of delayed objects for reading each partition
    partition_data = tuple(read_partition(block) for block in file_blocks)

    if dataframe_info:
        # first: load the first file in order to sketch out the dask dataframe structure
        last_ex = None
        for filename in file_blocks[0]:
            try:
                stub_data = read_partition([filename], errors='raise').compute(scheduler='synchronous')
            except BaseException as ex:
                last_ex = ex
                continue
            else:
                if stub_data is not None:
                    break
        else:
            raise ValueError("couldn't read any files from the first partition") from last_ex

        # these work as a-priori column stubs that dask DataFrames use to speed up concat operations
        meta_map = {
            name: multiindex_to_index(df.iloc[:0])
            for name, df in stub_data.items()
            if isinstance(df, pd.DataFrame)
        }

        # the index value boundaries between data file partitions
        divisions = _read_seamf_zipfile_divisions(zfile, partition_size, filelist)

        df_info = {
            k: dict(meta=meta_map[k], divisions=divisions)
            for k in meta_map.keys()
        }

        return partition_data, df_info
    else:
        return partition_data

debug = {}

def zipfile_dask_dfs(data_path, partition_func=None, limit_count: int=None, partition_size=40) -> typing.Dict[str, dask.dataframe.DataFrame]:
    """ scans the file(s) specified by data_path, returning a dictionary of dask DataFrame objects for setting up operations.

    The returned dask dataframes are repartitioned to one day.

    Args:
        see `zipfile_delayed`
    """

    # passthrough first before other variables enter the local namespace 
    partition_data, df_info = zipfile_delayed(**locals(), dataframe_info=True)

    @dask.delayed
    def select_data_product(partition_data: dict, key: str) -> pd.DataFrame:
        # select the data product, and make sure that it has a single-level index,
        # since dask dataframes do not support multiindexing
        return multiindex_to_index(partition_data[key])

    ddfs = {}
    for key in df_info.keys(): # 'sweep_metadata', 
        if key == 'sweep_metadata':
            # TODO: achieve timestamp indexing in seamf.py so we can get rid of this
            # hard-coded special case
            continue

        # make delayed objects for this specific key 
        delayed_list = [
            select_data_product(d, key)
            for d in partition_data
            if d is not None
        ]

        ddfs[key] = (
            dataframe
            .from_delayed(delayed_list, **df_info[key])
            .repartition(freq='1D')
        )

    return ddfs

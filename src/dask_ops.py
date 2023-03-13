import dask
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
    """ a multiprocessing queue duck-typed as a stream adapter """
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


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
    """
    A multiprocessing queue duck-typed as a stream adapter.

    Args:
        filename (str): The name of the file to write to.
        file_mode (str): The mode in which the file is opened. Must be a binary mode ("wb" or "ab").

    Raises:
        ValueError: If the file_mode is not in binary mode.
    """

    def __init__(self, filename: str, file_mode="ab"):
        """
        Initializes the QueuedLogWriter.

        Args:
            filename (str): The name of the file to write to.
            file_mode (str): The mode in which the file is opened. Must be a binary mode ("wb" or "ab").

        Raises:
            ValueError: If the file_mode is not in binary mode.
        """
        self._run = True

        if not file_mode.endswith("b"):
            raise ValueError('open file in binary mode ("wb" or "ab")')

        self._file_args = filename, file_mode
        super().__init__(ctx=multiprocessing.get_context())
        threading.Thread(target=self.poll_to_file, name="monitor").start()

    def write(self, item):
        """
        Writes an item to the queue.

        Args:
            item: The item to write to the queue.
        """
        self.put(item)

    def flush(self):
        """
        Flushes the queue. This is a no-op for this implementation.
        """
        pass

    def close(self):
        """
        Closes the writer by stopping the polling thread.
        """
        self._run = False

    def poll_to_file(self):
        """
        Polls the queue and writes items to the file.

        This runs in the parent/main process, which polls the queue and writes items to the file.
        """
        with open(*self._file_args) as fd:
            while self._run:
                try:
                    fd.write(self.get(timeout=0.5))
                except Empty:
                    pass

    def __del__(self):
        """
        Destructor to ensure the writer is closed.
        """
        self.close()


def log_to_json(file_name_or_queue, mode="wb"):
    """set up logging via eliot that queued from worker processes"""
    if not isinstance(file_name_or_queue, QueuedLogWriter):
        queue = QueuedLogWriter(file_name_or_queue, mode)

    eliot.to_file(queue)


def multiindex_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """shift index levels into columns, leaving only the first.

    This is necessary when packing a dask.dataframe, which doesn't yet support multilevel indexing.
    """
    return df.reset_index(df.index.names[1:])

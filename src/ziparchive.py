import quickle
import zipfile#from zipfile import ZipFile, ZipInfo
import functools
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .seamf import read_seamf
except ImportError:
    from seamf import read_seamf

class QuackZipInfo(quickle.Struct):
    """ duck-typed ZipInfo clone for fast IPC serialized with quickle """

    orig_filename: str
    filename: str
    date_time: tuple
    compress_type: int
    _compresslevel: int
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
        kws = {k: getattr(zinfo, k) for k in zinfo.__slots__}
        return cls(**kws)

    @classmethod
    def from_filelist(cls, filelist):
        return [cls.from_zipinfo(zinfo) for zinfo in filelist]

    
class CachedZipFile(zipfile.ZipFile):
    _encoder = quickle.Encoder(registry=[QuackZipInfo])
    _decoder = quickle.Decoder(registry=[QuackZipInfo])

    def __init__(self, cache, *args, **kws):
        self.cache = cache
        super().__init__(*args, **kws)

    def _RealGetContents(self):
        """ monkeypatched: not so 'real' any more :) """
        cache = self._decoder.loads(self.cache)
        for k,v in cache.items():
            setattr(self, k, v)

class MultiProcessingZipFile:
    """ Serializable ZipFile for fast IPC.

    It remains closed unless a context is opened. While open, the
    object is not serializable.
    """

    def __init__(self, *args, **kws):
        self._zfile = None

        self.cache = kws.pop('cache', None)

        if self.cache is None:
            # do a one-time cache using the file scan done by ZipFile .__init__
            zfile = zipfile.ZipFile(*args, **kws)
            filelist = QuackZipInfo.from_filelist(zfile.filelist)
            NameToInfo = dict(zip(zfile.NameToInfo.keys(), filelist))
            self.cache = CachedZipFile._encoder.dumps(dict(
                _comment=zfile._comment,
                start_dir=zfile.start_dir,
                filelist=filelist,
                NameToInfo=NameToInfo,
            ))
            self._names = sorted([f.filename for f in filelist])
            zfile.close()

        # reuse these on entry
        self.args = tuple(args)
        self.kws = dict(kws)

    def namelist(self):
        return self._names

    @classmethod
    def from_zipfile(cls, zfile: zipfile.ZipFile):
        kws = dict(
            file=zfile.filename,
            mode=zfile.mode,
            compression=zfile.compression,
            compresslevel=zfile.compresslevel,
            strict_timestamps=zfile._strict_timestamps,
            allowZip64=zfile._allowZip64
        )
        return cls(**kws)
    
    @classmethod
    def from_mpzipfile(cls, zfile):
        return cls(cache=zfile.cache, *zfile.args, **zfile.kws)


    def open(self, fn):
        if self._zfile is None:
            raise IOError('open a zipfile context first')
        return self._zfile.open(fn)

    def __enter__(self):
        self._zfile = CachedZipFile(self.cache, *self.args, **self.kws)
        return self

    def __exit__(self, type, value, traceback):
        self._zfile, _ = None, self._zfile.__exit__(type, value, traceback)


def concat_dicts(*dicts):
    """ concats dictionaries of dataframes by key """
    if len(dicts)==1:
        return dicts[0]
    
    return {
        k: pd.concat([d[k] for d in dicts])
        for k in dicts[0].keys()
        if isinstance(dicts[0][k], pd.DataFrame)
    }


def read_seamf_zipfile(zipfile_or_path, force_loader_cls=False, dtype=pd.DataFrame, allowlist:list=None, trace_type:str=None) -> list:
    """ reads SEA-SigMF sensor data file(s) from the specified archive

    If allowlist is None, then all files are read. For other arguments, see `read_sea_sigmf`.
    """
    def single_read(zfile, fn):
        with zfile.open(fn) as fd:
            sigmf = read_seamf(
                fd,
                force_loader_cls=force_loader_cls,
                container_cls=dtype
            )

            if trace_type is not None:
                return sigmf[trace_type]
            else:
                return sigmf

    if isinstance(zipfile_or_path, zipfile.ZipFile):
        # TODO: this will probably fail if zipfile_or_path was opened as anything
        # besides a path string
        zfile = MultiProcessingZipFile.from_zipfile(zipfile_or_path)
    elif isinstance(zipfile_or_path, (str, Path)):
        zfile = MultiProcessingZipFile(zipfile_or_path)
    elif isinstance(zipfile_or_path, MultiProcessingZipFile):
        zfile = MultiProcessingZipFile.from_mpzipfile(zipfile_or_path)
    else:
        raise TypeError(f'unsupported type {type(zipfile_or_path)} in argument "zipfile_or_path"')

    with zfile:
        if allowlist is None:
            allowlist = zfile.filelist

        ret = [single_read(zfile, filename) for filename in allowlist]

    if trace_type is None:
        return concat_dicts(*ret)
    else:
        return pd.concat(ret)
import functools
import hashlib
import lzma
import typing
from collections import defaultdict, namedtuple
from pathlib import Path
from tarfile import TarFile

import methodtools
import numpy as np
import pandas as pd
from frozendict import frozendict
#from timezonefinder import TimezoneFinder

from .util import (
    _cartesian_multiindex,
    _flatten_dict,
    _iso_to_datetime,
    localize_timestamps,
    cpd,
)

from .schemas import (
    SchemaBase,
    MetadataPre0_4,
    Metadata0_4,
    Metadata0_5,
    Metadata0_6,
    VersionInfo,
)

TypeInfo = namedtuple("_TI", field_names=("type", "metadata"))
_UNLABELED_TRACE = frozendict({None: None})

# cached timezone lookups from location

#timezone_at = functools.lru_cache(TimezoneFinder().unique_timezone_at)
def timezone_at(lng,lat):
    if -119<lng<-116:
        return("America/Los_Angeles")
    elif -116<lng<-80:
        return("America/Denver")
    elif -80<lng<60:
        return("America/Boston")
    else:
        return None
def _capture_index(channel_metadata: dict) -> pd.MultiIndex:
    """
    Returns a pd.MultiIndex containing datetime and frequency levels.

    Args:
        channel_metadata (dict): Mapping keyed on frequency; each entry is a dict that includes a 'datetime' key.

    Returns:
        pd.MultiIndex: MultiIndex with datetime and frequency levels.
    """
    times, freqs = zip(*[[d["datetime"], k] for k, d in channel_metadata.items()])

    return pd.MultiIndex(
        levels=(times, freqs),
        codes=2 * [list(range(len(channel_metadata)))],
        names=("datetime", "frequency"),
        sortorder=0,
    )


@functools.lru_cache()
def _pfp_index(
    pfp_sample_count: int, pvt_sample_count: int, iq_capture_duration_sec: float
) -> pd.MultiIndex:
    """
    Returns a pd.MultiIndex for PFP index.

    Args:
        pfp_sample_count (int): Number of PFP samples.
        pvt_sample_count (int): Number of PVT samples.
        iq_capture_duration_sec (float): IQ capture duration in seconds.

    Returns:
        pd.MultiIndex: MultiIndex for PFP index.
    """
    base = pd.RangeIndex(pfp_sample_count, name="Frame time elapsed (s)")
    return base * (iq_capture_duration_sec / pfp_sample_count / pvt_sample_count)


@functools.lru_cache()
def _pvt_index(pvt_sample_count: int, iq_capture_duration_sec: float) -> pd.MultiIndex:
    """
    Returns a pd.MultiIndex for PVT index.

    Args:
        pvt_sample_count (int): Number of PVT samples.
        iq_capture_duration_sec (float): IQ capture duration in seconds.

    Returns:
        pd.MultiIndex: MultiIndex for PVT index.
    """
    base = pd.RangeIndex(pvt_sample_count, name="Capture time elapsed (s)")
    return base * (iq_capture_duration_sec / pvt_sample_count)


@functools.lru_cache()
def _psd_index(psd_sample_count, analysis_bandwidth_Hz) -> pd.MultiIndex:
    """
    Returns a pd.MultiIndex for PSD index.

    Args:
        psd_sample_count (int): Number of PSD samples.
        analysis_bandwidth_Hz (float): Analysis bandwidth in Hertz.

    Returns:
        pd.MultiIndex: MultiIndex for PSD index.
    """
    base = pd.RangeIndex(psd_sample_count, name="Frequency (Hz)")
    return base * (analysis_bandwidth_Hz / psd_sample_count)


@functools.lru_cache()
def _trace_index(trace_dicts: typing.Tuple[frozendict]) -> pd.MultiIndex:
    df = pd.DataFrame(trace_dicts).sort_index(axis=1)
    if "detector" in df.columns:
        df.loc[:, "detector"].replace({"max": "peak", "mean": "rms"}, inplace=True)
    if "capture_statistic" in df.columns:
        df.loc[:, "capture_statistic"].replace(
            {"maximum": "max", "minimum": "min"}, inplace=True
        )
    return pd.MultiIndex.from_frame(df)


class _LoaderBase:
    TABULAR_GROUPS: tuple
    trace_starts: dict
    trace_axes: dict
    meta: frozendict

    def __init__(self, json_meta: SchemaBase, tz: str):
        """initialize the object to unpack numpy.ndarray or pandas.DataFrame objects"""
        raise NotImplementedError

    def unpack_arrays(self, data):
        """split the flat data vector into a dictionary of named traces"""

        split_inds = list(self.trace_starts.keys())[1:]
        traces = np.array_split(data, split_inds)

        trace_groups = defaultdict(lambda: defaultdict(list))
        for (trace_type, trace_name), trace in zip(self.trace_starts.values(), traces):
            trace_groups[trace_type][trace_name].append(np.array(trace))

        trace_groups = {
            trace_type: {
                name: np.array(v) for name, v in trace_groups[trace_type].items()
            }
            for trace_type in self.TABULAR_GROUPS
        }

        return dict(trace_groups, **self.meta)

    def unpack_dataframes(self, data):
        trace_groups = self.unpack_arrays(data)

        # tabular data
        capture_index = _capture_index(self.meta["channel_metadata"])

        frames = {}
        for name in trace_groups.keys():
            if name not in self.TABULAR_GROUPS:
                # not tabular set of traces of the same size
                frames[name] = dict(trace_groups[name])
            elif tuple(trace_groups[name].keys()) != (_UNLABELED_TRACE,):
                # keyed sub-traces at each frequency
                group_data = np.array(list(trace_groups[name].values())).swapaxes(0, 1)
                group_data = group_data.reshape(
                    (group_data.shape[0] * group_data.shape[1], group_data.shape[2])
                )
                trace_index = _trace_index(tuple(trace_groups[name].keys()))
                index = _cartesian_multiindex(capture_index, trace_index)

                # take the column definition from the first trace key, under the assumption
                # that they are the same for tabular data
                columns = self.trace_axes[name]
                frames[name] = pd.DataFrame(
                    group_data, index=index, columns=columns, dtype=group_data.dtype
                )

                if len(frames[name].index.names) > len(capture_index):
                    sort_order = ["datetime"] + frames[name].index.names[
                        len(capture_index) :
                    ]
                    frames[name].sort_index(inplace=True, level=sort_order)
            else:
                # a single unlabeled trace per frequency
                group_data = list(trace_groups[name].values())[0]
                columns = self.trace_axes[name]
                frames[name] = pd.DataFrame(
                    group_data,
                    index=capture_index,
                    columns=columns,
                    dtype=group_data.dtype,
                )

        # channel metadata
        values = np.array(
            [list(v.values()) for v in self.meta["channel_metadata"].values()]
        )
        for v in self.meta["channel_metadata"].values():
            columns = v.keys()
            break

        channel_metadata = cpd.DataFrame(
            values,
            cache_key=("channel_metadata", type(self)),
            index=capture_index,
            columns=columns,
        )

        for col_name in ("datetime", "frequency"):
            if col_name in columns:
                channel_metadata.drop(col_name, axis=1, inplace=True)

        sweep_metadata = cpd.DataFrame(
            [self.meta["sweep_metadata"]],
            cache_key=("sweep_metadata", type(self)),
            copy=False,
        )

        return dict(
            frames,
            channel_metadata=channel_metadata,
            sweep_metadata=sweep_metadata,
            sensor_metadata=self.meta["sensor_metadata"],
        )


class _Loader_v1(_LoaderBase):
    TABULAR_GROUPS = "psd", "pvt", "pfp"

    # these are hard-coded since introspection of the trace info was difficult
    # and data were only generated in this structure
    TRACE_INFO = {
        "psd_max_power": TypeInfo("psd", frozendict(capture_statistic="max")),
        "psd_mean_power": TypeInfo("psd", frozendict(capture_statistic="mean")),
        "pvt_max_power": TypeInfo("pvt", frozendict(detector="peak")),
        "pvt_mean_power": TypeInfo("pvt", frozendict(detector="rms")),
        "pfp_rms_min_power": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="min")
        ),
        "pfp_rms_max_power": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="max")
        ),
        "pfp_rms_mean_power": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="mean")
        ),
        "pfp_peak_min_power": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="min")
        ),
        "pfp_peak_max_power": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="max")
        ),
        "pfp_peak_mean_power": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="mean")
        ),
        "apd_p_pct": TypeInfo("apd", "percentile"),
        "apd_a_dBm": TypeInfo("apd", "sample_amplitude_dBm"),
    }

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta: MetadataPre0_4):
        if json_meta.timezone is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify on load (e.g., "America/New_York")'
            )

        self.trace_starts = {}
        self.trace_axes = {}

        channel_meta = defaultdict(dict)

        sample_rate = json_meta.global_.sample_rate
        sweep_meta = dict(
            sample_rate=sample_rate,
            version=json_meta.global_.version,
            metadata_version="v0.1",
            calibration_datetime=_iso_to_datetime(
                json_meta.global_.calibration_datetime,
                # json_meta["timezone"]
            ),
            schedule_name=json_meta.global_.schedule["name"],
            schedule_start_datetime=json_meta.global_.schedule["start"],
        )

        capture_fields = {d["core:sample_start"]: d for d in json_meta.captures}

        for annot in json_meta.annotations:
            if annot["ntia-core:annotation_type"] == "CalibrationAnnotation":
                frequency = capture_fields[annot["core:sample_start"]]["core:frequency"]
                channel_meta[frequency].update(
                    cal_gain_dB=annot["ntia-sensor:gain_sensor"],
                    cal_noise_figure_dB=annot["ntia-sensor:noise_figure_sensor"],
                )

                sweep_meta["calibration_enbw"] = annot["ntia-sensor:enbw_sensor"]
                sweep_meta["calibration_temperature_degC"] = annot[
                    "ntia-sensor:temperature"
                ]

            elif annot["ntia-core:annotation_type"] == "SensorAnnotation":
                capture = capture_fields[annot["core:sample_start"]]
                frequency = capture["core:frequency"]
                timestamp = _iso_to_datetime(
                    capture["core:datetime"],
                )
                channel_meta[frequency].update(
                    frequency=frequency,
                    datetime=timestamp,
                    overload=annot["ntia-sensor:overload"],
                    sigan_attenuation_dB=annot["ntia-sensor:attenuation_setting_sigan"],
                )

            else:
                # in this case, proceed under the assumption that this is a trace
                trace_type_label = self._trace_label(annot)
                trace_key = self.TRACE_INFO[trace_type_label]
                self.trace_starts[annot["core:sample_start"]] = trace_key

                self._update_frame_axes(annot, trace_key[0], sample_rate)

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=sweep_meta,
            sensor_metadata=frozendict(timezone=json_meta.timezone),
        )

    def _update_frame_axes(self, annot, trace_type, sample_rate):
        if trace_type == "pfp":
            self.trace_axes[trace_type] = _pfp_index(
                annot["core:sample_count"],
                400,  # TODO: pass from recent pvt trace data
                4.0,  # TODO: pass from recent pvt trace data
            )

        elif trace_type == "pvt":
            self.trace_axes[trace_type] = _pvt_index(
                annot["core:sample_count"],
                annot["ntia-algorithm:number_of_samples"] / sample_rate,
            )

        elif trace_type == "psd":
            self.trace_axes[trace_type] = _psd_index(annot["core:sample_count"], 10e6)

    @staticmethod
    def _trace_label(annot: dict):
        """returns a trace label generated from the annotation string"""
        if annot["ntia-core:annotation_type"] == "FrequencyDomainDetection":
            return "psd_" + annot["ntia-algorithm:detector"][4:]
        elif annot["ntia-core:annotation_type"] == "TimeDomainDetection":
            return "pvt_" + annot["ntia-algorithm:detector"]
        else:
            return annot.get("core:label", None)


class _Loader_v2(_LoaderBase):
    TABULAR_GROUPS = "psd", "pvt", "pfp"

    # these are hard-coded since introspection of the trace info was difficult
    # and data were only generated in this structure
    TRACE_INFO = {
        "max_fft": TypeInfo("psd", frozendict(capture_statistic="max")),
        "mean_fft": TypeInfo("psd", frozendict(capture_statistic="mean")),
        "max_td_pwr_series": TypeInfo("pvt", frozendict(detector="peak")),
        "mean_td_pwr_series": TypeInfo("pvt", frozendict(detector="rms")),
        "min_rms_pfp": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="min")
        ),
        "max_rms_pfp": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="max")
        ),
        "mean_rms_pfp": TypeInfo(
            "pfp", frozendict(detector="rms", capture_statistic="mean")
        ),
        "min_peak_pfp": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="min")
        ),
        "max_peak_pfp": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="max")
        ),
        "mean_peak_pfp": TypeInfo(
            "pfp", frozendict(detector="peak", capture_statistic="mean")
        ),
        "apd_p": TypeInfo("apd", "percentile"),
        "apd_a": TypeInfo("apd", "sample_amplitude_dBm"),
    }

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta: MetadataPre0_4):
        self.trace_starts = {}
        self.trace_axes = {}

        channel_meta = defaultdict(dict)
        diagnostics = json_meta.global_.diagnostics

        sweep_meta = dict(
            sample_rate=json_meta.global_.sample_rate,
            version=json_meta.global_.version,
            metadata_version=json_meta.global_.extensions["ntia-nasctn-sea"],
            calibration_datetime=_iso_to_datetime(
                json_meta.global_.calibration_datetime,
            ),
            schedule_name=json_meta.global_.schedule["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta.global_.schedule["start"],
            ),
            task=json_meta.global_.task,
            diagnostics_datetime=_iso_to_datetime(diagnostics["diagnostics_datetime"]),
        )
        # Interval key may not exist in schedule entry, for one-off action results
        if "interval" in json_meta.global_.schedule:
            sweep_meta.update(dict(schedule_interval=json_meta.global_.schedule["interval"]))
        else:
            sweep_meta.update(dict(schedule_interval=None))

        for v in diagnostics.values():
            if isinstance(v, dict):
                sweep_meta.update(v)

        for capture in json_meta.captures:
            frequency = capture["core:frequency"]

            for k, v in capture.items():
                if k == "core:frequency" or k.endswith("sample_count"):
                    continue
                elif k.endswith("sample_start") and not k.startswith("core:"):
                    pass
                elif k == "core:datetime":
                    ts = _iso_to_datetime(
                        v,
                    )
                    channel_meta[frequency][k.split(":", 1)[-1]] = ts
                    continue
                else:
                    channel_meta[frequency][k.split(":", 1)[-1]] = v
                    continue

                # continues here only if this is a *_sample_start entry
                trace_name, _ = k.rsplit("_sample_start", 1)
                trace_key = self.TRACE_INFO[trace_name]
                self.trace_starts[v] = trace_key

        # use the most recent capture info to populate the trace axis objects
        self.trace_axes["pfp"] = _pfp_index(
            capture["pfp_sample_count"],
            capture["td_pwr_sample_count"],
            capture["iq_capture_duration_msec"] / 1000.0,
        )
        self.trace_axes["pvt"] = _pvt_index(
            capture["td_pwr_sample_count"], capture["iq_capture_duration_msec"] / 1000.0
        )
        self.trace_axes["psd"] = _psd_index(capture["fft_sample_count"], 10e6)

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=sweep_meta,
            sensor_metadata=frozendict(timezone=json_meta.timezone),
        )


class _Loader_v3(_LoaderBase):
    TABULAR_GROUPS = dict(
        psd="power_spectral_density",
        pvt="time_series_power",
        pfp="periodic_frame_power",
    )

    DETECTOR_NAMEMAP = dict(max="peak", mean="rms")

    @classmethod
    def _parse_trace_string(cls, dp_type, meta_str) -> tuple:
        """returns a (detector,) or (detector, capture_statistic) tuple"""
        split = tuple(meta_str.split("_"))
        if dp_type == "pvt":
            detector = cls.DETECTOR_NAMEMAP[split[0]]
            return frozendict(detector=detector)
        elif dp_type == "psd":
            return frozendict(capture_statistic=split[0])
        elif dp_type == "pfp":
            return frozendict(detector=split[0], capture_statistic=split[1])
        else:
            raise ValueError(f"unknown data product type '{dp_type}'")

    @classmethod
    @methodtools.lru_cache()
    def _get_trace_metadata(cls, trace_info: MetadataPre0_4.Global.DataProducts):
        # this is cached since we expect data_products not to change very often
        offset_total = 0
        trace_offsets = []
        trace_labels = []
        for short_name, json_name in cls.TABULAR_GROUPS.items():
            dp_field = getattr(trace_info, json_name)
            for trace_name in getattr(dp_field, "detector", [None]):
                trace_offsets.append(offset_total)
                trace_meta = cls._parse_trace_string(short_name, trace_name)
                trace_labels.append(TypeInfo(short_name, trace_meta))
                offset_total += dp_field.sample_count

        cls._trace_offsets = dict(zip(trace_offsets, trace_labels))

        return np.array(trace_offsets), trace_labels

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta: MetadataPre0_4):
        # get the vectors of offset indices of the each trace relative to the capture start
        trace_info = json_meta.global_.data_products
        trace_offsets, trace_labels = self._get_trace_metadata(trace_info)

        # in v0.4 we can add apd here too
        self.trace_axes = {}
        sample_rate = json_meta.global_.sample_rate
        capture_duration = json_meta.captures[0]["iq_capture_duration_msec"] / 1000.0

        self.trace_axes["pfp"] = _pfp_index(
            trace_info.periodic_frame_power.sample_count,
            trace_info.time_series_power.sample_count,
            capture_duration,
        )

        self.trace_axes["pvt"] = _pvt_index(
            trace_info.time_series_power.sample_count, capture_duration
        )
        psd_samples = trace_info.power_spectral_density.sample_count
        self.trace_axes["psd"] = _psd_index(
            psd_samples,
            sample_rate
            * psd_samples
            / trace_info.power_spectral_density.number_of_samples_in_fft,
        )

        # cycle through the channels for metadata and data index maps
        channel_meta = {}
        trace_starts_keys = []
        trace_starts_values = []
        apd_start_offset = trace_offsets[-1] + (trace_offsets[-1] - trace_offsets[-2])
        apd_trace_info = [
            TypeInfo(type="apd_p", metadata=frozendict({})),
            TypeInfo(type="apd_a", metadata=frozendict({})),
        ]
        for capture, apd_length in zip(
            json_meta.captures,
            trace_info.amplitude_probability_distribution.sample_count,
        ):
            capture = dict(capture)

            frequency = capture.pop("core:frequency")
            sample_start = capture.pop("core:sample_start")
            capture["datetime"] = _iso_to_datetime(
                capture.pop("core:datetime"),
            )

            channel_meta[frequency] = capture

            trace_starts_keys.extend(sample_start + trace_offsets)
            trace_starts_values.extend(trace_labels)

            # the following is a messy hack that can be removed in v0.4
            trace_starts_keys.extend(
                [
                    sample_start + apd_start_offset,
                    sample_start + apd_start_offset + apd_length,
                ]
            )
            trace_starts_values.extend(apd_trace_info)

        self.trace_starts = dict(zip(trace_starts_keys, trace_starts_values))

        # slurp up the metadata
        diagnostics = json_meta.global_.diagnostics
        sweep_meta = dict(
            sample_rate=json_meta.global_.sample_rate,
            version=json_meta.global_.version,
            metadata_version=json_meta.global_.extensions["ntia-nasctn-sea"],
            calibration_datetime=_iso_to_datetime(
                json_meta.global_.calibration_datetime,
            ),
            calibration_temperature_degC=json_meta.global_.calibration_temperature_degC,
            schedule_name=json_meta.global_.schedule["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta.global_.schedule["start"],
            ),
            task=json_meta.global_.task,
            diagnostics_datetime=_iso_to_datetime(diagnostics["diagnostics_datetime"]),
        )
        # Interval key may not exist in schedule entry, for one-off action results
        if "interval" in json_meta.global_.schedule:
            sweep_meta.update(dict(schedule_interval=json_meta.global_.schedule["interval"]))
        else:
            sweep_meta.update(dict(schedule_interval=None))

        sweep_meta.update(_flatten_dict(dict(diagnostics=diagnostics)))

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=frozendict(sweep_meta),
            sensor_metadata=frozendict(timezone=json_meta.timezone),
        )


class _Loader_v4(_LoaderBase):
    TABULAR_GROUPS = dict(
        psd="power_spectral_density",
        pvt="time_series_power",
        pfp="periodic_frame_power",
        apd="amplitude_probability_distribution",
    )

    CAPTURE_KEYMAP = {
        "ntia-sensor:overload": "overload",
        "ntia-sensor:duration": "iq_capture_duration_ms",
        "noise_figure": "cal_noise_figure_dB",
        "gain": "cal_gain_dB",
        "temperature": "cal_temperature_degC",
        "reference_level": "sigan_reference_level_dBm",
        "attenuation": "sigan_attenuation_dB",
        "preamp_enable": "sigan_preamp_enable",
    }

    TRACE_FIELD_NAMEMAP = {
        # ordered and name mapping from metadata into dataframe.
        # should be all-inclusive
        "statistic": "capture_statistic",
        "detector": "detector",
        None: None,
    }

    @classmethod
    @methodtools.lru_cache()
    def _get_trace_metadata(cls, data_products: Metadata0_4.Global.DataProducts):
        # this is cached since we expect data_products not to change very often
        offset_total = 0
        trace_offsets = []
        trace_labels = []

        for short_name, json_name in cls.TABULAR_GROUPS.items():
            dp_field = getattr(data_products, json_name)

            for trace_obj in getattr(dp_field, "traces", [_UNLABELED_TRACE]):
                # APD has no trace object; temporarily populate trace_labels
                # This is later removed in unpack_dataframes or unpack_arrays
                trace_offsets.append(offset_total)

                trace_obj = frozendict(
                    {
                        cls.TRACE_FIELD_NAMEMAP.get(k, k): trace_obj[k]
                        for k in cls.TRACE_FIELD_NAMEMAP.keys()
                        if k in trace_obj.keys()
                    }
                )
                trace_labels.append(TypeInfo(short_name, trace_obj))
                offset_total += dp_field.length

        cls._trace_offsets = dict(zip(trace_offsets, trace_labels))

        return np.array(trace_offsets), trace_labels

    @staticmethod
    @functools.lru_cache()
    def _apd_index(apd_min_bin_dBm, apd_max_bin_dBm, apd_bin_size_dB) -> pd.MultiIndex:
        return pd.RangeIndex(
            start=apd_min_bin_dBm,
            stop=apd_max_bin_dBm + apd_bin_size_dB,
            step=apd_bin_size_dB,
            name="Channel Power (dBm/10MHz)",
        )

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta: Metadata0_4):
        # get the vectors of offset indices of the each trace relative to the capture start
        data_products = json_meta.global_.data_products
        trace_offsets, trace_labels = self._get_trace_metadata(data_products)

        self.trace_axes = {}
        sample_rate = json_meta.global_.sample_rate
        # (safely) assume all capture durations are identical
        capture_duration = json_meta.captures[0]["ntia-sensor:duration"] / 1000.0

        self.trace_axes["pfp"] = _pfp_index(
            data_products.periodic_frame_power.length,
            data_products.time_series_power.length,
            capture_duration,
        )
        self.trace_axes["pvt"] = _pvt_index(
            data_products.time_series_power.length, capture_duration
        )
        psd_samples = data_products.power_spectral_density.length
        self.trace_axes["psd"] = _psd_index(
            psd_samples,
            sample_rate * psd_samples / data_products.power_spectral_density.samples,
        )
        self.trace_axes["apd"] = self._apd_index(
            data_products.amplitude_probability_distribution.min_amplitude,
            data_products.amplitude_probability_distribution.max_amplitude,
            data_products.amplitude_probability_distribution.amplitude_bin_size,
        )

        # cycle through the channels for metadata and data index maps
        channel_meta = {}
        trace_starts_keys = []
        trace_starts_values = []

        for capture in json_meta.captures:
            capture = dict(capture)

            frequency = capture.pop("core:frequency")
            sample_start = capture.pop("core:sample_start")
            timestamp = capture.pop("core:datetime")

            # pull calibration info and sigan settings up a level
            for k in ["sensor_calibration", "sigan_settings"]:
                capture.update(capture.pop(f"ntia-sensor:{k}"))
            # change key names for backwards-compatibility
            # note: "cal_temperature_degC" key is new in v4
            capture = {self.CAPTURE_KEYMAP.get(k, k): v for k, v in capture.items()}
            capture["datetime"] = _iso_to_datetime(timestamp)

            channel_meta[frequency] = capture

            trace_starts_keys.extend(sample_start + trace_offsets)
            trace_starts_values.extend(trace_labels)

        self.trace_starts = dict(zip(trace_starts_keys, trace_starts_values))

        # slurp up the metadata
        sweep_meta = dict(
            sample_rate=json_meta.global_.sample_rate,
            version=json_meta.global_.version,
            metadata_version=json_meta.global_.extensions[5]["version"],
            schedule_name=json_meta.global_.schedule["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta.global_.schedule["start"],
            ),
            task=json_meta.global_.task,
            diagnostics_datetime=_iso_to_datetime(
                json_meta.global_.diagnostics["datetime"],
            ),
        )
        # Interval key may not exist in schedule entry, for one-off action results
        if "interval" in json_meta.global_.schedule:
            sweep_meta.update(dict(schedule_interval=json_meta.global_.schedule["interval"]))
        else:
            sweep_meta.update(dict(schedule_interval=None))
        
        sweep_meta.update(json_meta.global_.diagnostics)

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=frozendict(sweep_meta),
            sensor_metadata=frozendict(timezone=json_meta.timezone),
        )


class _Loader_v5(_LoaderBase):
    TABULAR_GROUPS = dict(
        psd="Power Spectral Density",
        pvt="Power vs. Time",
        pfp="Periodic Frame Power",
        apd="Amplitude Probability Distribution",
    )

    CAPTURE_KEYMAP = {
        "ntia-sensor:overload": "overload",
        "ntia-sensor:duration": "iq_capture_duration_ms",
        "noise_figure": "cal_noise_figure_dB",
        "gain": "cal_gain_dB",
        "temperature": "cal_temperature_degC",
        "reference_level": "sigan_reference_level_dBm",
        "attenuation": "sigan_attenuation_dB",
        "preamp_enable": "sigan_preamp_enable",
    }

    TRACE_FIELD_NAMEMAP = {
        # ordered and name mapping from metadata into dataframe.
        # should be all-inclusive
        "statistic": "capture_statistic",
        "detector": "detector",
        None: None,
    }

    TRACE_INDEX_LABELMAP = dict(
        # Units filled from metadata object
        psd="Baseband Frequency ({})",
        pvt="Capture time elapsed ({})",
        pfp="Frame time elapsed ({})",
        apd="Channel Power ({}/10MHz)",
    )

    @classmethod
    @methodtools.lru_cache()
    def _get_trace_metadata(cls, data_products: typing.Tuple[Metadata0_5.Global.Graph]):
        # Cached since data_products should not change often
        offset_total = 0
        trace_offsets = []
        trace_labels = []
        for graph_obj in data_products:
            short_name = list(cls.TABULAR_GROUPS.keys())[
                list(cls.TABULAR_GROUPS.values()).index(graph_obj.name)
            ]
            if short_name not in cls.TABULAR_GROUPS:
                raise ValueError(
                    f"Unknown data product encountered in metadata: {graph_obj.name}"
                )
            if short_name == "apd":
                # APD handling: mimic v4 loading with temporary unlabeled trace object
                # The APD "Graph" object does not have a "series" (graph_obj.series is None here)
                trace_offsets.append(offset_total)
                trace_obj = frozendict({k: v for k, v in _UNLABELED_TRACE.items()})
                trace_labels.append(TypeInfo(short_name, trace_obj))
                offset_total += graph_obj.length
            else:
                # For other data products, graph_obj.series is an iterable of series (trace) names
                # This is different from the trace objects present in v4, so handle them individually
                # by the data product type (parsed from the name).
                for series_name in graph_obj.series:
                    trace_offsets.append(offset_total)
                    if short_name == "pfp":
                        # PFP handling: statistic and detector, delimited by "_"
                        # Example: "mean_minimum" is RMS detector, minimum statistic.
                        trace_obj = frozendict(
                            {
                                "capture_statistic": series_name.split("_")[1],
                                "detector": series_name.split("_")[0],
                            }
                        )
                    elif short_name == "psd":
                        # PSD series names are statistic names
                        trace_obj = frozendict({"capture_statistic": series_name})
                    elif short_name == "pvt":
                        # PVT series names are detector names
                        trace_obj = frozendict({"detector": series_name})
                    offset_total += graph_obj.length
                    trace_labels.append(TypeInfo(short_name, trace_obj))
        cls._trace_offsets = dict(zip(trace_offsets, trace_labels))
        return np.array(trace_offsets), trace_labels

    def _set_trace_axes(self, data_products: typing.Tuple[Metadata0_5.Global.Graph]):
        # Set the axes/index for each data product, using the Graph metadata
        for graph in data_products:
            short_name = list(self.TABULAR_GROUPS.keys())[
                list(self.TABULAR_GROUPS.values()).index(graph.name)
            ]
            if short_name == "apd":
                # APD handling: use y-values instead of x-values
                start, step, units = graph.y_start, graph.y_step, graph.y_units
            else:
                start, step, units = graph.x_start, graph.x_step, graph.x_units
            assert all([len(s) == 1 for s in [start, step]])
            start, step = start[0], step[0]
            if short_name == "apd":
                start, step = int(start), int(step)
            # Generate and store the index
            self.trace_axes[short_name] = (
                pd.RangeIndex(
                    graph.length,
                    name=self.TRACE_INDEX_LABELMAP[short_name].format(units),
                )
                * step
                + start
            )

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta: Metadata0_5):
        data_products = json_meta.global_.data_products
        trace_offsets, trace_labels = self._get_trace_metadata(data_products)
        self.trace_axes = {}

        self._set_trace_axes(data_products)

        channel_meta = {}
        trace_starts_keys = []
        trace_starts_values = []

        for capture in json_meta.captures:
            capture = dict(capture)
            frequency = capture.pop("core:frequency")
            sample_start = capture.pop("core:sample_start")
            timestamp = capture.pop("core:datetime")

            for k in ["sensor_calibration", "sigan_settings"]:
                capture.update(capture.pop(f"ntia-sensor:{k}"))
            del capture["reference"]

            capture = {self.CAPTURE_KEYMAP.get(k, k): v for k, v in capture.items()}
            # note: calibration datetime is discarded, since the
            # "datetime" key now refers to the "core:datetime" value (measurement timestamp)
            # To retain the cal_timestamp, uncomment the following line:
            # cal_timestamp = capture["datetime"]
            capture["datetime"] = _iso_to_datetime(timestamp)
            channel_meta[frequency] = capture
            trace_starts_keys.extend(sample_start + trace_offsets)
            trace_starts_values.extend(trace_labels)
        self.trace_starts = dict(zip(trace_starts_keys, trace_starts_values))

        sweep_meta = dict(
            sample_rate=json_meta.global_.sample_rate,
            version=json_meta.global_.version,
            metadata_version=json_meta.global_.extensions[6]["version"],
            schedule_name=json_meta.global_.schedule["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta.global_.schedule["start"],
            ),
            task=json_meta.global_.task,
            diagnostics_datetime=_iso_to_datetime(
                json_meta.global_.diagnostics["datetime"],
            ),
        )
        # Interval key may not exist in schedule entry, for one-off action results
        if "interval" in json_meta.global_.schedule:
            sweep_meta.update(dict(schedule_interval=json_meta.global_.schedule["interval"]))
        else:
            sweep_meta.update(dict(schedule_interval=None))
        
        sweep_meta.update(json_meta.global_.diagnostics)
        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=frozendict(sweep_meta),
            sensor_metadata=frozendict(timezone=json_meta.timezone),
        )


def select_loader(
    json_str: str,
) -> typing.Tuple[SchemaBase, typing.Type[_LoaderBase]]:
    """return metadata schema and loader type from the json metadata str"""
    version_meta = VersionInfo.fromstr(json_str)
    extensions = version_meta.global_.extensions

    version = None
    if isinstance(extensions, (list, tuple)):
        # v0.4+
        version = [e["version"] for e in extensions if e["name"] == "ntia-nasctn-sea"][
            0
        ]
    elif isinstance(extensions, (dict, frozendict)):
        # v0.1, 0.2, or 0.3
        version = extensions.get("ntia-nasctn-sea", None)

    if version is None:
        return MetadataPre0_4.fromstr(json_str), _Loader_v1
    elif version == "v0.2":
        return MetadataPre0_4.fromstr(json_str), _Loader_v2
    elif version == "v0.3":
        return MetadataPre0_4.fromstr(json_str), _Loader_v3
    elif version == "v0.4":
        return Metadata0_4.fromstr(json_str), _Loader_v4
    elif version == "v0.4.0":  # This is confusing but correct for v5
        # The following manual JSON fixes are needed for parsing v5 metadata:
        proc_info_str = b'        "ntia-algorithm:processing_info": [\n            {'
        dft_info_str = b'},\n            {\n                "id": "psd_fft",'
        proc_info_str_min = b'"ntia-algorithm:processing_info":[{'
        dft_info_str_min = b'},{"id":"psd_fft",'
        json_str = json_str.replace(
            proc_info_str, proc_info_str + b'\n                "type": "DigitalFilter",'
        )
        json_str = json_str.replace(
            dft_info_str, dft_info_str + b'\n                "type": "DFT",'
        )
        json_str = json_str.replace(
            proc_info_str_min, proc_info_str_min + b'"type":"DigitalFilter",'
        )
        json_str = json_str.replace(
            dft_info_str_min, dft_info_str_min + b'"type":"DFT",'
        )
        return Metadata0_5.fromstr(json_str), _Loader_v5
    elif version in ["v0.6.0", "0.6.0"]:
        # This is not an error- _Loader_v5 works for v0.6.0 metadata due to the
        # similarity between the two versions (and the lack of explicit struct-based
        # handling for sensor hardware information and diagnostics -see the note in
        # the Metadata0_6 class for more information on this.)
        return Metadata0_6.fromstr(json_str), _Loader_v5
    else:
        raise ValueError(f'unrecognized format version "{version}"')


def read_seamf(
    file,
    force_loader_cls=False,
    container_cls=pd.DataFrame,
    hash_check=True,
    tz=None,
    localize=False,
) -> dict:
    """unpacks a sensor data file into a dictionary of numpy or pandas objects

    When `force_loader_cls` is False, the loader is selected automatically based on the
    metadata version. Other valid values are `None` (to return metadata dictionary and a flat numpy data array)
    or `bytes` (for the metadata dictionary and data in bytes form) or one of the _Loader_v* classes.

    The supported container object types are pandas.DataFrame and numpy.ndarray.

    When hash_check evaluates as True, the data contents are compared against the SHA512 hash
    in the metadata file.
    """
    if isinstance(file, (str, Path)):
        kws = {"name": file}
    else:
        kws = {"fileobj": file}

    if localize and not issubclass(container_cls, pd.DataFrame):
        raise ValueError("localize is only supported for pd.DataFrame container")

    with TarFile(**kws) as tar_fd:
        tar_names = tar_fd.getnames()

        # meta is plain json
        meta_name = [n for n in tar_names if n.endswith(".sigmf-meta")][0]
        json_str = tar_fd.extractfile(meta_name).read()

        data_name = [n for n in tar_names if n.endswith(".sigmf-data")][0]
        lzma_data = tar_fd.extractfile(data_name).read()

    meta, loader_cls = select_loader(json_str)

    # validate data binary
    if hash_check:
        data_hash = hashlib.sha512(lzma_data).hexdigest()
        if data_hash != meta.global_.sha512:
            raise IOError("seamf file data failed sha512 hash check")

    # try to automatically update time zone from metadata
    if tz is None:
        loc = meta.global_.geolocation.get("coordinates", None)

        if loc is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify, e.g., tz="America/New_York"'
            )
        else:
            tz = timezone_at(lng=loc[0], lat=loc[1])
    meta.timezone = tz

    # the duration of this operation is dominated by lzma.decompress, not disk access or numpy
    # (on a 2020-vintage laptop with an SSD)
    byte_data = lzma.decompress(lzma_data, format=lzma.FORMAT_XZ)
    if isinstance(force_loader_cls, type) and issubclass(force_loader_cls, bytes):
        return byte_data, meta

    data = np.frombuffer(byte_data, dtype="half")

    # pick the loader
    if force_loader_cls is None:
        return data, meta
    elif force_loader_cls == False:
        loader = loader_cls(meta)
    elif isinstance(force_loader_cls, type) and issubclass(
        force_loader_cls, _LoaderBase
    ):
        loader = force_loader_cls(meta)
    else:
        raise TypeError(
            f"unsupported type '{type(force_loader_cls)}' for argument force_loader_cls"
        )

    # unpack the data
    if issubclass(container_cls, pd.DataFrame):
        ret = loader.unpack_dataframes(data)
        if localize:
            localize_timestamps(ret)
        return ret

    elif issubclass(container_cls, np.ndarray):
        return loader.unpack_arrays(data)
    else:
        raise TypeError('invalid "container_cls"')


def read_seamf_meta(file, parse=True, tz=None):
    if isinstance(file, (str, Path)):
        kws = {"name": file}
    else:
        kws = {"fileobj": file}

    with TarFile(**kws) as tar_fd:
        name = [m for m in tar_fd.getmembers() if m.name.endswith(".sigmf-meta")]
        if len(name) != 1:
            raise Exception(f"SigMF metadata file not found in archive {file}")
        meta_contents = tar_fd.extractfile(name[0]).read()

    if not parse:
        return meta_contents

    meta, _ = select_loader(meta_contents)

    if tz is None:
        # try to automatically update time zone from metadata
        loc = meta.global_.geolocation.get("coordinates", None)

        if loc is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify, e.g., tz="America/New_York"'
            )
        else:
            tz = timezone_at(lng=loc[0], lat=loc[1])

    meta.timezone = tz

    return meta

import functools
import pandas as pd
import json
from pathlib import Path
from tarfile import TarFile
import numpy as np
import lzma
import hashlib
from collections import defaultdict, namedtuple
import methodtools
from frozendict import frozendict
import typing
from timezonefinder import TimezoneFinder


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
    )


def _iso_to_datetime(s, tz=None):
    """returns a timezone aware datetime from a metadata timestamp string"""

    if tz is None:
        return pd.Timestamp(np.datetime64(s[:-1])).tz_localize("utc")
    else:
        return pd.Timestamp(np.datetime64(s[:-1])).tz_localize("utc").tz_convert(tz)


_TI = namedtuple("_TI", field_names=("type", "metadata"))
_UNLABELED_TRACE = frozendict({None: None})

# cache the timezone lookups since we expect a discrete, small number of unique timezones
_unique_timezone_at = functools.lru_cache(TimezoneFinder().unique_timezone_at)


class _LoaderBase:
    def __init__(self, json_meta: dict, tz: str):
        """initialize the object to unpack numpy.ndarray or pandas.DataFrame objects"""
        raise NotImplementedError

    @staticmethod
    def _capture_index(channel_metadata: dict) -> pd.MultiIndex:
        """returns a pd.MultiIndex containing datetime and frequency levels.

        Args:
            channel_metadata: mapping keyed on frequency; each entry is a dict that includes a 'datetime' key
        """

        times, freqs = zip(*[[d["datetime"], k] for k, d in channel_metadata.items()])

        return pd.MultiIndex(
            levels=(times, freqs),
            codes=2 * [range(len(channel_metadata))],
            names=("datetime", "frequency"),
            sortorder=0,
        )

    @staticmethod
    @functools.lru_cache()
    def _pfp_index(
        pfp_sample_count: int, pvt_sample_count: int, iq_capture_duration_sec: float
    ) -> pd.MultiIndex:
        base = pd.RangeIndex(pfp_sample_count, name="Frame time elapsed (s)")
        return base * (iq_capture_duration_sec / pfp_sample_count / pvt_sample_count)

    @staticmethod
    @functools.lru_cache()
    def _pvt_index(
        pvt_sample_count: int, iq_capture_duration_sec: float
    ) -> pd.MultiIndex:
        base = pd.RangeIndex(pvt_sample_count, name="Capture time elapsed (s)")
        return base * (iq_capture_duration_sec / pvt_sample_count)

    @staticmethod
    @functools.lru_cache()
    def _psd_index(psd_sample_count, analysis_bandwidth_Hz) -> pd.MultiIndex:
        base = pd.RangeIndex(psd_sample_count, name="Baseband Frequency (Hz)")
        bin_center_offset = +analysis_bandwidth_Hz / psd_sample_count / 2
        return (
            base * (analysis_bandwidth_Hz / psd_sample_count)
            - analysis_bandwidth_Hz / 2
            + bin_center_offset
        )

    @staticmethod
    @functools.lru_cache()
    def _trace_index(trace_dicts: typing.Tuple[frozendict]) -> pd.MultiIndex:
        df = pd.DataFrame(trace_dicts)
        if 'detector' in df.columns:
            df.loc[:,'detector'].replace({'max': 'peak', 'mean': 'rms'},inplace=True)
        return pd.MultiIndex.from_frame(df)

    def unpack_arrays(self, data):
        """split the flat data vector into a dictionary of named traces"""
        split_inds = list(self.trace_starts.keys())[1:]
        traces = np.split(data, split_inds)

        trace_groups = defaultdict(lambda: defaultdict(list))
        for (trace_type, trace_name), trace in zip(self.trace_starts.values(), traces):
            trace_groups[trace_type][trace_name].append(np.array(trace))

        for trace_type in self.TABULAR_GROUPS:
            first_key, *_ = trace_groups[trace_type].keys()
            if len(trace_groups[trace_type]) == 1 and first_key is _UNLABELED_TRACE:
                # a single unlabeled trace
                first_value, *_ = trace_groups[trace_type].values()
                trace_groups[trace_type] = np.array(first_value)
            else:
                # dictionary of multiple traces
                trace_groups[trace_type] = {
                    name: np.array(v) for name, v in trace_groups[trace_type].items()
                }

        return dict(trace_groups, **self.meta)

    def unpack_dataframes(self, data):
        trace_groups = self.unpack_arrays(data)

        # tabular data
        capture_index = self._capture_index(self.meta["channel_metadata"])

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
                trace_index = self._trace_index(tuple(trace_groups[name].keys()))
                index = _cartesian_multiindex(capture_index, trace_index)

                # take the column definition from the first trace key, under the assumption
                # that they are the same for tabular data
                columns = self.trace_axes[name]
                frames[name] = pd.DataFrame(group_data, index=index, columns=columns)

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
                    group_data, index=capture_index, columns=columns
                )

        # channel metadata
        values = np.array(
            [list(v.values()) for v in self.meta["channel_metadata"].values()]
        )
        for v in self.meta["channel_metadata"].values():
            columns = v.keys()
            break

        channel_metadata = pd.DataFrame(values, index=capture_index, columns=columns)
        for col_name in ("datetime", "frequency"):
            if col_name in columns:
                channel_metadata.drop(col_name, axis=1, inplace=True)

        return dict(
            frames,
            channel_metadata=channel_metadata,
            sweep_metadata=pd.DataFrame([self.meta["sweep_metadata"]]),
            site_metadata=self.meta["site_metadata"]
        )


class _Loader_v1(_LoaderBase):
    TABULAR_GROUPS = "psd", "pvt", "pfp"

    # these are hard-coded since introspection of the trace info was difficult
    # and data were only generated in this structure
    TRACE_INFO = {
        "psd_max_power": _TI("psd", frozendict(capture_statistic="max")),
        "psd_mean_power": _TI("psd", frozendict(capture_statistic="mean")),
        "pvt_max_power": _TI("pvt", frozendict(detector="peak")),
        "pvt_mean_power": _TI("pvt", frozendict(detector="rms")),
        "pfp_rms_min_power": _TI(
            "pfp", frozendict(detector="rms", capture_statistic="min")
        ),
        "pfp_rms_max_power": _TI(
            "pfp", frozendict(detector="rms", capture_statistic="max")
        ),
        "pfp_rms_mean_power": _TI(
            "pfp", frozendict(detector="rms", capture_statistic="mean")
        ),
        "pfp_peak_min_power": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="min")
        ),
        "pfp_peak_max_power": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="max")
        ),
        "pfp_peak_mean_power": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="mean")
        ),
        "apd_p_pct": _TI("apd", "percentile"),
        "apd_a_dBm": _TI("apd", "sample_amplitude_dBm"),
    }

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta):
        if json_meta["timezone"] is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify on load (e.g., "America/New_York")'
            )

        self.trace_starts = {}
        self.trace_axes = {}

        channel_meta = defaultdict(dict)

        sample_rate = json_meta["global"]["core:sample_rate"]
        sweep_meta = dict(
            sample_rate=sample_rate,
            version=json_meta["global"]["core:version"],
            metadata_version="v0.1",
            calibration_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-sensor:calibration_datetime"],
                # json_meta["timezone"]
            ),
            schedule_name=json_meta["global"]["ntia-scos:schedule"]["name"],
            schedule_start_datetime=json_meta["global"]["ntia-scos:schedule"]["start"],
        )

        capture_fields = {d["core:sample_start"]: d for d in json_meta["captures"]}

        for annot in json_meta["annotations"]:
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
            site_metadata=frozendict(timezone=json_meta["timezone"])
        )

    def _update_frame_axes(self, annot, trace_type, sample_rate):
        if trace_type == "pfp":
            self.trace_axes[trace_type] = self._pfp_index(
                annot["core:sample_count"],
                400,  # TODO: pass from recent pvt trace data
                4.0,  # TODO: pass from recent pvt trace data
            )

        elif trace_type == "pvt":
            self.trace_axes[trace_type] = self._pvt_index(
                annot["core:sample_count"],
                annot["ntia-algorithm:number_of_samples"] / sample_rate,
            )

        elif trace_type == "psd":
            self.trace_axes[trace_type] = self._psd_index(
                annot["core:sample_count"], 10e6
            )

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
        "max_fft": _TI("psd", frozendict(capture_statistic="max")),
        "mean_fft": _TI("psd", frozendict(capture_statistic="mean")),
        "max_td_pwr_series": _TI("pvt", frozendict(detector="peak")),
        "mean_td_pwr_series": _TI("pvt", frozendict(detector="rms")),
        "min_rms_pfp": _TI("pfp", frozendict(detector="rms", capture_statistic="min")),
        "max_rms_pfp": _TI("pfp", frozendict(detector="rms", capture_statistic="max")),
        "mean_rms_pfp": _TI(
            "pfp", frozendict(detector="rms", capture_statistic="mean")
        ),
        "min_peak_pfp": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="min")
        ),
        "max_peak_pfp": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="max")
        ),
        "mean_peak_pfp": _TI(
            "pfp", frozendict(detector="peak", capture_statistic="mean")
        ),
        "apd_p": _TI("apd", "percentile"),
        "apd_a": _TI("apd", "sample_amplitude_dBm"),
    }

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta):
        self.trace_starts = {}
        self.trace_axes = {}

        channel_meta = defaultdict(dict)

        sweep_meta = dict(
            sample_rate=json_meta["global"]["core:sample_rate"],
            version=json_meta["global"]["core:version"],
            metadata_version=json_meta["global"]["core:extensions"]["ntia-nasctn-sea"],
            calibration_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-sensor:calibration_datetime"],
            ),
            schedule_name=json_meta["global"]["ntia-scos:schedule"]["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-scos:schedule"]["start"],
            ),
            schedule_interval=json_meta["global"]["ntia-scos:schedule"]["interval"],
            task=json_meta["global"]["ntia-scos:task"],
            diagnostics_datetime=_iso_to_datetime(
                json_meta["global"]["diagnostics"]["diagnostics_datetime"],
            ),
        )

        for v in json_meta["global"]["diagnostics"].values():
            if isinstance(v, dict):
                sweep_meta.update(v)

        for capture in json_meta["captures"]:
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
        self.trace_axes["pfp"] = self._pfp_index(
            capture["pfp_sample_count"],
            capture["td_pwr_sample_count"],
            capture["iq_capture_duration_msec"] / 1000.0,
        )
        self.trace_axes["pvt"] = self._pvt_index(
            capture["td_pwr_sample_count"], capture["iq_capture_duration_msec"] / 1000.0
        )
        self.trace_axes["psd"] = self._psd_index(capture["fft_sample_count"], 10e6)

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=sweep_meta,
            site_metadata=frozendict(timezone=json_meta["timezone"])
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
    def _get_trace_metadata(cls, data_products: frozendict):
        # this is cached since we expect data_products not to change very often
        offset_total = 0
        trace_offsets = []
        trace_labels = []
        for short_name, json_name in cls.TABULAR_GROUPS.items():
            dp_field = data_products[json_name]
            for trace_name in dp_field.get("detector", [None]):
                trace_offsets.append(offset_total)
                trace_meta = cls._parse_trace_string(short_name, trace_name)
                trace_labels.append(_TI(short_name, trace_meta))
                offset_total += dp_field["sample_count"]

        cls._trace_offsets = dict(zip(trace_offsets, trace_labels))

        return np.array(trace_offsets), trace_labels

    @functools.wraps(_LoaderBase.__init__)
    def __init__(self, json_meta):
        # get the vectors of offset indices of the each trace relative to the capture start
        data_products = json_meta["global"]["data_products"]
        trace_offsets, trace_labels = self._get_trace_metadata(data_products)

        # in v0.4 we can add apd here too
        self.trace_axes = {}
        sample_rate = json_meta["global"]["core:sample_rate"]
        capture_duration = json_meta["captures"][0]["iq_capture_duration_msec"] / 1000.0

        self.trace_axes["pfp"] = self._pfp_index(
            data_products["periodic_frame_power"]["sample_count"],
            data_products["time_series_power"]["sample_count"],
            capture_duration,
        )

        self.trace_axes["pvt"] = self._pvt_index(
            data_products["time_series_power"]["sample_count"], capture_duration
        )
        psd_samples = data_products["power_spectral_density"]["sample_count"]
        self.trace_axes["psd"] = self._psd_index(
            psd_samples,
            sample_rate
            * psd_samples
            / data_products["power_spectral_density"]["number_of_samples_in_fft"],
        )

        # cycle through the channels for metadata and data index maps
        channel_meta = {}
        trace_starts_keys = []
        trace_starts_values = []
        apd_start_offset = trace_offsets[-1] + (trace_offsets[-1] - trace_offsets[-2])
        apd_trace_info = [
            _TI(type="apd_p", metadata=frozendict({})),
            _TI(type="apd_a", metadata=frozendict({})),
        ]
        for capture, apd_length in zip(
            json_meta["captures"],
            data_products["amplitude_probability_distribution"]["sample_count"],
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
        sweep_meta = dict(
            sample_rate=json_meta["global"]["core:sample_rate"],
            version=json_meta["global"]["core:version"],
            metadata_version=json_meta["global"]["core:extensions"]["ntia-nasctn-sea"],
            calibration_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-sensor:calibration_datetime"],
            ),
            calibration_temperature_degC=json_meta["global"][
                "calibration_temperature_degC"
            ],
            schedule_name=json_meta["global"]["ntia-scos:schedule"]["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-scos:schedule"]["start"],
            ),
            schedule_interval=json_meta["global"]["ntia-scos:schedule"]["interval"],
            task=json_meta["global"]["ntia-scos:task"],
            diagnostics_datetime=_iso_to_datetime(
                json_meta["global"]["diagnostics"]["diagnostics_datetime"],
            ),
        )
        for v in json_meta["global"]["diagnostics"].values():
            if isinstance(v, dict):
                sweep_meta.update(v)

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=frozendict(sweep_meta),
            site_metadata=frozendict(timezone=json_meta["timezone"])
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
    def _get_trace_metadata(cls, data_products: frozendict):
        # this is cached since we expect data_products not to change very often
        offset_total = 0
        trace_offsets = []
        trace_labels = []
        FIXED_TRACE_NAME_SET = set(cls.TRACE_FIELD_NAMEMAP.keys())

        for short_name, json_name in cls.TABULAR_GROUPS.items():
            dp_field = data_products[json_name]

            for trace_obj in dp_field.get("traces", [_UNLABELED_TRACE]):
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
                trace_labels.append(_TI(short_name, trace_obj))
                offset_total += dp_field["length"]

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
    def __init__(self, json_meta):
        # get timezone from sensor location
        # if json_meta["timezone"] is None:
        #     # Overriden if tz is specified on init
        #     loc = json_meta["global"]["core:geolocation"]["coordinates"]
        #     tz = _unique_timezone_at(lng=loc[0], lat=loc[1])
        # else:
        #     tz = json_meta["timezone"]

        # get the vectors of offset indices of the each trace relative to the capture start
        data_products = json_meta["global"]["ntia-algorithm:data_products"]
        trace_offsets, trace_labels = self._get_trace_metadata(data_products)

        self.trace_axes = {}
        sample_rate = json_meta["global"]["core:sample_rate"]
        # (safely) assume all capture durations are identical
        capture_duration = json_meta["captures"][0]["ntia-sensor:duration"] / 1000.0

        self.trace_axes["pfp"] = self._pfp_index(
            data_products["periodic_frame_power"]["length"],
            data_products["time_series_power"]["length"],
            capture_duration,
        )
        self.trace_axes["pvt"] = self._pvt_index(
            data_products["time_series_power"]["length"], capture_duration
        )
        psd_samples = data_products["power_spectral_density"]["length"]
        self.trace_axes["psd"] = self._psd_index(
            psd_samples,
            sample_rate
            * psd_samples
            / data_products["power_spectral_density"]["samples"],
        )
        self.trace_axes["apd"] = self._apd_index(
            data_products["amplitude_probability_distribution"]["min_amplitude"],
            data_products["amplitude_probability_distribution"]["max_amplitude"],
            data_products["amplitude_probability_distribution"]["amplitude_bin_size"],
        )

        # cycle through the channels for metadata and data index maps
        channel_meta = {}
        trace_starts_keys = []
        trace_starts_values = []

        for capture in json_meta["captures"]:
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
            sample_rate=json_meta["global"]["core:sample_rate"],
            version=json_meta["global"]["core:version"],
            metadata_version=json_meta["global"]["core:extensions"][5]["version"],
            schedule_name=json_meta["global"]["ntia-scos:schedule"]["name"],
            schedule_start_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-scos:schedule"]["start"],
            ),
            schedule_interval=json_meta["global"]["ntia-scos:schedule"]["interval"],
            task=json_meta["global"]["ntia-scos:task"],
            diagnostics_datetime=_iso_to_datetime(
                json_meta["global"]["ntia-diagnostics:diagnostics"]["datetime"],
            ),
        )
        sweep_meta.update(json_meta["global"]["ntia-diagnostics:diagnostics"])

        self.meta = frozendict(
            channel_metadata=frozendict(channel_meta),
            sweep_metadata=frozendict(sweep_meta),
            site_metadata=frozendict(timezone=json_meta["timezone"])
        )


def _get_loader(json_meta: dict):
    """return a loader object appropriate to the metadata version"""
    if not hasattr(json_meta, "keys") or not callable(json_meta.keys):
        raise ValueError('argument "meta" must be dict-like')

    try:
        extensions = json_meta["global"]["core:extensions"]
    except KeyError as ex:
        raise IOError("invalid metadata dictionary structure") from ex

    version = None
    if isinstance(extensions, tuple):
        # v0.4+
        version = [e["version"] for e in extensions if e["name"] == "ntia-nasctn-sea"][
            0
        ]
    elif isinstance(extensions, frozendict):
        # v0.1, 0.2, or 0.3
        version = extensions.get("ntia-nasctn-sea", None)

    if version is None:
        return _Loader_v1(json_meta)
    elif version == "v0.2":
        return _Loader_v2(json_meta)
    elif version == "v0.3":
        return _Loader_v3(json_meta)
    elif version == "v0.4":
        return _Loader_v4(json_meta)
    else:
        raise ValueError(f'unrecognized format version "{version}"')


def _freeze_meta(pairs):
    return frozendict(
        [(k, tuple(v) if isinstance(v, list) else v) for k, v in pairs.items()]
    )


def read_seamf(
    file, force_loader_cls=False, container_cls=pd.DataFrame, hash_check=True, tz=None
) -> dict:
    """unpacks a sensor data file into a dictionary of numpy or pandas objects

    When `force_loader_cls` is False, the loader is selected automatically based on the
    metadata version. Other valid values are `None` (to return metadata dictionary and a flat numpy data array)
    or `bytes` (for the metadata dictionary and data in bytes form) or one of the _Loader_v* classes.

    The supported types of container object are pandas.DataFrame and numpy.ndarray.

    When hash_check evaluates as True, the data contents are compared against the SHA512 hash
    in the metadata file.
    """
    if isinstance(file, (str, Path)):
        kws = {"name": file}
    else:
        kws = {"fileobj": file}

    with TarFile(**kws) as tar_fd:
        tar_names = tar_fd.getnames()

        # meta is plain json
        meta_name = [n for n in tar_names if n.endswith(".sigmf-meta")][0]
        meta = json.loads(
            tar_fd.extractfile(meta_name).read(), object_hook=_freeze_meta
        )

        data_name = [n for n in tar_names if n.endswith(".sigmf-data")][0]
        lzma_data = tar_fd.extractfile(data_name).read()

    if hash_check:
        data_hash = hashlib.sha512(lzma_data).hexdigest()
        if data_hash != meta["global"]["core:sha512"]:
            raise IOError("seamf file data failed sha512 hash check")

    if tz is None:
        # try to automatically update time zone from metadata
        loc = (
            meta.get("global", {}).get("core:geolocation", {}).get("coordinates", None)
        )

        if loc is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify, e.g., tz="America/New_York"'
            )
        else:
            tz = _unique_timezone_at(lng=loc[0], lat=loc[1])

    meta = frozendict(meta, timezone=tz)

    # the duration of this operation is dominated by lzma.decompress, not disk access or numpy
    # (on a 2020-vintage laptop with an SSD)
    byte_data = lzma.decompress(lzma_data)
    if isinstance(force_loader_cls, type) and issubclass(force_loader_cls, bytes):
        return byte_data, meta

    data = np.frombuffer(byte_data, dtype="half")

    # pick the loader
    if force_loader_cls is None:
        return data, meta
    elif force_loader_cls == False:
        loader = _get_loader(meta)
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
        return loader.unpack_dataframes(data)
    elif issubclass(container_cls, np.ndarray):
        return loader.unpack_arrays(data)
    else:
        raise TypeError('invalid "container_cls"')


def read_seamf_meta(file, tz=None):
    if isinstance(file, (str, Path)):
        kws = {"name": file}
    else:
        kws = {"fileobj": file}

    with TarFile(**kws) as tar_fd:
        name = tar_fd.getnames()[0]

        # meta is plain json
        meta_name = "/".join((name, name + ".sigmf-meta"))
        meta = json.loads(
            tar_fd.extractfile(meta_name).read(), object_hook=_freeze_meta
        )

    if tz is None:
        # try to automatically update time zone from metadata
        loc = (
            meta.get("global", {}).get("core:geolocation", {}).get("coordinates", None)
        )

        if loc is None:
            raise ValueError(
                'could not automatically identify time zone, need to specify, e.g., tz="America/New_York"'
            )
        else:
            tz = _unique_timezone_at(lng=loc[0], lat=loc[1])

    meta = frozendict(meta, timezone=tz)

    return meta

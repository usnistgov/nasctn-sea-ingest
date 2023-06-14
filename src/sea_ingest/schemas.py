import msgspec
import typing
from frozendict import frozendict
from pathlib import Path


class SchemaBase(msgspec.Struct, kw_only=True):
    _GLOBAL_KEYS = [
        "core:version",
        "core:extensions",
        "core:geolocation",
        "core:datatype",
        "core:sample_rate",
        "core:num_channels",
        "ntia-sensor:calibration_datetime",
        "ntia-scos:task",
        "ntia-scos:schedule",
        "ntia-sensor:sensor",
        "ntia-algorithm:digital_filters",
        "ntia-nasctn-sea:max_of_max_channel_powers",
        "ntia-nasctn-sea:median_of_mean_channel_powers",
        "core:sha512",
    ]

    _GLOBAL_KEYS_RENAME = {k.rsplit(":", 1)[1]: k for k in _GLOBAL_KEYS}

    @classmethod
    def fromfile(cls, path_or_buf):
        if isinstance(path_or_buf, (str, Path)):
            with open(path_or_buf, "rb") as fb:
                raw = fb.read()
        else:
            raw = path_or_buf.read()

        return cls.fromstr(raw)

    @classmethod
    def fromstr(cls, json_str):
        def dec_hook(type_, obj):
            return type_(obj)

        return msgspec.json.decode(json_str, type=cls, dec_hook=dec_hook)

    timezone: typing.Union[str, None] = None


class VersionInfo(SchemaBase, frozen=True):
    """a minimal schema to quickly load version information from any version of the json metadata"""

    REMAP = {"version": "core:version", "extensions": "core:extensions"}

    class Global(msgspec.Struct, rename=REMAP, frozen=True):
        version: str
        extensions: typing.Union[typing.Tuple[frozendict, ...], dict]

    global_: Global = msgspec.field(name="global")


class GlobalSchemaBase(
    msgspec.Struct, kw_only=True, frozen=True, rename=SchemaBase._GLOBAL_KEYS_RENAME
):
    version: str
    datatype: str
    extensions: typing.Union[typing.Tuple[frozendict, ...], dict]
    sample_rate: float
    sha512: str

    data_products: frozendict = frozendict()

    task: typing.Union[int, None] = None
    schedule: frozendict = frozendict()
    sensor: frozendict = frozendict()
    num_channels: int = 15
    geolocation: frozendict = frozendict()


class MetadataPre0_4(SchemaBase, kw_only=True):
    """a general-purpose schema for the 'SeaMF' metadata for v0.3 and earlier"""

    class Global(
        GlobalSchemaBase, rename=SchemaBase._GLOBAL_KEYS_RENAME, frozen=True
    ):
        class DataProducts(msgspec.Struct, frozen=True):
            class PSD(msgspec.Struct, frozen=True):
                detector: typing.Tuple[str, ...]
                sample_count: int
                equivalent_noise_bandwidth: float
                number_of_samples_in_fft: int
                number_of_ffts: int
                units: str
                window: str

            class PFP(msgspec.Struct, frozen=True):
                detector: typing.Tuple[str, ...]
                sample_count: int
                units: str

            class PVT(msgspec.Struct, frozen=True):
                detector: typing.Tuple[str, ...]
                sample_count: int
                number_of_samples: int
                units: str

            class APD(msgspec.Struct, frozen=True):
                sample_count: typing.Tuple[int, ...]
                number_of_samples: int
                probability_units: str
                power_bin_size: float
                units: str

            # digital_filter: frozendict = frozendict()
            reference: typing.Union[str, None] = None

            power_spectral_density: typing.Union[PSD, None] = None
            periodic_frame_power: typing.Union[PFP, None] = None
            time_series_power: typing.Union[PVT, None] = None
            amplitude_probability_distribution: typing.Union[APD, None] = None

        extensions: frozendict
        data_products: typing.Union[DataProducts, None] = None
        diagnostics: frozendict = frozendict()

        calibration_datetime: typing.Union[str, None] = None
        calibration_temperature_degC: typing.Union[float, None] = None

        # digital_filters - TODO
        # max_of_max_channel_powers - TODO
        # median_of_mean_channel_powers - TODO

    global_: Global = msgspec.field(name="global")
    annotations: typing.Tuple[frozendict, ...]
    captures: typing.Tuple[frozendict, ...]
    timezone: typing.Union[str, None] = None


class MetadataSince0_4(SchemaBase, kw_only=True):
    """a general-purpose schema for the 'SeaMF' metadata since v0.4"""

    _GLOBAL_KEYS = SchemaBase._GLOBAL_KEYS + [
        "ntia-diagnostics:diagnostics",
        "ntia-algorithm:data_products",
    ]

    _GLOBAL_KEYS_RENAME = {k.rsplit(":", 1)[1]: k for k in _GLOBAL_KEYS}

    class Global(
        GlobalSchemaBase, kw_only=True, rename=_GLOBAL_KEYS_RENAME, frozen=True
    ):
        class DataProducts(msgspec.Struct, frozen=True):
            class PSD(msgspec.Struct, frozen=True):
                traces: typing.Tuple[frozendict, ...]
                length: int
                equivalent_noise_bandwidth: float
                samples: int
                ffts: int
                units: str
                window: str

            class PFP(msgspec.Struct, frozen=True):
                traces: typing.Tuple[frozendict, ...]
                length: int
                units: str

            class PVT(msgspec.Struct, frozen=True):
                traces: typing.Tuple[frozendict, ...]
                length: int
                samples: int
                units: str

            class APD(msgspec.Struct, frozen=True):
                length: int
                samples: int
                probability_units: str
                amplitude_bin_size: float
                min_amplitude: float
                max_amplitude: float

            digital_filter: str = None
            reference: typing.Union[str, None] = None

            power_spectral_density: typing.Union[PSD, None] = None
            periodic_frame_power: typing.Union[PFP, None] = None
            time_series_power: typing.Union[PVT, None] = None
            amplitude_probability_distribution: typing.Union[APD, None] = None

        extensions: typing.Tuple[frozendict, ...]
        data_products: DataProducts
        diagnostics: frozendict = frozendict()

        # digital_filters - TODO
        # max_of_max_channel_powers - TODO
        # median_of_mean_channel_powers - TODO

    global_: Global = msgspec.field(name="global")
    annotations: typing.Tuple[frozendict, ...]
    captures: typing.Tuple[frozendict, ...]

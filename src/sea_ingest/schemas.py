import msgspec
from typing import Optional, Tuple, Union
from frozendict import frozendict
from pathlib import Path


class SchemaBase(msgspec.Struct, kw_only=True):
    """
    Base class for schemas.

    Attributes:
        _GLOBAL_KEYS (list): List of global keys.
        _GLOBAL_KEYS_RENAME (dict): Dictionary mapping short keys to full keys.
        timezone (Union[str, None]): Timezone information.
    """
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
        """
        Creates an instance of the schema from a file.

        Args:
            path_or_buf (Union[str, Path, file-like]): Path to the file or file-like object.

        Returns:
            SchemaBase: An instance of the schema.
        """
        if isinstance(path_or_buf, (str, Path)):
            with open(path_or_buf, "rb") as fb:
                raw = fb.read()
        else:
            raw = path_or_buf.read()

        return cls.fromstr(raw)

    @classmethod
    def fromstr(cls, json_str):
        """
        Creates an instance of the schema from a JSON string.

        Args:
            json_str (str): JSON string.

        Returns:
            SchemaBase: An instance of the schema.
        """
        def dec_hook(type_, obj):
            return type_(obj)

        return msgspec.json.decode(json_str, type=cls, dec_hook=dec_hook)

    timezone: Union[str, None] = None


class VersionInfo(SchemaBase, frozen=True):
    """
    A minimal schema to quickly load version information from any version of the JSON metadata.
    """

    REMAP = {"version": "core:version", "extensions": "core:extensions"}

    class Global(msgspec.Struct, rename=REMAP, frozen=True):
        version: str
        extensions: Union[Tuple[frozendict, ...], dict]

    global_: Global = msgspec.field(name="global")


class GlobalSchemaBase(
    msgspec.Struct, kw_only=True, frozen=True, rename=SchemaBase._GLOBAL_KEYS_RENAME
):
    version: str
    datatype: str
    extensions: Union[Tuple[frozendict, ...], dict]
    sample_rate: float
    sha512: str

    data_products: frozendict = frozendict()

    task: Union[int, None] = None
    schedule: frozendict = frozendict()
    sensor: frozendict = frozendict()
    num_channels: int = 15
    geolocation: frozendict = frozendict()


class MetadataPre0_4(SchemaBase, kw_only=True):
    """a general-purpose schema for the 'SeaMF' metadata for v0.3 and earlier"""

    class Global(GlobalSchemaBase, rename=SchemaBase._GLOBAL_KEYS_RENAME, frozen=True):
        class DataProducts(msgspec.Struct, frozen=True):
            class PSD(msgspec.Struct, frozen=True):
                detector: Tuple[str, ...]
                sample_count: int
                equivalent_noise_bandwidth: float
                number_of_samples_in_fft: int
                number_of_ffts: int
                units: str
                window: str

            class PFP(msgspec.Struct, frozen=True):
                detector: Tuple[str, ...]
                sample_count: int
                units: str

            class PVT(msgspec.Struct, frozen=True):
                detector: Tuple[str, ...]
                sample_count: int
                number_of_samples: int
                units: str

            class APD(msgspec.Struct, frozen=True):
                sample_count: Tuple[int, ...]
                number_of_samples: int
                probability_units: str
                power_bin_size: float
                units: str

            # digital_filter: frozendict = frozendict()
            reference: Union[str, None] = None

            power_spectral_density: Union[PSD, None] = None
            periodic_frame_power: Union[PFP, None] = None
            time_series_power: Union[PVT, None] = None
            amplitude_probability_distribution: Union[APD, None] = None

        extensions: frozendict
        data_products: Union[DataProducts, None] = None
        diagnostics: frozendict = frozendict()

        calibration_datetime: Union[str, None] = None
        calibration_temperature_degC: Union[float, None] = None

        # digital_filters - TODO
        # max_of_max_channel_powers - TODO
        # median_of_mean_channel_powers - TODO

    global_: Global = msgspec.field(name="global")
    annotations: Tuple[frozendict, ...]
    captures: Tuple[frozendict, ...]
    timezone: Union[str, None] = None


class Metadata0_4(SchemaBase, kw_only=True):
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
                traces: Tuple[frozendict, ...]
                length: int
                equivalent_noise_bandwidth: float
                samples: int
                ffts: int
                units: str
                window: str

            class PFP(msgspec.Struct, frozen=True):
                traces: Tuple[frozendict, ...]
                length: int
                units: str

            class PVT(msgspec.Struct, frozen=True):
                traces: Tuple[frozendict, ...]
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
            reference: Union[str, None] = None

            power_spectral_density: Union[PSD, None] = None
            periodic_frame_power: Union[PFP, None] = None
            time_series_power: Union[PVT, None] = None
            amplitude_probability_distribution: Union[APD, None] = None

        extensions: Tuple[frozendict, ...]
        data_products: DataProducts
        diagnostics: frozendict = frozendict()

        # digital_filters - TODO
        # max_of_max_channel_powers - TODO
        # median_of_mean_channel_powers - TODO

    global_: Global = msgspec.field(name="global")
    annotations: Tuple[frozendict, ...]
    captures: Tuple[frozendict, ...]


class Metadata0_5(SchemaBase, kw_only=True):
    _GLOBAL_KEYS = SchemaBase._GLOBAL_KEYS + [
        "core:recorder",
        "ntia-scos:schedule",
        "ntia-scos:action",
        "ntia-core:classification",
        "ntia-algorithm:processing",
        "ntia-algorithm:processing_info",
        "ntia-algorithm:data_products",
        "ntia-diagnostics:diagnostics",
    ]
    _GLOBAL_KEYS.pop(_GLOBAL_KEYS.index("ntia-sensor:calibration_datetime"))
    _GLOBAL_KEYS.pop(_GLOBAL_KEYS.index("ntia-algorithm:digital_filters"))

    _GLOBAL_KEYS_RENAME = {k.rsplit(":", 1)[1]: k for k in _GLOBAL_KEYS}

    class Global(
        GlobalSchemaBase, kw_only=True, rename=_GLOBAL_KEYS_RENAME, frozen=True
    ):
        class Graph(msgspec.Struct, frozen=True):
            name: str
            series: Optional[Tuple[str, ...]] = None
            length: Optional[int] = None
            x_units: Optional[str] = None
            x_axis: Optional[Tuple[Union[int, float, str], ...]] = None
            x_start: Optional[Tuple[float, ...]] = None
            x_stop: Optional[Tuple[float, ...]] = None
            x_step: Optional[Tuple[float, ...]] = None
            y_units: Optional[str] = None
            y_axis: Optional[Tuple[Union[int, float, str], ...]] = None
            y_start: Optional[Tuple[float, ...]] = None
            y_stop: Optional[Tuple[float, ...]] = None
            y_step: Optional[Tuple[float, ...]] = None
            processing: Optional[Tuple[str, ...]] = None
            reference: Optional[str] = None
            description: Optional[str] = None

        class DigitalFilter(msgspec.Struct, frozen=True, tag="DigitalFilter"):
            id: str
            filter_type: str
            feedforward_coefficients: Optional[Tuple[float, ...]] = None
            feedback_coefficients: Optional[Tuple[float, ...]] = None
            attenuation_cutoff: Optional[float] = None
            frequency_cutoff: Optional[float] = None
            description: Optional[str] = None

        class DFT(msgspec.Struct, frozen=True, tag="DFT"):
            id: str
            equivalent_noise_bandwidth: float
            samples: int
            dfts: int
            window: str
            baseband: bool
            description: Optional[str] = None

        extensions: Tuple[frozendict, ...]
        processing: Tuple[str]
        processing_info: Tuple[Union[DigitalFilter, DFT], ...]
        data_products: Tuple[Graph, ...]
        max_of_max_channel_powers: Tuple[float, ...]
        median_of_mean_channel_powers: Tuple[float, ...]
        diagnostics: frozendict = frozendict()

    global_: Global = msgspec.field(name="global")
    annotations: Tuple[frozendict, ...]
    captures: Tuple[frozendict, ...]


class Metadata0_6(SchemaBase, kw_only=True):
    _GLOBAL_KEYS = Metadata0_5._GLOBAL_KEYS + [
        "ntia-nasctn-sea:mean_channel_powers",
        "ntia-nasctn-sea:median_channel_powers",
    ]
    _GLOBAL_KEYS_RENAME = {k.rsplit(":", 1)[1]: k for k in _GLOBAL_KEYS}

    # The following is identical to v0.5, with mean_channel_powers and
    # median_channel_powers added.

    # Other metadata changes in this version only affect diagnostics and sensor hardware
    # information, which are not loaded using a msgspect Struct by this package at present.
    # The existing loading of the global diagnostics and ntia-sensor:sensor objects as
    # dictionaries will still work, but will have additional keys compared to v0.5.0.
    class Global(
        GlobalSchemaBase, kw_only=True, rename=_GLOBAL_KEYS_RENAME, frozen=True
    ):
        class Graph(msgspec.Struct, frozen=True):
            name: str
            series: Optional[Tuple[str, ...]] = None
            length: Optional[int] = None
            x_units: Optional[str] = None
            x_axis: Optional[Tuple[Union[int, float, str], ...]] = None
            x_start: Optional[Tuple[float, ...]] = None
            x_stop: Optional[Tuple[float, ...]] = None
            x_step: Optional[Tuple[float, ...]] = None
            y_units: Optional[str] = None
            y_axis: Optional[Tuple[Union[int, float, str], ...]] = None
            y_start: Optional[Tuple[float, ...]] = None
            y_stop: Optional[Tuple[float, ...]] = None
            y_step: Optional[Tuple[float, ...]] = None
            processing: Optional[Tuple[str, ...]] = None
            reference: Optional[str] = None
            description: Optional[str] = None

        class DigitalFilter(msgspec.Struct, frozen=True, tag="DigitalFilter"):
            id: str
            filter_type: str
            feedforward_coefficients: Optional[Tuple[float, ...]] = None
            feedback_coefficients: Optional[Tuple[float, ...]] = None
            attenuation_cutoff: Optional[float] = None
            frequency_cutoff: Optional[float] = None
            description: Optional[str] = None

        class DFT(msgspec.Struct, frozen=True, tag="DFT"):
            id: str
            equivalent_noise_bandwidth: float
            samples: int
            dfts: int
            window: str
            baseband: bool
            description: Optional[str] = None

        extensions: Tuple[frozendict, ...]
        processing: Tuple[str]
        processing_info: Tuple[Union[DigitalFilter, DFT], ...]
        data_products: Tuple[Graph, ...]
        max_of_max_channel_powers: Tuple[float, ...]
        median_of_mean_channel_powers: Tuple[float, ...]
        mean_channel_powers: Tuple[float, ...]
        median_channel_powers: Tuple[float, ...]
        diagnostics: frozendict = frozendict()

    global_: Global = msgspec.field(name="global")
    annotations: Tuple[frozendict, ...]
    captures: Tuple[frozendict, ...]

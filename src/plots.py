import matplotlib as mpl
from matplotlib.ticker import EngFormatter, MultipleLocator
from matplotlib.dates import DateFormatter
from matplotlib import rc
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seamf import trace


def transposed_legend(ax, *args, **kws):
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    ncol = kws.get("ncol", 1)
    handles, labels = ax.get_legend_handles_labels()
    if len(args) > 0:
        handles = args[0]
        args = args[1:]
    if len(args) > 0:
        labels = args[1]
        args = args[1:]

    if ncol > 1:
        handles = flip(handles, ncol)
        labels = flip(labels, ncol)

    return ax.legend(handles, labels, *args, **kws)


def nearest_datetimes(df, targets):
    return df.index[df.index.get_indexer(list(targets), method="nearest")]


def plot_pvt_detail(day, freq, detail_datetimes, legend_ax_index=0):
    fig, axs = plt.subplots(
        nrows=len(detail_datetimes),
        figsize=(6 + 2 / 3, 4),
        layout="constrained",
        sharey=True,
    )

    detail_label_style = dict(
        fontdict={"size": 12},
        bbox=dict(boxstyle="square", facecolor="white"),
        color="black",
    )

    # detail view
    for (label, datetime), ax in zip(detail_datetimes.items(), axs[0:]):
        # remaining index levels will be ('capture_statistic', 'detector')
        pvt_traces = trace(dfs=day, type="pvt", datetime=datetime, frequency=freq)
        ax.text(
            0.01,
            0.97,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            **detail_label_style,
        )

        pvt_traces.T.iloc[:, ::-1].plot(ax=ax, legend=False)

        ax.get_shared_x_axes().join(*axs[-len(detail_datetimes) :])
        ax.grid(True)

    axs[legend_ax_index].legend(["RMS detector", "Peak detector"], loc="best", ncol=2)

    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.set_xlabel("Capture time elapsed")
    fig.supylabel("Channel power (dBm/10 MHz)", fontsize=10)
    return fig


def plot_psd_detail(day, detail_trace_targets):
    detail_label_style = dict(
        fontdict={"size": 12},
        bbox=dict(boxstyle="square", facecolor="white"),
        color="black",
    )

    f0_datetimes = nearest_datetimes(
        trace(day, "psd", frequency=3.555e9, capture_statistic="max"),
        detail_trace_targets.values(),
    )

    psd = trace(day, "psd")

    fig, axs = plt.subplots(
        nrows=len(f0_datetimes),
        figsize=(6 + 2 / 3, 4),
        layout="constrained",
        sharey=True,
        sharex=True,
    )
    for datetime, label, ax in zip(f0_datetimes, detail_trace_targets.keys(), axs):
        Nfreqs = psd.index.levels[psd.index.names.index("frequency")].shape[0]
        Ntraces = psd.loc[datetime].shape[0]
        sweep = psd.loc[datetime:].iloc[: Nfreqs * Ntraces]

        sweep = sweep.reset_index("datetime", drop=True).unstack("frequency").T
        sweep.index = pd.Index(
            np.array(tuple(sweep.index.values)).sum(axis=1), name="Frequency"
        )
        sweep.sort_index().iloc[:, ::-1].plot(ax=ax, lw=1, legend=False)
        ax.grid(True, which="both", axis="x")
        ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
        ax.xaxis.set_minor_locator(MultipleLocator(10e6))

        ax.text(
            0.01,
            0.97,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            **detail_label_style,
        )

        if ax == axs[0]:
            fig.legend(ncol=2, loc="lower right")

    fig.supylabel("Power spectral density (dBm/Hz)")

    return fig


def plot_pfp_span_with_detail(day, freq, pfp_indicators, span, detail_datetimes):
    detail_label_style = dict(
        fontdict={"size": 12},
        bbox=dict(boxstyle="square", facecolor="white"),
        color="black",
    )

    fig, axs = plt.subplots(
        nrows=len(detail_datetimes) + 1, figsize=(6 + 2 / 3, 6), layout="constrained"
    )

    # mid-scale view
    (pfp_indicators.loc[span].plot(ax=axs[0], marker="s", ls=":", lw=1, legend=False))
    for label, index in detail_datetimes.items():
        axs[0].axvline(index, ls=":", color="k")
        axs[0].text(index, -55, label, va="center", ha="center", **detail_label_style)
    axs[0].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    axs[0].set_xlabel("Local time")
    axs[0].set_ylim([None, pfp_indicators["Frame max (peak detect)"].max() + 18])
    axs[0].grid(True)
    fig.legend(ncol=3, columnspacing=0.5, loc="upper right", framealpha=1)
    for t in axs[0].xaxis.get_ticklabels():
        t.set_rotation(0)
        t.set_horizontalalignment("center")

    # detail view
    for (label, datetime), ax in zip(detail_datetimes.items(), axs[1:]):
        # remaining index levels will be ('capture_statistic', 'detector')
        global pfp_traces
        pfp_traces = trace(dfs=day, type="pfp", datetime=datetime, frequency=freq)
        ax.text(
            0.01,
            0.97,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            **detail_label_style,
        )

        for detector, color in dict(RMS="C0", peak="C1").items():
            (
                pfp_traces.loc["mean", detector.lower()].plot(
                    color=color, ax=ax, lw=1, label=f"{detector} detect: mean"
                )
            )

            ax.fill_between(
                pfp_traces.columns.values.astype("float32"),
                pfp_traces.loc["min", detector.lower()],
                pfp_traces.loc["max", detector.lower()],
                color=color,
                alpha=0.25,
                lw=0,
                label=f"extrema",
                rasterized=True,
            )

        pfp_ind = pfp_indicators.loc[datetime].loc[
            ["Frame median (RMS detect)", "Frame max (peak detect)"]
        ]

        for power, color in zip(pfp_ind, ["C0", "C1"]):
            ax.axhline(power, color=color, ls=":")

        # ax.set_xlim([0,10.3e-3])

        ax.get_shared_x_axes().join(*axs[-len(detail_datetimes) :])
        ax.get_shared_y_axes().join(*axs[-len(detail_datetimes) :])
        ax.grid(True)

        if label == list(detail_datetimes.keys())[0]:
            transposed_legend(
                ax, markerfirst=False, columnspacing=0.5, ncol=4, loc="upper right"
            )

    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.set_xlabel("Time elapsed each reference frame")
    fig.supylabel("Channel power (dBm/10 MHz)", fontsize=10)

    return fig

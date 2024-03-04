import os

from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

"""
Create and return plots for various analytics.
"""


def plot_history(events):
    """
    Plot the events from a history.
    args:
        events: pandas DataFrame
    return:
        plot: matplotlib.axes.Axes
    """
    plot = "NYI"

    return plot


def plot_groupby(all_events, group_keys, score_dimensions):
    keys = group_keys + ["Reward"] + score_dimensions
    data = pd.DataFrame(columns=keys)
    for events in all_events:
        if len(data) == 0:
            data = events[
                keys
            ].copy()  # needed to avoid Pandas complaining about empty dataframe
        else:
            data = pd.concat([data, events[keys]])

    data["Reward"] = data["Reward"].astype(float)
    data[score_dimensions] = data[score_dimensions].astype(float)
    data["Score"] = data[score_dimensions].sum(axis=1)

    plot_data = data.groupby(group_keys).mean()

    return plot_data


def plot_performance(all_events, num_train_episodes, num_train_pipeline_cycles, score_dimensions, save_path: Optional[str], title: Optional[str] = ""):
    """
    Plot performance between rewards and scores.
    Accepts a list of event records from which a boxplot is done.
    TODO: further consideration should be had on *what* to average over.
    """

    if all_events[0].columns[-1] == "Score":  # old AIntelope environment
        score_dimensions = ["Score"]

    #train_events = [events[events["Episode"] < num_train_episodes] for events in all_events]
    #test_events = [events[events["Episode"] >= num_train_episodes] for events in all_events]
    train_events = [events[events["Pipeline cycle"] < num_train_pipeline_cycles] for events in all_events]
    test_events = [events[events["Pipeline cycle"] >= num_train_pipeline_cycles] for events in all_events]

    #plot_data1 = (
    #    "Episode",
    #    plot_groupby(all_events, ["Run_id", "Episode", "Agent_id"], score_dimensions),
    #)
    plot_data1 = (
        "Pipeline cycle",
        plot_groupby(all_events, ["Run_id", "Pipeline cycle", "Agent_id"], score_dimensions),
    )
    plot_data2 = (
        "Train Step",
        plot_groupby(train_events, ["Run_id", "Step", "Agent_id"], score_dimensions),
    )
    plot_data3 = (
        "Test Step",
        plot_groupby(test_events, ["Run_id", "Step", "Agent_id"], score_dimensions),
    )
    plot_datas = [plot_data1, plot_data2, plot_data3]

    # fig = plt.figure()
    fig, subplots = plt.subplots(3)

    linewidth = 0.75    # TODO: config

    for index, subplot in enumerate(subplots):
        (plot_label, plot_data) = plot_datas[index]

        subplot.plot(plot_data["Reward"].to_numpy(), label="Reward", linewidth=linewidth)
        subplot.plot(plot_data["Score"].to_numpy(), label="Score", linewidth=linewidth)
        for score_dimension in score_dimensions:
            subplot.plot(plot_data[score_dimension].to_numpy(), label=score_dimension, linewidth=linewidth)

        subplot.set_title((title + " by " + plot_label).strip())
        subplot.set(xlabel=plot_label, ylabel="Mean Reward")
        subplot.legend()

    if save_path:
        save_plot(fig, save_path)

    # enable this code if you want the plot to open automatically
    plt.ion()
    fig.show()
    plt.draw()
    # TODO: use multithreading for rendering the plot
    plt.pause(60)  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

    return fig


def plot_heatmap(agent, env):
    """
    Plot how the agent sees the values in an environment.
    """
    plot = "NYI"
    return plot


def save_plot(plot, save_path):
    """
    Save plot to file. Will get deprecated if nothing else comes here.
    """
    # save in two formats. SVG is good for resizing during viewing
    plot.savefig(save_path + ".png", dpi=200)
    plot.savefig(save_path + ".svg", dpi=200)


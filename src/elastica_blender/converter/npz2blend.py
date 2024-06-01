import os
import sys
from pathlib import Path

import click
import numpy as np

import bsr


def confirm_pyelastica_npz_structure(path: str, tags: list[str] | None = None):
    data = np.load(path)
    keys = list(data.keys())

    # Check if all keys match one of the following patterns
    required_key_pattern = ["time"]
    if tags is not None:
        for tag in tags:
            required_key_pattern.append(tag + "_position_history")
            required_key_pattern.append(tag + "_radius_history")
    else:
        required_key_pattern.append("position_history")
        required_key_pattern.append("radius_history")

    for key in keys:
        if not any([pattern in key for pattern in required_key_pattern]):
            raise KeyError(
                f"Key {key} does not match any of the required patterns."
            )


def construct_blender_file(
    path: str | Path, output: str | Path, tags: list[str] | None, fps: int = -1
):
    """
    Read npz file containing the position and radius data of multiple elatica rods.
    The shape of time is [n_timesteps]
    The shape of position is [n_rods, n_timesteps, 3, n_nodes]
    The shape of radius is [n_rods, n_timesteps, n_nodes-1]
    """
    confirm_pyelastica_npz_structure(path, tags)
    data = np.load(path)

    time = data["time"]
    start_time, end_time = time[0], time[-1]
    if fps == -1:
        probe_time = np.arange(time.size)
    else:  # pragma: no cover
        NotImplementedError("Not implemented yet.")

    if tags is None:
        position_history = data["position_history"]
        radius_history = data["radius_history"]
        num_rods = position_history.shape[0]
        num_elements = position_history.shape[3]
        rods = bsr.create_rod_collection(num_rods, num_elements)
        rods.update_history(
            keytimes=time, position=position_history, radius=radius_history
        )
    else:
        for tag in tags:
            position_history = data[tag + "_position_history"]
            radius_history = data[tag + "_radius_history"]
            num_rods = position_history.shape[0]
            num_elements = position_history.shape[3]
            rods = bsr.create_rod_collection(num_rods, num_elements, tag)
            rods.update_history(
                keytimes=time, position=position_history, radius=radius_history
            )

    bsr.save(output)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to the npz file containing the data.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    help="Path to the output blend file.",
)
@click.option(
    "--tags",
    "-t",
    type=str,
    multiple=True,
    default=None,
    help="Tags to identify rods.",
)
@click.option(
    "--fps",
    type=int,
    default=-1,
    help="Frames per second. By default, the value is set to -1, which means every frame is saved.",
)
def main(path, output, tags, fps):  # pragma: no cover
    construct_blender_file(path, output, tags, fps)

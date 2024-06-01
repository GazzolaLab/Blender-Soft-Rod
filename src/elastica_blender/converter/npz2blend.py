import os
import sys
from pathlib import Path

import click


def func(path: str, Path):
    pass


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(file_okay=True, dir_okay=False),
)
def main(path):
    func(path)

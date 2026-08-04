# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import nes


def plot_cli(
    input_file: str,
    variable: str,
    level: int,
    time_step: int,
    method: str = "colormesh",
    animation_method: str = "colormesh",
    cmap: str = "viridis",
    save: bool = False,
    scale: str = "linear",
    threshold: float = 0.0,
    percentile_clip: tuple = (),
) -> None:
    """
    Plot data from NetCDF files.

    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file.
    variable : str
        Name of the variable to plot.
    level : int
        Vertical level index to plot.
    time_step : int
        Time step index to plot.
    method : str, optional
        Plotting method to use ("colormesh", "contour", "histogram") (default: "colormesh").
    animation_method : str, optional
        Animation method to use ("colormesh", "contour") (default: "colormesh").
    cmap : str, optional
        Colormap to use for plotting (default: "viridis").
    save : bool, optional
        Whether to save the plot instead of displaying it (default: False).
    scale : str, optional
        What type of scale to use for the plot ("linear", "log", "power") (default: "linear").
    threshold : float, optional
        Threshold value for filtering data (default: 0.0).
    percentile_clip : tuple, optional
        Percentile clipping range (e.g., (5, 95)) (default: ()).
    """
    a = nes.open_netcdf(input_file)
    a.load()
    if method == "animation":
        a.plot(
            variable,
            level,
            time_step,
            method=method,
            animation_method=animation_method,
            cmap=cmap,
            scale=scale,
            threshold=threshold,
            percentile_clip=percentile_clip,
        )
    else:
        a.plot(
            variable,
            level,
            time_step,
            method=method,
            cmap=cmap,
            save=save,
            scale=scale,
            threshold=threshold,
            percentile_clip=percentile_clip,
        )
    return None

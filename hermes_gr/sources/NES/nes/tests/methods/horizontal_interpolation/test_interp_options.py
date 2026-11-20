# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test of the conservative interpolation options, with flux=False and flux=True.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4]; dst: 8x8 grid of 0.5 deg cells (unless stated).
- Weight matrix file: the result with a weight matrix created and saved (weight_matrix_path), then reloaded
  from that file, equals the in-memory result (no proxy and a points proxy). Regression: the reload used to
  drop the source cell 0.
- Weight matrix object: only_create_wm=True and its reuse through wm= give the in memory result.
- Several time steps and levels with different values: conserved per time step and level, with the Y and (if
  there are enough time steps for the processes) T parallel methods.
- Precision inferred from the inputs on 0.1 / 0.05 deg nested grids starting at 41.7 deg: output data use the
  least precise floating dtype of the source variable and source/destination grid coordinates. Float32 source or
  destination coordinates round every geometry to float32 so no mass leaks through sliver overlaps (a single
  src cell sends exactly zero mass or flux outside its nested dst cells) and mass is conserved (rtol 1e-6,
  1e-4 with float32 coordinates whose rounded edges change the cell areas). Nearest neighbour also keeps
  float32 data.
- keep_nan on a non-nested dst (4x4 cells of 0.75 deg over [0,3]x[0,3]) with a NaN in src cell [1,2)x[1,2):
  keep_nan=True gives NaN exactly on the dst cells overlapping it, also with a reloaded weight matrix
  (regression: the padding slots of a reloaded weight matrix used to give NaN everywhere); keep_nan=False
  drops its mass and conserves the rest of the src mass inside the dst domain.

Run: python test_interp_options_synthetic.py
     mpirun -np 4 python test_interp_options_synthetic.py
"""

import numpy as np

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    SIZE,
    TOTAL_RTOL,
    cell_areas,
    check_flux_and_mass,
    default_values,
    feature,
    interpolate,
    make_grid,
    parent_field,
    tmp_path,
    write_geojson,
)


def assert_same(result, reference, label):
    if RANK == 0:
        assert np.allclose(result, reference, rtol=TOTAL_RTOL, atol=0.0, equal_nan=True), (
            f"[{label}] differs from the in-memory result:\n{result - reference}"
        )
        print(f"[{label}] equals the in-memory result", flush=True)


def weight_matrix_options():
    points = write_geojson(
        "points.geojson",
        [feature(1.0, "Point", [0.25, 0.25]), feature(3.0, "Point", [0.75, 0.75]), feature(2.0, "Point", [3.25, 2.25])],
    )
    values = default_values()
    a_src = cell_areas(1.0, 4)
    for proxy in (None, points):
        name = "points" if proxy else "no proxy"
        # Conservation of the in-memory results (both modes)
        check_flux_and_mass(values, f"wm {name}", proxy=proxy, proxy_type="points")
        for flux in (False, True):
            data = values if flux else values * a_src
            label = f"wm {name} flux={flux}"
            memory = interpolate(data, flux=flux, proxy=proxy, proxy_type="points")
            path = tmp_path(f"wm_{name.replace(' ', '_')}_{flux}.nc")
            assert_same(
                interpolate(data, flux=flux, proxy=proxy, proxy_type="points", weight_matrix_path=path),
                memory,
                f"{label} created",
            )
            assert_same(
                interpolate(data, flux=flux, proxy=proxy, proxy_type="points", weight_matrix_path=path),
                memory,
                f"{label} reloaded",
            )
            src_grid, dst_grid = make_grid(1.0, 4), make_grid(0.5, 8)
            wm = src_grid.interpolate_horizontal(
                dst_grid=dst_grid,
                kind="Conservative",
                flux=flux,
                info=False,
                only_create_wm=True,
                proxy=proxy,
                proxy_type="points",
            )
            assert_same(interpolate(data, flux=flux, wm=wm), memory, f"{label} wm object")


def time_levels():
    n_times = max(3, SIZE)
    rng = np.random.default_rng(7)
    values = rng.uniform(0.5, 5.0, size=(n_times, 2, 4, 4))
    points = write_geojson(
        "points_tl.geojson", [feature(1.0, "Point", [1.25, 1.75]), feature(2.0, "Point", [1.75, 1.25])]
    )
    for method in ("Y", "T"):
        check_flux_and_mass(
            values, f"{n_times} times x 2 levels {method}", parallel_method=method, proxy=points, proxy_type="points"
        )


def precision():
    values = default_values()
    expected_flux = parent_field(values, 2)
    # Mass only in the src cell [1, 2]: any mass outside its 4 nested dst cells is a sliver leak
    single = np.zeros((4, 4))
    single[1, 2] = 5.0
    children = parent_field(single, 2) > 0
    grids = {"src": (0.1, 4), "dst": (0.05, 8), "lat0": 41.7}
    for label, kwargs, out_dtype in (
        ("float64 data and coordinates", {}, np.float64),
        ("float32 data", {"data_dtype": np.float32}, np.float32),
        ("float32 src coordinates", {"src_coord_dtype": np.float32}, np.float32),
        ("float32 dst coordinates", {"dst_coord_dtype": np.float32}, np.float32),
        (
            "float32 data and src coordinates",
            {"data_dtype": np.float32, "src_coord_dtype": np.float32},
            np.float32,
        ),
    ):
        # 0.1 deg cells starting at 41.7 deg: float32(41.7) != 41.7, so mixing float32 and float64 edges
        # without rounding them to the lowest dtype creates sliver overlaps with the neighbouring cells.
        # Rounded float32 edges move ~1e-6 deg, changing the cell areas used by the flux conversion by ~1e-5
        # relative to the float64 reference areas of the checks.
        rtol = 1e-4 if any(key.endswith("coord_dtype") for key in kwargs) else 1e-6
        result = check_flux_and_mass(values, label, total_rtol=rtol, **grids, **kwargs)
        leak = check_flux_and_mass(single, f"{label} single cell", total_rtol=rtol, **grids, **kwargs)
        if RANK != 0:
            continue
        for out in result[:2] + leak[:2]:
            assert out.dtype == out_dtype, f"[{label}] output dtype {out.dtype} is not {np.dtype(out_dtype)}"
        for name, out in (("mass", leak[0]), ("flux", leak[1])):
            assert np.all(out[0, 0][~children] == 0.0), f"[{label}] {name} leaked out of the nested cells:\n{out[0, 0]}"
        print(f"[{label}] {np.dtype(out_dtype)} output, no sliver leaks", flush=True)

    nn = interpolate(
        values, src=(0.1, 4), dst=(0.05, 8), lat0=41.7, kind="NearestNeighbour", n_neighbours=1,
        data_dtype=np.float32, src_coord_dtype=np.float32,
    )
    if RANK == 0:
        assert nn.dtype == np.float32, f"[NN float32] output dtype {nn.dtype} is not float32"
        assert np.array_equal(nn[0, 0], expected_flux.astype(np.float32)), f"[NN float32] wrong values:\n{nn[0, 0]}"
        print("[NN float32] float32 output equals the parent values", flush=True)


def keep_nan():
    values = default_values()
    values[1, 1] = np.nan
    a_src = cell_areas(1.0, 4)
    a_dst = cell_areas(0.75, 4)
    dst = (0.75, 4)
    nan_cells = np.zeros((4, 4), dtype=bool)
    # dst cells overlapping src [1,2)x[1,2): [0.75,1.5) and [1.5,2.25) on both axes
    nan_cells[1:3, 1:3] = True
    for flux in (False, True):
        data = values if flux else values * a_src
        label = f"keep_nan flux={flux}"
        kept = interpolate(data, dst=dst, flux=flux, keep_nan=True)
        path = tmp_path(f"wm_keep_nan_{flux}.nc")
        interpolate(data, dst=dst, flux=flux, keep_nan=True, weight_matrix_path=path)
        reloaded = interpolate(data, dst=dst, flux=flux, keep_nan=True, weight_matrix_path=path)
        dropped = interpolate(data, dst=dst, flux=flux, keep_nan=False)
        if RANK != 0:
            continue
        for name, result in (("in memory", kept), ("reloaded", reloaded)):
            assert np.array_equal(np.isnan(result[0, 0]), nan_cells), (
                f"[{label} {name}] NaN cells:\n{np.isnan(result[0, 0])}"
            )
        assert_same(reloaded, kept, f"{label} reloaded")
        assert not np.isnan(dropped).any(), f"[{label}] keep_nan=False gave NaN"
        # The dst domain [0,3]x[0,3] covers the src cells [0:3, 0:3] entirely
        expected = np.nansum((values * a_src)[0:3, 0:3])
        total = np.sum(dropped[0, 0] * (a_dst if flux else 1.0))
        assert np.isclose(total, expected, rtol=TOTAL_RTOL), f"[{label}] keep_nan=False total {total} != {expected}"
        # The cells not overlapping the NaN src cell are the same with both options
        assert np.allclose(kept[0, 0][~nan_cells], dropped[0, 0][~nan_cells], rtol=TOTAL_RTOL, atol=0.0)
        print(f"[{label}] NaN on the overlapping dst cells only; keep_nan=False conserved {total:.6e}", flush=True)


def main():
    weight_matrix_options()
    time_levels()
    precision()
    keep_nan()
    if RANK == 0:
        print("ALL CHECKS PASSED")


def test_interpolation_options():
    main()


if __name__ == "__main__":
    main()

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Synthetic CF LAEA tests, independent of archive data and external CRS names."""

import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import Dataset
from pyproj import CRS, Transformer

import nes


EUROPE = dict(latitude_of_projection_origin=52., longitude_of_projection_origin=10.,
              false_easting=4321000., false_northing=3210000.,
              semi_major_axis=6378137., inverse_flattening=298.257222101)
SPHERE = dict(latitude_of_projection_origin=-30., longitude_of_projection_origin=135.,
              earth_radius=6370000., false_easting=1500., false_northing=-2000.)


def grid(cf=None, comm=MPI.COMM_SELF, **kwargs):
    """Create a small LAEA test grid containing an extensive emission field.

    Parameters
    ----------
    cf : dict or None, optional
        CF projection attributes. None uses the European ellipsoid fixture.
    comm : mpi4py.MPI.Comm, optional
        Grid communicator. Default is MPI.COMM_SELF.
    **kwargs : dict
        Overrides for geometry or base NES creation options.

    Returns
    -------
    nessy : nes.LAEANes
        Grid with rank-local data, emission units and descriptive global attributes.
    """
    params = dict(nx=4, ny=4, inc_x=1000., inc_y=1000., x_0=4320000., y_0=3210000.)
    params.update(kwargs)
    n = nes.create_nes(comm=comm, projection="laea", **params, **(EUROPE if cf is None else cf))
    n.variables["emission"] = {"data": np.ones((1, 1) + n.lat["data"].shape),
                               "units": "kg", "long_name": "emission per cell"}
    n.global_attrs.update(title="Synthetic LAEA emissions", history="Created for NES LAEA tests")
    return n


def synthetic(path, cf=None, x=None, y=None, bounds=True, units="m"):
    """Write independent CF input with arbitrary coordinate, bounds and mapping names.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination NetCDF path.
    cf : dict or None, optional
        CF mapping attributes; None uses the European ellipsoid fixture.
    x, y : array_like or None, optional
        Projected coordinate values in units. None uses the metre-valued default arrays.
    bounds : bool, optional
        Write explicit asymmetric coordinate bounds if True. Default is True.
    units : {'m', 'km'}, optional
        Projected coordinate units; also controls the test interval widths. Default is m.
    """
    cf = EUROPE if cf is None else cf
    x = np.array([4320500., 4321500., 4322500.]) if x is None else np.asarray(x)
    y = np.array([3210500., 3211500.]) if y is None else np.asarray(y)
    with Dataset(path, "w") as ds:
        ds.Conventions = "CF-1.7"
        ds.title = "Synthetic projected input"
        ds.history = "Created independently of NES"
        ds.createDimension("easting", len(x))
        ds.createDimension("northing", len(y))
        ds.createDimension("nv", 2)
        for name, values, axis in (("easting", x, "x"), ("northing", y, "y")):
            v = ds.createVariable(name, "f8", (name,))
            v.standard_name = "projection_{}_coordinate".format(axis)
            v.units = units
            v[:] = values
            if bounds:
                v.bounds = name + "_edges"
                b = ds.createVariable(v.bounds, "f8", (name, "nv"))
                b[:] = np.column_stack((values - .4 * (1000 if units == "m" else 1),
                                        values + .6 * (1000 if units == "m" else 1)))
        mapping = ds.createVariable("arbitrary_mapping", "i4", fill_value=-999)
        mapping.setncatts(dict(cf, grid_mapping_name="lambert_azimuthal_equal_area"))
        v = ds.createVariable("emission", "f8", ("northing", "easting"))
        v.grid_mapping = "arbitrary_mapping"
        v.long_name = "emission per cell"
        v.units = "kg"
        v[:] = np.arange(len(x) * len(y)).reshape(len(y), len(x)) + 1


@pytest.mark.parametrize("cf,reference", [
    (EUROPE, CRS.from_epsg(3035)),
    (SPHERE, CRS.from_proj4("+proj=laea +lat_0=-30 +lon_0=135 +R=6370000 +x_0=1500 +y_0=-2000")),
])
def test_creation_transform_and_roundtrip(tmp_path, cf, reference):
    """Verify independent centre/corner transforms and lossless native NetCDF round trips.
    """
    n = grid(cf, x_0=4320000 if cf is EUROPE else 0, y_0=3210000 if cf is EUROPE else 0)
    assert isinstance(n, nes.LAEANes)
    target = reference.geodetic_crs
    transform = Transformer.from_crs(reference, target, always_xy=True)
    xx, yy = np.meshgrid(n.x["data"], n.y["data"])
    lon, lat = transform.transform(xx, yy)
    np.testing.assert_allclose(n.lat["data"], lat, atol=1e-10)
    np.testing.assert_allclose(n.lon["data"], lon, atol=1e-10)
    corners_x = np.array([n.x["bounds_data"][0, 0], n.x["bounds_data"][0, 1],
                          n.x["bounds_data"][0, 1], n.x["bounds_data"][0, 0]])
    corners_y = np.array([n.y["bounds_data"][0, 0], n.y["bounds_data"][0, 0],
                          n.y["bounds_data"][0, 1], n.y["bounds_data"][0, 1]])
    lon, lat = transform.transform(corners_x, corners_y)
    np.testing.assert_allclose(n.lat_bnds["data"][0, 0], lat, atol=1e-10)
    np.testing.assert_allclose(n.lon_bnds["data"][0, 0], lon, atol=1e-10)
    path = str(tmp_path / "laea.nc")
    n.to_netcdf(path)
    with Dataset(path) as ds:
        assert ds.Conventions == "CF-1.7"
        mapping = ds.variables[ds.variables["emission"].grid_mapping]
        assert mapping.grid_mapping_name == "lambert_azimuthal_equal_area"
        for key, value in cf.items():
            assert mapping.getncattr(key) == value
        for axis in ("x", "y"):
            v = ds.variables[axis]
            assert v.standard_name == "projection_{}_coordinate".format(axis)
            assert v.units == "m"
            assert ds.variables[v.bounds].shape == (4, 2)
        assert not hasattr(ds.variables["lat"], "axis")
    r = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    r.load()
    assert isinstance(r, nes.LAEANes)
    for name in ("x", "y", "lat", "lon", "lat_bnds", "lon_bnds"):
        np.testing.assert_allclose(getattr(n, name)["data"], getattr(r, name)["data"], atol=1e-11)
    assert r.projection.crs.equals(n.projection.crs)
    assert list(r.variables) == ["emission"]
    assert r.create_shapefile().geometry.is_valid.all()
    assert r.shapefile.crs.ellipsoid.semi_major_metre == n.earth_radius[1]
    assert len(r.get_centroids_from_coordinates()) == 16
    copied = r.copy(copy_vars=True)
    assert isinstance(copied, nes.LAEANes)
    np.testing.assert_array_equal(copied.x["bounds_data"], r.x["bounds_data"])
    restored = pickle.loads(pickle.dumps(r))
    assert restored.projection.crs.equals(r.projection.crs)
    r.close()


@pytest.mark.parametrize("units,descending,bounds", [("m", False, True), ("km", True, True), ("m", True, False)])
def test_cf_input_arbitrary_names_units_and_bounds(tmp_path, units, descending, bounds):
    """Check CF discovery, unit conversion, axis direction and explicit/inferred bounds.
    """
    factor = 1000 if units == "km" else 1
    cf = EUROPE.copy()
    cf["false_easting"] /= factor
    cf["false_northing"] /= factor
    x, y = np.array([4320500., 4321500., 4322500.]), np.array([3210500., 3211500.])
    if descending:
        x, y = x[::-1], y[::-1]
    path = str(tmp_path / "input.nc")
    synthetic(path, cf, x=x / factor, y=y / factor, bounds=bounds, units=units)
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    n.load()
    np.testing.assert_array_equal(n.x["data"], x / factor)
    np.testing.assert_array_equal(n.y["data"], y / factor)
    transformer = Transformer.from_crs(3035, 4258, always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    np.testing.assert_allclose(n.lat["data"], lat, atol=1e-10)
    np.testing.assert_allclose(n.lon["data"], lon, atol=1e-10)
    xb = n.x["bounds_data"][0] * factor
    yb = n.y["bounds_data"][0] * factor
    lon, lat = transformer.transform([min(xb), max(xb), max(xb), min(xb)],
                                     [min(yb), min(yb), max(yb), max(yb)])
    np.testing.assert_allclose(n.lat_bnds["data"][0, 0], lat, atol=1e-10)
    np.testing.assert_allclose(n.lon_bnds["data"][0, 0], lon, atol=1e-10)
    output = str(tmp_path / "output.nc")
    n.to_netcdf(output)
    r = nes.open_netcdf(output, comm=MPI.COMM_SELF)
    r.load()
    np.testing.assert_array_equal(n.variables["emission"]["data"], r.variables["emission"]["data"])
    np.testing.assert_array_equal(n.x["bounds_data"], r.x["bounds_data"])
    np.testing.assert_allclose(n.lat_bnds["data"], r.lat_bnds["data"], atol=1e-11)
    n.close()
    r.close()


@pytest.mark.parametrize("change,message", [
    ({"latitude_of_projection_origin": None}, "latitude_of_projection_origin"),
    ({"longitude_of_projection_origin": None}, "longitude_of_projection_origin"),
    ({"semi_major_axis": None}, "requires earth_radius"),
    ({"inverse_flattening": -1}, "inverse_flattening"),
    ({"latitude_of_projection_origin": 91}, "between -90 and 90"),
    ({"semi_major_axis": 0}, "semi_major_axis"),
    ({"false_easting": np.nan}, "finite scalar"),
    ({"semi_minor_axis": 6300000}, "disagree"),
])
def test_invalid_cf_metadata(tmp_path, change, message):
    """Reject missing or inconsistent numerical CF projection attributes.
    """
    cf = EUROPE.copy()
    for key, value in change.items():
        if value is None:
            cf.pop(key)
        else:
            cf[key] = value
    path = str(tmp_path / "invalid.nc")
    synthetic(path, cf)
    with pytest.raises(ValueError, match=message):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


@pytest.mark.parametrize("earth", [dict(earth_radius=6370000),
                                   dict(semi_major_axis=6370000, inverse_flattening=0),
                                   dict(semi_major_axis=6378137, semi_minor_axis=6356752.314140356)])
def test_alternative_earth_definitions(earth):
    """Accept radius, zero flattening and explicit minor-axis Earth definitions.
    """
    cf = dict(latitude_of_projection_origin=40, longitude_of_projection_origin=-100, **earth)
    n = grid(cf)
    assert np.isfinite(n.lat["data"]).all()
    assert n.earth_radius[1] == earth.get("earth_radius", earth.get("semi_major_axis"))


@pytest.mark.parametrize("values", [[1], [1, 2, 4]])
def test_ambiguous_bounds_require_explicit_bounds(tmp_path, values):
    """Reject singleton/nonuniform projected axes without explicit bounds.
    """
    path = str(tmp_path / "ambiguous.nc")
    synthetic(path, x=values, bounds=False)
    with pytest.raises(ValueError, match="explicit bounds"):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


def test_nonuniform_explicit_bounds(tmp_path):
    """Accept nonuniform and singleton axes when their bounds are explicit.
    """
    path = str(tmp_path / "nonuniform.nc")
    synthetic(path, x=[4320500, 4322000, 4325000], y=[3210500], bounds=True)
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    assert n.lat_bnds["data"].shape == (1, 3, 4)
    n.close()


def test_conservative_extensive_and_intensive():
    # A regular destination enclosing the source accounts for all source emissions.
    """Conserve cell totals and area-integrated densities on an enclosing regular grid.
    """
    src = grid(SPHERE, x_0=0, y_0=0)
    src.variables["emission"]["data"][:] = np.arange(16).reshape(1, 1, 4, 4) + 1
    dst = nes.create_nes(comm=MPI.COMM_SELF, projection="regular", lat_orig=-30.1, lon_orig=134.9,
                         inc_lat=.02, inc_lon=.02, n_lat=12, n_lon=12)
    result = src.interpolate_horizontal(dst, kind="Conservative")
    np.testing.assert_allclose(np.nansum(result.variables["emission"]["data"]), 136, rtol=1e-9)
    # Intensive data: convert to cell totals first, then back using destination areas,
    # following NES's flux=True convention. Compare integrated quantities.
    src_area = src.create_shapefile().to_crs("+proj=cea +R=6371000").area.values
    dst_area = dst.create_shapefile().to_crs("+proj=cea +R=6371000").area.values
    src.variables["emission"]["data"][:] = 2.
    src.variables["emission"]["units"] = "kg m-2"
    flux = src.interpolate_horizontal(dst, kind="Conservative", flux=True)
    np.testing.assert_allclose(
        np.nansum(flux.variables["emission"]["data"].ravel() * dst_area), np.sum(2 * src_area), rtol=1e-8)


@pytest.mark.parametrize("cf,units", [(EUROPE, "m"), (SPHERE, "m"), (EUROPE, "km")])
def test_independent_cf_checker(tmp_path, cf, units):
    """Validate independent inputs and native outputs with the optional CF checker.
    """
    pytest.importorskip("compliance_checker")
    from compliance_checker.runner import ComplianceChecker, CheckSuite
    n = grid(cf)
    path = str(tmp_path / "compliant.nc")
    n.to_netcdf(path)
    # The directly authored input is checked separately from the NES round trip.
    input_path = str(tmp_path / "input.nc")
    input_cf = cf.copy()
    factor = 1000 if units == "km" else 1
    for key in ("false_easting", "false_northing"):
        if key in input_cf:
            input_cf[key] /= factor
    synthetic(input_path, input_cf, units=units,
              x=np.array([4320500., 4321500., 4322500.]) / factor,
              y=np.array([3210500., 3211500.]) / factor)
    read = nes.open_netcdf(input_path, comm=MPI.COMM_SELF, time_exception=True)
    read.load()
    normalized_path = str(tmp_path / "normalized.nc")
    read.to_netcdf(normalized_path)
    read.close()
    CheckSuite.load_all_available_checkers()
    for candidate in (input_path, path, normalized_path):
        passed, errors = ComplianceChecker.run_checker(candidate, ["cf:1.7"], 0, "normal",
                                                       output_filename=os.devnull)
        assert passed and not errors


@pytest.mark.skipif(not os.path.isdir("/esarchive/recon/uba_germany/gretav1.6.0.1/original_files"),
                    reason="GRETA archive is not mounted")
def test_greta_metadata_only():
    """Inspect accessible GRETA CF metadata without loading emission arrays.
    """
    from nes.projections.laea_nes import laea_mapping
    path = ("/esarchive/recon/uba_germany/gretav1.6.0.1/original_files/"
            "2022_Sub2024_Condensables_Inland_GNFR_Greta1.6.0.1.nc")
    # Inspection precedes opening NES; no emission arrays are loaded.
    with Dataset(path) as ds:
        assert "CF-" in getattr(ds, "Conventions", ""), "GRETA does not declare CF conventions"
        assert laea_mapping(ds) is not None, "GRETA needs a referenced CF LAEA mapping"
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    assert n.projection.crs.equals(CRS.from_epsg(3035), ignore_axis_order=True)
    n.close()


def test_grid_only_roundtrip(tmp_path):
    """Reopen a native grid-only file without requiring a scientific data variable.
    """
    n = grid()
    n.variables = {}
    path = str(tmp_path / "grid.nc")
    n.to_netcdf(path)
    r = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    assert isinstance(r, nes.LAEANes)
    np.testing.assert_array_equal(n.x["data"], r.x["data"])
    r.close()


@pytest.mark.parametrize("mutation,message", [
    ("angular_units", "length units"), ("nonmonotonic", "strictly monotonic"),
    ("missing_bounds", "bounds variable"), ("missing_mapping", "missing variable"),
    ("bad_bounds", "contain their centres"),
])
def test_invalid_coordinate_metadata(tmp_path, mutation, message):
    """Reject unsupported units and invalid coordinate, bounds or mapping references.
    """
    path = str(tmp_path / "invalid.nc")
    synthetic(path)
    with Dataset(path, "a") as ds:
        if mutation == "angular_units":
            ds.variables["easting"].units = "degrees_east"
        elif mutation == "nonmonotonic":
            ds.variables["easting"][:] = [1, 2, 1]
        elif mutation == "missing_bounds":
            ds.variables["easting"].bounds = "absent"
        elif mutation == "missing_mapping":
            ds.variables["emission"].grid_mapping = "absent"
        elif mutation == "bad_bounds":
            ds.variables["easting_edges"][:] = 0
    with pytest.raises(ValueError, match=message):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


def test_expanded_mapping_and_prime_meridian(tmp_path):
    """Interpret expanded CF mapping syntax and a non-Greenwich prime meridian.
    """
    path = str(tmp_path / "expanded.nc")
    cf = dict(EUROPE, longitude_of_prime_meridian=2.33722917)
    synthetic(path, cf)
    with Dataset(path, "a") as ds:
        ds.variables["emission"].grid_mapping = "arbitrary_mapping: easting northing"
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    reference = CRS.from_proj4(
        "+proj=laea +lat_0=52 +lon_0=10 +a=6378137 +rf=298.257222101 "
        "+x_0=4321000 +y_0=3210000 +pm=2.33722917")
    lon, lat = Transformer.from_crs(reference, 4258, always_xy=True).transform(
        *np.meshgrid(n.x["data"], n.y["data"]))
    np.testing.assert_allclose(n.lon["data"], lon, atol=1e-9)
    np.testing.assert_allclose(n.lat["data"], lat, atol=1e-9)
    n.close()


@pytest.mark.parametrize("inc", [1000, -1000])
def test_expand_contract_preserves_projected_geometry(inc):
    """Preserve projected geometry through expansion/contraction in both axis directions.
    """
    n = grid(inc_x=inc, inc_y=inc)
    original = n.copy(copy_vars=True)
    n.expand()
    assert n.lat["data"].shape == (6, 6)
    xx, yy = np.meshgrid(n.x["data"], n.y["data"])
    lon, lat = Transformer.from_crs(3035, 4258, always_xy=True).transform(xx, yy)
    np.testing.assert_allclose(n.lat["data"], lat, atol=1e-10)
    np.testing.assert_allclose(n.lon["data"], lon, atol=1e-10)
    n.contract()
    for name in ("x", "y", "lat", "lon", "lat_bnds", "lon_bnds"):
        np.testing.assert_allclose(getattr(n, name)["data"], getattr(original, name)["data"], atol=1e-10)
    latitude, longitude = n.create_providentia_exp_grid_edge_coordinates()
    assert latitude["data"].shape == longitude["data"].shape
    np.testing.assert_allclose(latitude["data"][[0, -1]], latitude["data"][0])


def test_selection_preserves_projected_bounds(tmp_path):
    """Keep projected axes and geographical corners aligned after spatial selection.
    """
    n = grid()
    path = str(tmp_path / "selection.nc")
    n.to_netcdf(path)
    read = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    read.sel(lat_min=float(read.lat["data"][1, 0]), lat_max=float(read.lat["data"][2, -1]))
    assert read.y["data"].shape == (2,)
    assert read.y["bounds_data"].shape == (2, 2)
    np.testing.assert_array_equal(read.y["data"], n.y["data"][1:3])
    np.testing.assert_array_equal(read.lat_bnds["data"], n.lat_bnds["data"][1:3])
    read.load()
    read.to_netcdf(str(tmp_path / "selected.nc"))
    read.close()


def test_named_crs_and_mapping_fill_value(tmp_path):
    """Retain a named datum ensemble while excluding mapping storage fill attributes.
    """
    reference = CRS.from_epsg(3035)
    path = str(tmp_path / "named.nc")
    synthetic(path, reference.to_cf())
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    assert n.projection.crs.equals(reference, ignore_axis_order=True)
    n.load()
    output = str(tmp_path / "named_output.nc")
    n.to_netcdf(output)
    with Dataset(output) as ds:
        cf = {key: ds.variables["laea"].getncattr(key) for key in ds.variables["laea"].ncattrs()}
        assert CRS.from_cf(cf).equals(reference, ignore_axis_order=True)
    n.close()


def test_outside_projection_domain():
    """Reject creation geometry outside the finite LAEA projection domain.
    """
    with pytest.raises(ValueError, match="outside the projection domain"):
        grid(x_0=1e10)


def test_wkt_kilometres_and_cf_consistency(tmp_path):
    """Convert kilometre WKT offsets and reject conflicting CF projection origins.
    """
    reference = CRS.from_proj4(
        "+proj=laea +lat_0=52 +lon_0=10 +a=6378137 +rf=298.257222101 "
        "+x_0=4321000 +y_0=3210000 +units=km")
    path = str(tmp_path / "wkt_km.nc")
    synthetic(path, reference.to_cf(), units="km", x=[4320.5, 4321.5, 4322.5], y=[3210.5, 3211.5])
    n = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    lon, lat = Transformer.from_crs(reference, reference.geodetic_crs, always_xy=True).transform(
        *np.meshgrid(n.x["data"], n.y["data"]))
    np.testing.assert_allclose(n.lon["data"], lon, atol=1e-10)
    np.testing.assert_allclose(n.lat["data"], lat, atol=1e-10)
    n.close()
    with Dataset(path, "a") as ds:
        ds.variables["arbitrary_mapping"].latitude_of_projection_origin = 40.
    with pytest.raises(ValueError, match="crs_wkt disagrees"):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


def test_concatenate_paths_uses_cf_reader(tmp_path):
    """Exclude CF coordinate/mapping metadata when concatenating LAEA variables.
    """
    paths = []
    for name in ("first", "second"):
        n = grid()
        n.variables[name] = n.variables.pop("emission")
        path = str(tmp_path / (name + ".nc"))
        n.to_netcdf(path)
        paths.append(path)
    result = nes.concatenate_netcdfs(paths, comm=MPI.COMM_SELF)
    assert set(result.variables) == {"first", "second"}
    assert isinstance(result, nes.LAEANes)
    result.close()

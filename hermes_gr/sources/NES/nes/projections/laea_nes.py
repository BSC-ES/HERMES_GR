# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""CF Lambert azimuthal equal area structured grids."""

from copy import deepcopy

import numpy as np
from pyproj import CRS, Proj, Transformer
from pyproj.exceptions import CRSError

from ..default_nes import Nes
from .mercator_nes import MercatorNes
from ._coordinates import build_projection, grid_perimeter, length_factor, mapping_references


EARTH_PARAMETERS = (
    "earth_radius", "semi_major_axis", "semi_minor_axis", "inverse_flattening",
    "longitude_of_prime_meridian",
)
CF_PARAMETERS = (
    "latitude_of_projection_origin", "longitude_of_projection_origin", "false_easting", "false_northing",
) + EARTH_PARAMETERS


def laea_mapping(dataset):
    """Find the CF LAEA mapping associated with a structured dataset.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Open input dataset; only variable metadata are inspected.

    Returns
    -------
    name : str or None
        Referenced LAEA mapping variable name, or None when no LAEA mapping variable exists.

    Raises
    ------
    ValueError
        A mapping reference is missing, multiple referenced mappings are present, or an LAEA
        mapping has no associated data variables.

    Notes
    -----
    Both simple and expanded CF grid_mapping syntax are supported. For a grid-only file, a
    single unreferenced LAEA mapping is accepted when the remaining variables are
    coordinates or bounds.
    """
    candidates = {name for name, var in dataset.variables.items()
                  if getattr(var, "grid_mapping_name", None) == "lambert_azimuthal_equal_area"}
    if not candidates:
        return None
    mappings = set()
    referenced = set()
    for variable in dataset.variables.values():
        text = getattr(variable, "grid_mapping", "")
        names = mapping_references(text)
        for name in names:
            if name not in dataset.variables:
                raise ValueError("CF grid_mapping references missing variable {!r}".format(name))
            referenced.add(name)
            if getattr(dataset.variables[name], "grid_mapping_name", None) == "lambert_azimuthal_equal_area":
                mappings.add(name)
    if mappings and (len(mappings) != 1 or len(referenced) != 1):
        raise ValueError("NES supports a single structured grid mapping per LAEA dataset")
    if not mappings:
        # NES also writes grid-only files. With no data variables there is no
        # grid_mapping reference to inspect, but the single mapping is unambiguous.
        coordinates = {name for name, v in dataset.variables.items()
                       if v.ndim <= 1 or getattr(v, "standard_name", "") in ("latitude", "longitude")}
        bounds = {getattr(v, "bounds", "") for name, v in dataset.variables.items() if name in coordinates}
        if len(candidates) == 1 and set(dataset.variables) <= coordinates | bounds:
            return next(iter(candidates))
        raise ValueError("LAEA mapping must be referenced by a data variable through grid_mapping")
    return next(iter(mappings))


def _metres(units):
    """Return the conversion factor from a supported CF length unit to metres.

    Parameters
    ----------
    units : str
        Case-sensitive m, km, cm or mm symbol, or a supported singular/plural metre-name
        spelling.

    Returns
    -------
    factor : float
        Multiplier converting coordinate values to metres.

    Raises
    ------
    ValueError
        Units are missing or unsupported.
    """
    return length_factor(units)


class LAEANes(Nes):
    """Represent a CF Lambert azimuthal equal area structured grid.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm or None, optional
        Communicator for collective initialization. None selects MPI.COMM_WORLD.
    path : str or None, optional
        Input NetCDF path when no open dataset is supplied and create_nes is False.
    info : bool, optional
        Print reading/writing diagnostics. Default is False.
    dataset : netCDF4.Dataset or None, optional
        Open input dataset. Takes precedence over path when reading.
    parallel_method : {'X', 'Y', 'T'}, optional
        Domain decomposition axis. Default is Y.
    avoid_first_hours : int, optional
        Hours excluded from the beginning or end of the input time range. Defaults are zero.
    avoid_last_hours : int, optional
        Hours excluded from the beginning or end of the input time range. Defaults are zero.
    first_level : int, optional
        First retained vertical index. Default is zero.
    last_level : int or None, optional
        Last-level selection limit, following the base NES slicing convention. None retains
        remaining levels.
    create_nes : bool, optional
        Create a grid from parameters if True; otherwise read a CF NetCDF grid. Default is
        False.
    balanced : bool, optional
        Balance cells between ranks. Default is False; balanced output cannot use chunking.
    times : sequence of datetime or None, optional
        Creation time coordinates. Defaults to 1996-12-31 when omitted.
    **kwargs : dict
        Base NES options and LAEA parameters described under Other Parameters.

    Other Parameters
    ----------------
    crs : str, int or pyproj.CRS, optional
        Explicit source Lambert Azimuthal Equal Area CRS for incomplete input metadata.
        Existing contradictory projection metadata raise ValueError.
    coordinate_names, coordinate_units : dict, optional
        Source axis variable names and missing units, keyed by x and/or y. Explicit
        descriptions permit lazy spatial-axis reordering without modifying the source.
    time_exception : bool, optional
        Allow input without a time dimension and use the base NES placeholder time. Default
        is False.
    latitude_of_projection_origin, longitude_of_projection_origin : float
        Required CF projection origins in degrees when creating a grid. lat_0 and lon_0 are
        accepted aliases.
    earth_radius : float, optional
        Explicit spherical radius in metres. Supply this or a complete ellipsoid definition.
    semi_major_axis, semi_minor_axis : float, optional
        Ellipsoid axes in metres. A semi-major axis requires a semi-minor axis or inverse
        flattening.
    inverse_flattening : float, optional
        Alternative ellipsoid definition with semi_major_axis. Zero describes a sphere.
    false_easting, false_northing : float, optional
        Projection offsets. Creation uses metres; input files use the respective projected
        axis units. Defaults are zero.
    longitude_of_prime_meridian : float, optional
        Prime meridian in degrees. The CRS default is Greenwich.
    x_0, y_0 : float
        First projected cell edges in metres when creating a grid; separate from false
        offsets.
    inc_x, inc_y : float
        Finite, nonzero signed cell increments in metres when creating a grid.
    nx, ny : int
        Positive projected cell counts when creating a grid.

    See Also
    --------

    nes.create_nes
        Create LAEA grids through the public factory.
    nes.open_netcdf
        Detect and read CF LAEA files automatically.


    Notes
    -----
    All ranks must initialize the same grid. Projected metadata and geometry are validated
    before base-class collectives. Input emission variables remain lazy.

    Automatic reading requires one rectilinear CF LAEA mapping. Explicit input options
    can complete missing metadata and adapt the source axis order. Projected axes may ascend or
    descend. Explicit bounds are required for singleton or nonuniform axes; uniform
    nonsingleton axes permit inferred edges. No Earth shape or EPSG code is assumed.

    Attributes
    ----------
    x, y : dict
        Rank-local projected axis dictionaries containing data, units and bounds_data.
    lat, lon : dict
        Rank-local geographical centre arrays with shape (ny_local, nx_local), in degrees.
    lat_bnds, lon_bnds : dict
        Rank-local geographical corner arrays with shape (ny_local, nx_local, 4), in
        degrees.
    projection_data : dict
        CF projection attributes retained for NetCDF serialization.
    projection : pyproj.Proj
        Projected CRS reconstructed from the explicit CF Earth shape and parameters.
    earth_radius : list of float
        Semi-minor and semi-major Earth axes, respectively, in metres.
    """

    # Reuse the existing projected-grid accessors and spatial integration.
    get_full_x = MercatorNes.get_full_x
    get_full_y = MercatorNes.get_full_y
    set_full_x = MercatorNes.set_full_x
    set_full_y = MercatorNes.set_full_y
    create_providentia_exp_centre_coordinates = MercatorNes.create_providentia_exp_centre_coordinates

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """Initialize and distribute a validated LAEA grid.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None, optional
            Communicator for collective initialization. None selects MPI.COMM_WORLD.
        path : str or None, optional
            Input NetCDF path when no open dataset is supplied and create_nes is False.
        info : bool, optional
            Print reading/writing diagnostics. Default is False.
        dataset : netCDF4.Dataset or None, optional
            Open input dataset. Takes precedence over path when reading.
        parallel_method : {'X', 'Y', 'T'}, optional
            Domain decomposition axis. Default is Y.
        avoid_first_hours : int, optional
            Hours excluded from the beginning or end of the input time range. Defaults are zero.
        avoid_last_hours : int, optional
            Hours excluded from the beginning or end of the input time range. Defaults are zero.
        first_level : int, optional
            First retained vertical index. Default is zero.
        last_level : int or None, optional
            Last-level selection limit, following the base NES slicing convention. None retains
            remaining levels.
        create_nes : bool, optional
            Create a grid from parameters if True; otherwise read a CF NetCDF grid. Default is
            False.
        balanced : bool, optional
            Balance cells between ranks. Default is False; balanced output cannot use chunking.
        times : sequence of datetime or None, optional
            Creation time coordinates. Defaults to 1996-12-31 when omitted.
        **kwargs : dict
            Base NES options and LAEA parameters described under Other Parameters.

        Other Parameters
        ----------------
        crs : str, int or pyproj.CRS, optional
            Explicit source Lambert Azimuthal Equal Area CRS for incomplete input metadata.
            Existing contradictory projection metadata raise ValueError.
        coordinate_names, coordinate_units : dict, optional
            Source axis variable names and missing units, keyed by x and/or y. Explicit
            descriptions permit lazy spatial-axis reordering without modifying the source.
        time_exception : bool, optional
            Allow input without a time dimension and use the base NES placeholder time. Default
            is False.
        latitude_of_projection_origin, longitude_of_projection_origin : float
            Required CF projection origins in degrees when creating a grid. lat_0 and lon_0 are
            accepted aliases.
        earth_radius : float, optional
            Explicit spherical radius in metres. Supply this or a complete ellipsoid definition.
        semi_major_axis, semi_minor_axis : float, optional
            Ellipsoid axes in metres. A semi-major axis requires a semi-minor axis or inverse
            flattening.
        inverse_flattening : float, optional
            Alternative ellipsoid definition with semi_major_axis. Zero describes a sphere.
        false_easting, false_northing : float, optional
            Projection offsets. Creation uses metres; input files use the respective projected
            axis units. Defaults are zero.
        longitude_of_prime_meridian : float, optional
            Prime meridian in degrees. The CRS default is Greenwich.
        x_0, y_0 : float
            First projected cell edges in metres when creating a grid; separate from false
            offsets.
        inc_x, inc_y : float
            Finite, nonzero signed cell increments in metres when creating a grid.
        nx, ny : int
            Positive projected cell counts when creating a grid.

        Raises
        ------
        ValueError
            Projection metadata, units, axis geometry or projected-domain coordinates are
            invalid.

        Notes
        -----
        All ranks must initialize the same grid. Projected metadata and geometry are validated
        before base-class collectives. Input emission variables remain lazy.

        Automatic reading requires one rectilinear CF LAEA mapping. Explicit input options
        can complete missing metadata and adapt the source axis order. Projected axes may ascend or
        descend. Explicit bounds are required for singleton or nonuniform axes; uniform
        nonsingleton axes permit inferred edges. No Earth shape or EPSG code is assumed.
        """
        self._input_options = {key: kwargs.pop(key, None)
                               for key in ("crs", "coordinate_names", "coordinate_units")}
        self._input_orders = {}
        self._full_x = self._full_y = None
        self._laea_axes = {}
        self._laea_units = {"x": "m", "y": "m"}
        self._laea_mapping_name = "laea"
        if create_nes:
            # Validate on every rank before the base constructor enters collectives.
            for axis in ("x", "y"):
                n, inc, origin = kwargs["n" + axis], kwargs["inc_" + axis], kwargs[axis + "_0"]
                if int(n) != n or n < 1 or not np.isfinite([inc, origin]).all() or inc == 0:
                    raise ValueError("LAEA grid sizes must be positive integers and increments finite and nonzero")
            self.projection_data = self._get_projection_data(True, **kwargs)
            self.projection = self._get_pyproj_projection()
            self._validate_projected_domain(
                [kwargs["x_0"], kwargs["x_0"] + kwargs["nx"] * kwargs["inc_x"]],
                [kwargs["y_0"], kwargs["y_0"] + kwargs["ny"] * kwargs["inc_y"]])
        else:
            if dataset is None:
                from netCDF4 import Dataset
                dataset = Dataset(path)
            try:
                self._prepare_input(dataset)
            except Exception:
                dataset.close()
                raise
        super().__init__(comm=comm, path=path, info=info, dataset=dataset, parallel_method=parallel_method,
                         avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                         first_level=first_level, last_level=last_level, create_nes=create_nes,
                         balanced=balanced, times=times, **kwargs)
        if create_nes:
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")
        else:
            self.set_full_x(deepcopy(self._laea_axes["x"]))
            self.set_full_y(deepcopy(self._laea_axes["y"]))
        self.x = self._get_coordinate_values(self.get_full_x(), "X")
        self.y = self._get_coordinate_values(self.get_full_y(), "Y")
        self._var_dim = self._lat_dim = self._lon_dim = ("y", "x")
        self.create_spatial_bounds()
        # Input coordinate arrays are retained only in the usual master-owned dictionaries.
        self._laea_axes = {}

    @classmethod
    def new(cls, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
        """Construct an LAEA object using the standard NES reconstruction hook.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None, optional
            Communicator for collective initialization. None selects MPI.COMM_WORLD.
        path : str or None, optional
            Input NetCDF path when no open dataset is supplied and create_nes is False.
        info : bool, optional
            Print reading/writing diagnostics. Default is False.
        dataset : netCDF4.Dataset or None, optional
            Open input dataset. Takes precedence over path when reading.
        parallel_method : {'X', 'Y', 'T'}, optional
            Domain decomposition axis. Default is Y.
        avoid_first_hours : int, optional
            Hours excluded from the beginning or end of the input time range. Defaults are zero.
        avoid_last_hours : int, optional
            Hours excluded from the beginning or end of the input time range. Defaults are zero.
        first_level : int, optional
            First retained vertical index. Default is zero.
        last_level : int or None, optional
            Last-level selection limit, following the base NES slicing convention. None retains
            remaining levels.
        create_nes : bool, optional
            Create a grid from parameters if True; otherwise read a CF NetCDF grid. Default is
            False.
        balanced : bool, optional
            Balance cells between ranks. Default is False; balanced output cannot use chunking.
        times : sequence of datetime or None, optional
            Creation time coordinates. Defaults to 1996-12-31 when omitted.
        **kwargs : dict
            Base NES options and LAEA parameters described under Other Parameters.

        Returns
        -------
        nessy : LAEANes
            New LAEA grid on the requested communicator.

        See Also
        --------

        LAEANes
            Constructor and supported arguments.
        """
        return cls(comm=comm, path=path, info=info, dataset=dataset, parallel_method=parallel_method,
                   avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                   first_level=first_level, last_level=last_level, create_nes=create_nes,
                   balanced=balanced, times=times, **kwargs)

    def _prepare_input(self, dataset):
        """Read and validate projected input metadata before MPI collectives.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open CF input dataset. Only coordinate/bounds arrays and metadata are read.

        Raises
        ------
        ValueError
            The mapping, axes, units, projected bounds, Earth shape or coordinate domain is
            unsupported or invalid.

        Notes
        -----
        Populate temporary axis dictionaries, native units, projection attributes and coordinate
        transformers on each rank. Exclude storage-only mapping attributes from CRS metadata.
        Explicit crs, coordinate_names and coordinate_units options complete absent
        source metadata in memory and permit a different source spatial dimension order.
        """
        explicit = any(value is not None for value in self._input_options.values())
        names = self._input_options["coordinate_names"]
        units = self._input_options["coordinate_units"]
        for label, value in (("coordinate_names", names), ("coordinate_units", units)):
            if value is not None and (not isinstance(value, dict) or set(value) - {"x", "y"}
                                      or any(not isinstance(v, str) or not v for v in value.values())):
                raise ValueError(f"{label} must map x and/or y to nonempty strings")
        names, units = names or {}, units or {}
        name = laea_mapping(dataset)
        if explicit and name is None and any(hasattr(v, "grid_mapping_name") for v in dataset.variables.values()):
            raise ValueError("Explicit LAEA input conflicts with an existing source grid mapping")
        if name is None and self._input_options["crs"] is None:
            raise ValueError("LAEA requires a CF grid_mapping or an explicit crs")
        self._laea_mapping_name = name
        source_cf = {}
        if name is not None:
            mapping = dataset.variables[name]
            source_cf = {key: mapping.getncattr(key) for key in mapping.ncattrs()
                         if key not in ("_FillValue", "missing_value", "scale_factor", "add_offset")}
        mapped = [v for v in dataset.variables.values()
                  if name is not None and name in mapping_references(getattr(v, "grid_mapping", ""))]
        for axis in ("x", "y"):
            if axis in names:
                if names[axis] not in dataset.variables:
                    raise ValueError(f"Source {axis} coordinate {names[axis]!r} does not exist")
                candidates = [dataset.variables[names[axis]]]
            else:
                candidates = [v for v in dataset.variables.values()
                              if getattr(v, "standard_name", "") == f"projection_{axis}_coordinate"
                              and v.ndim == 1 and all(v.dimensions[0] in data.dimensions for data in mapped)]
            if len(candidates) != 1 or candidates[0].ndim != 1:
                raise ValueError(f"LAEA requires one 1D projection_{axis}_coordinate; supply coordinate_names")
            variable = candidates[0]
            source_units = getattr(variable, "units", None)
            selected_units = units.get(axis, source_units)
            if axis in units and source_units and not np.isclose(_metres(source_units), _metres(selected_units)):
                raise ValueError(f"Explicit {axis} units conflict with source coordinate units")
            info = {"data": np.asarray(np.ma.filled(variable[:].astype(float), np.nan)),
                    "units": selected_units}
            _metres(info["units"])
            self._laea_units[axis] = info["units"]
            self._validate_axis(info["data"], axis)
            bounds = getattr(variable, "bounds", None)
            if bounds is not None:
                if bounds not in dataset.variables:
                    raise ValueError("LAEA {} bounds variable {!r} is missing".format(axis, bounds))
                bv = dataset.variables[bounds]
                if bv.dimensions[:-1] != variable.dimensions or bv.shape != (len(info["data"]), 2):
                    raise ValueError("LAEA {} bounds must have dimensions (coordinate, 2)".format(axis))
                info["bounds_data"] = np.asarray(np.ma.filled(bv[:].astype(float), np.nan))
            else:
                info["bounds_data"] = self._infer_bounds(info["data"], axis)
            self._validate_bounds(info, axis)
            info["source_name"] = variable.name
            self._laea_axes[axis] = info
        xd = dataset.variables[self._laea_axes["x"]["source_name"]].dimensions[0]
        yd = dataset.variables[self._laea_axes["y"]["source_name"]].dimensions[0]
        if xd == yd:
            raise ValueError("LAEA requires separate x and y coordinate dimensions")
        if name is None:
            mapped = [v for v in dataset.variables.values() if xd in v.dimensions and yd in v.dimensions]
            if not mapped:
                raise ValueError("No source data variable uses both projected coordinate dimensions")
        for variable in mapped:
            dims = variable.dimensions
            if dims.count(xd) != 1 or dims.count(yd) != 1:
                raise ValueError("LAEA data require separate x and y coordinate dimensions")
            leading = tuple(d for d in dims if d not in (xd, yd))
            order = leading + (yd, xd)
            if not explicit and dims[-2:] != (yd, xd):
                raise ValueError("LAEA data must end in separate (y, x) projected dimensions")
            if explicit:
                if len(dims) not in (2, 3, 4) or len(set(dims)) != len(dims):
                    raise ValueError(
                        "Explicit projected input supports numeric raster variables with 2 to 4 dimensions")
                if not np.issubdtype(variable.dtype, np.number):
                    raise ValueError("Explicit projected input requires numeric raster data")
                time_dims = {v.dimensions[0] for key, v in dataset.variables.items()
                             if key in ("time", "TIME", "valid_time") and v.ndim == 1}
                level_dims = {v.dimensions[0] for key, v in dataset.variables.items()
                              if key in ("lev", "level", "lm", "plev") and v.ndim == 1}
                if leading and (leading[0] not in time_dims
                                or (len(leading) == 2 and leading[1] not in level_dims)):
                    raise ValueError("Non-spatial source dimensions must be time or time/level in that order")
                self._input_orders[variable.name] = tuple(dims.index(d) for d in order)
        self._laea_cf = self._complete_input_crs(source_cf)
        self._validate_cf(self._laea_cf)
        self.projection_data = deepcopy(self._laea_cf)
        self.projection = self._get_pyproj_projection()
        axes = []
        for axis in ("x", "y"):
            bounds = self._laea_axes[axis]["bounds_data"] * _metres(self._laea_units[axis])
            axes.append([bounds.min(), bounds.max()])
        self._validate_projected_domain(*axes)

    def _complete_input_crs(self, source_cf):
        """Complete absent CF projection attributes from an explicit source CRS.

        Parameters
        ----------
        source_cf : dict
            Existing source mapping attributes. Numeric offsets use native axis units.

        Returns
        -------
        dict
            Independent CF attributes retaining existing information and explicit CRS WKT.

        Raises
        ------
        ValueError
            Source WKT conflicts with the supplied CRS or the CRS cannot be parsed.
        NotImplementedError
            The explicit CRS does not describe Lambert Azimuthal Equal Area.

        Notes
        -----
        Existing numeric attributes remain authoritative for consistency checks against
        the explicit WKT in build_projection. EPSG false offsets are converted from metres
        to the declared native units before merging, without changing coordinate values.
        """
        supplied = self._input_options["crs"]
        if supplied is None:
            return deepcopy(source_cf)
        try:
            reference = CRS.from_user_input(supplied)
        except CRSError as error:
            raise ValueError(f"Invalid explicit crs: {supplied!r}") from error
        cf = reference.to_cf()
        if cf.get("grid_mapping_name") != "lambert_azimuthal_equal_area":
            raise NotImplementedError("Explicit crs input currently supports Lambert Azimuthal Equal Area only")
        if "crs_wkt" in source_cf and not reference.equals(
                CRS.from_wkt(source_cf["crs_wkt"]), ignore_axis_order=True):
            raise ValueError("Explicit crs conflicts with source crs_wkt")
        # PROJ exposes conversion offsets in the CRS's own linear units.
        for axis, key, direction in (("x", "false_easting", "east"), ("y", "false_northing", "north")):
            native = next((a for a in reference.axis_info if a.direction == direction), None)
            if native is None:
                raise ValueError("Explicit projected CRS must have easting and northing axes")
            cf[key] = float(cf.get(key, 0.)) * native.unit_conversion_factor / _metres(self._laea_units[axis])
        if "earth_radius" in source_cf:
            for key in ("semi_major_axis", "semi_minor_axis", "inverse_flattening"):
                cf.pop(key, None)
        elif any(key in source_cf for key in ("semi_major_axis", "semi_minor_axis", "inverse_flattening")):
            # Do not manufacture a second, potentially inconsistent ellipsoid representation.
            cf.pop("semi_minor_axis", None)
            if "semi_minor_axis" in source_cf and "inverse_flattening" not in source_cf:
                cf.pop("inverse_flattening", None)
        cf.update(source_cf)
        cf["crs_wkt"] = reference.to_wkt()
        return cf

    def _read_variable(self, var_name):
        """Read an explicitly described raster in canonical time/level/Y/X order.

        Parameters
        ----------
        var_name : str
            Scientific source variable name.

        Returns
        -------
        numpy.ndarray
            Unmasked rank-local values with singleton time/level axes added as needed.

        Notes
        -----
        Slice source dimensions before transposing, so MPI ranks read only their own
        spatial partition. Leading axes retain the existing NES convention: time for
        three-dimensional data and time/level for four-dimensional data. Ordinary CF
        input continues through the base reader.
        """
        if var_name not in self._input_orders:
            return super()._read_variable(var_name)
        variable = self.dataset.variables[var_name]
        order = self._input_orders[var_name]
        axes = {2: ("y", "x"), 3: ("t", "y", "x"), 4: ("t", "z", "y", "x")}[variable.ndim]
        selection = [slice(None)] * variable.ndim
        for source_index, axis in zip(order, axes):
            selection[source_index] = slice(self.read_axis_limits[axis + "_min"],
                                            self.read_axis_limits[axis + "_max"])
        data = variable[tuple(selection)].transpose(order)
        if variable.ndim == 2:
            data = data[None, None, ...]
        elif variable.ndim == 3:
            data = data[:, None, ...]
        return self._unmask_array(data)

    def _validate_projected_domain(self, x, y):
        """Check that projected extrema transform to finite geographical positions.

        Parameters
        ----------
        x, y : array_like
            Projected axis extrema in metres. Their Cartesian product defines the tested
            corners.

        Raises
        ------
        ValueError
            At least one transformed position falls outside the finite LAEA domain.
        """
        lon, lat = self._to_geographical.transform(*np.meshgrid(x, y))
        if not np.isfinite([lon, lat]).all():
            raise ValueError("LAEA coordinates or cell corners fall outside the projection domain")

    @staticmethod
    def _validate_axis(values, axis):
        """Validate a finite, nonempty and strictly monotonic projected axis.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional projected centres in their native coordinate units.
        axis : {'x', 'y'}
            Axis label used in validation errors.

        Raises
        ------
        ValueError
            Centres are empty, nonfinite or not strictly ascending/descending.
        """
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("LAEA {} coordinate must be nonempty and finite".format(axis))
        delta = np.diff(values)
        if len(delta) and not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError("LAEA {} coordinate must be strictly monotonic".format(axis))

    @staticmethod
    def _infer_bounds(values, axis):
        """Infer projected cell edges from uniformly spaced axis centres.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional projected centres in their native coordinate units.
        axis : {'x', 'y'}
            Axis label used in validation errors.

        Returns
        -------
        bounds : numpy.ndarray
            Array of shape (n, 2) in the native coordinate units, preserving axis direction.

        Raises
        ------
        ValueError
            The axis is singleton or nonuniform, so explicit projected bounds are required.
        """
        delta = np.diff(values)
        if not len(delta) or not np.allclose(delta, delta[0], rtol=1e-10, atol=1e-10):
            raise ValueError("LAEA {} needs explicit bounds for singleton or nonuniform axes".format(axis))
        return np.column_stack((values - delta[0] / 2, values + delta[0] / 2))

    @staticmethod
    def _validate_bounds(info, axis):
        """Check that finite, nonzero projected intervals contain their centres.

        Parameters
        ----------
        info : dict
            Projected axis dictionary with one-dimensional ``data``, ``units`` and
            ``bounds_data`` of shape (n, 2).
        axis : {'x', 'y'}
            Axis label used in validation errors.

        Raises
        ------
        ValueError
            An interval is nonfinite, has zero width or excludes its centre.
        """
        bounds = info["bounds_data"]
        values = info["data"]
        if (not np.isfinite(bounds).all() or np.any(bounds[:, 0] == bounds[:, 1])
                or np.any(values < bounds.min(axis=1)) or np.any(values > bounds.max(axis=1))):
            raise ValueError("LAEA {} bounds must be finite, nonzero and contain their centres".format(axis))

    @staticmethod
    def _validate_cf(cf):
        """Validate explicit CF LAEA origins and spherical or ellipsoidal Earth shape.

        Parameters
        ----------
        cf : dict
            CF grid-mapping attributes. Projection angles are in degrees and Earth axes/radii
            are in metres.

        Raises
        ------
        ValueError
            Required attributes are missing, numerical parameters are invalid, Earth definitions
            conflict or pyproj rejects the CRS.

        Notes
        -----
        A semi-major axis requires a semi-minor axis or inverse flattening. Zero inverse
        flattening denotes a sphere. An explicit earth_radius is an alternative to ellipsoid
        attributes.
        """
        for key in ("latitude_of_projection_origin", "longitude_of_projection_origin"):
            if key not in cf:
                raise ValueError("LAEA missing mandatory CF attribute {}".format(key))
        for key in CF_PARAMETERS:
            if key in cf and (isinstance(cf[key], str) or np.ndim(cf[key]) != 0
                              or not np.isfinite(float(cf[key]))):
                raise ValueError("LAEA CF attribute {} must be a finite scalar".format(key))
        if abs(float(cf["latitude_of_projection_origin"])) > 90:
            raise ValueError("LAEA latitude_of_projection_origin must be between -90 and 90")
        if "earth_radius" in cf:
            if float(cf["earth_radius"]) <= 0 or any(
                    k in cf for k in ("semi_major_axis", "semi_minor_axis", "inverse_flattening")):
                raise ValueError("LAEA earth_radius must be positive and cannot be combined with ellipsoid axes")
        elif "semi_major_axis" not in cf or not any(k in cf for k in ("semi_minor_axis", "inverse_flattening")):
            raise ValueError("LAEA requires earth_radius or semi_major_axis with semi_minor_axis or inverse_flattening")
        for key in ("semi_major_axis", "semi_minor_axis"):
            if key in cf and float(cf[key]) <= 0:
                raise ValueError("LAEA {} must be positive".format(key))
        if "inverse_flattening" in cf and float(cf["inverse_flattening"]) < 0:
            raise ValueError("LAEA inverse_flattening must be nonnegative")
        if "semi_minor_axis" in cf:
            a, b = float(cf["semi_major_axis"]), float(cf["semi_minor_axis"])
            if b > a:
                raise ValueError("LAEA semi_minor_axis cannot exceed semi_major_axis")
            if "inverse_flattening" in cf:
                rf = float(cf["inverse_flattening"])
                if not np.isclose(b, a if rf == 0 else a * (1 - 1 / rf), rtol=1e-10):
                    raise ValueError("LAEA ellipsoid axes and inverse_flattening disagree")
        try:
            CRS.from_cf(cf)
        except Exception as error:
            raise ValueError("Invalid LAEA CF projection: {}".format(error)) from error

    def _get_projection_data(self, create_nes, **kwargs):
        """Return validated LAEA attributes for creation or reconstruction.

        Parameters
        ----------
        create_nes : bool
            Build CF attributes from creation keywords if True; otherwise copy input mapping
            metadata.
        **kwargs : dict
            CF projection attributes and lat_0/lon_0 origin aliases used during creation.

        Returns
        -------
        projection_data : dict
            Independent copy of the CF mapping attributes.

        Raises
        ------
        ValueError
            Origin aliases conflict or required CF parameters/Earth definitions are invalid.
        """
        if create_nes:
            cf = {key: kwargs[key] for key in CF_PARAMETERS if key in kwargs}
            for alias, key in (("lat_0", "latitude_of_projection_origin"), ("lon_0", "longitude_of_projection_origin")):
                if alias in kwargs:
                    if key in cf and cf[key] != kwargs[alias]:
                        raise ValueError("Conflicting LAEA {} and {}".format(alias, key))
                    cf[key] = kwargs[alias]
            cf["grid_mapping_name"] = "lambert_azimuthal_equal_area"
            self._validate_cf(cf)
            self._laea_cf = cf
        return deepcopy(self._laea_cf)

    def _get_pyproj_projection(self):
        # Retain CF datum/ellipsoid names as well as numeric parameters. Rebuild
        # from CF attributes so WKT cannot override projected-unit conversion.
        """Build the projected CRS and the geographical coordinate transformer.

        Returns
        -------
        projection : pyproj.Proj
            LAEA projection with false offsets expressed in metres.

        Raises
        ------
        ValueError
            WKT disagrees with the numerical CF projection definition.

        Notes
        -----
        Update earth_radius, the Greenwich-based geographical CRS and an always_xy transformer
        using the explicit Earth shape. Validate optional WKT before retaining richer CRS
        information; coordinate calculations use metre-valued projected inputs.
        """
        return build_projection(self)

    def _create_centre_coordinates(self, **kwargs):
        """Create projected cell centres and transform them to geographical coordinates.

        Parameters
        ----------
        **kwargs : dict
            Creation geometry: x_0/y_0 edge origins and inc_x/inc_y increments in metres, plus
            positive nx/ny counts.

        Returns
        -------
        latitude, longitude : dict or None
            Centre arrays of shape (ny, nx), in degrees, on rank zero; None on other ranks.

        Raises
        ------
        ValueError
            Cell counts, increments or edge origins are invalid.

        Notes
        -----
        Store full projected centres, native metre units and explicit two-vertex bounds on rank
        zero.
        """
        if not self.master:
            return None, None
        for axis in ("x", "y"):
            n, inc, origin = kwargs["n" + axis], kwargs["inc_" + axis], kwargs[axis + "_0"]
            if int(n) != n or n < 1 or not np.isfinite([inc, origin]).all() or inc == 0:
                raise ValueError("LAEA grid sizes must be positive integers and increments finite and nonzero")
            edges = origin + np.arange(int(n) + 1) * inc
            info = {"data": (edges[:-1] + edges[1:]) / 2, "units": "m",
                    "bounds_data": np.column_stack((edges[:-1], edges[1:]))}
            setattr(self, "_full_" + axis, info)
        x, y = np.meshgrid(self._full_x["data"], self._full_y["data"])
        lon, lat = self._to_geographical.transform(x, y)
        return {"data": lat}, {"data": lon}

    def _get_coordinate_dimension(self, possible_names):
        """Read a base coordinate or derive geographical centres from projected axes.

        Parameters
        ----------
        possible_names : str or list of str
            Base coordinate names. Latitude/longitude name lists trigger projected-to-
            geographical transformation.

        Returns
        -------
        coordinate : dict or None
            Full coordinate dictionary on rank zero, or None on other ranks.

        Raises
        ------
        ValueError
            A geographical centre lies outside the finite projection domain.
        """
        if isinstance(possible_names, list) and possible_names[0] in ("lat", "lon"):
            if not self.master:
                return None
            self._get_pyproj_projection()
            x, y = np.meshgrid(self._laea_axes["x"]["data"] * _metres(self._laea_axes["x"]["units"]),
                               self._laea_axes["y"]["data"] * _metres(self._laea_axes["y"]["units"]))
            lon, lat = self._to_geographical.transform(x, y)
            if not np.isfinite([lon, lat]).all():
                raise ValueError("LAEA coordinates fall outside the projection domain")
            return {"data": lat if possible_names[0] == "lat" else lon}
        return super()._get_coordinate_dimension(possible_names)

    def _get_coordinate_values(self, coordinate_info, coordinate_axis, bounds=False):
        """Slice coordinate centres and associated projected bounds for this rank.

        Parameters
        ----------
        coordinate_info : dict or None
            Full coordinate dictionary, optionally containing projected bounds_data.
        coordinate_axis : {'X', 'Y', 'Z'}
            Axis used by the base NES coordinate decomposition.
        bounds : bool, optional
            Whether data contains a final vertex dimension. Default is False.

        Returns
        -------
        coordinate : dict or None
            Independent local coordinate dictionary, or None if no coordinate was supplied.
        """
        values = super()._get_coordinate_values(coordinate_info, coordinate_axis, bounds=bounds)
        if values is not None and "bounds_data" in values:
            # The base slicer distributes data only; projected bounds live in a separate metadata entry.
            axis = coordinate_axis.lower()
            section = slice(self.read_axis_limits[axis + "_min"], self.read_axis_limits[axis + "_max"])
            values["bounds_data"] = coordinate_info["bounds_data"][section].copy()
        return values

    def _get_lazy_variables(self):
        # The base initializer needs the projection before it reads geographical coordinates.
        """
        Read data-variable metadata while excluding coordinate and mapping variables.

        Returns
        -------
        variables : dict
            Variable metadata with unloaded data entries.

        Notes
        -----
        Collective operation. Mapping/coordinate names and bounds references come from CF
        metadata, including arbitrarily named latitude and longitude variables.
        Explicit source descriptions reorder variable dimension metadata consistently
        with the lazy reader; source NetCDF metadata are never modified.
        """
        self.projection_data = deepcopy(self._laea_cf)
        variables = super()._get_lazy_variables()
        excluded = {self._laea_mapping_name}
        for axis in self._laea_axes.values():
            name = axis["source_name"]
            excluded.add(name)
            excluded.add(getattr(self.dataset.variables[name], "bounds", ""))
        for name, variable in self.dataset.variables.items():
            if getattr(variable, "standard_name", "") in ("latitude", "longitude"):
                excluded.add(name)
                excluded.add(getattr(variable, "bounds", ""))
        for name, order in self._input_orders.items():
            if name in variables:
                dims = variables[name]["dimensions"]
                variables[name]["dimensions"] = tuple(dims[i] for i in order)
        return {name: value for name, value in variables.items() if name not in excluded}

    def create_shapefile(self, overwrite=False):
        """Create geographical grid-cell polygons using the grid Earth shape.

        Parameters
        ----------
        overwrite : bool, optional
            Rebuild cached polygons if True. Default is False.

        Returns
        -------
        shapefile : geopandas.GeoDataFrame
            Rank-local cell polygons indexed by global FID and stored in self.shapefile.

        Notes
        -----
        Call on all MPI ranks. Reuse the projected-grid polygon constructor and label
        coordinates with the actual geographical CRS rather than a default Earth shape.
        """
        geometry = MercatorNes.create_shapefile(self, overwrite=overwrite)
        geometry.set_crs(self._geographic_crs, allow_override=True, inplace=True)
        return geometry

    def get_centroids_from_coordinates(self):
        """Create cell-centre points from the local geographical coordinate arrays.

        Returns
        -------
        centroids : geopandas.GeoDataFrame
            Rank-local points indexed by global FID, with the grid geographical CRS.

        Notes
        -----
        Collective operation because global cell identifiers are distributed through the
        communicator.
        """
        geometry = MercatorNes.get_centroids_from_coordinates(self)
        return geometry.set_crs(self._geographic_crs, allow_override=True)

    def create_spatial_bounds(self):
        """Transform projected cell corners and store full and local geographical bounds.

        Raises
        ------
        ValueError
            A projected cell corner has no finite geographical transform.

        Notes
        -----
        Collective operation. Corners are ordered lower-left, lower-right, upper-right and
        upper-left in projected space, independently of ascending or descending axis order. Full
        arrays are retained on rank zero; local arrays follow the NES read limits.
        """
        lat, lon = self._project_bounds(self.get_full_x(), self.get_full_y())
        self.set_full_latitudes_boundaries({"data": lat})
        self.set_full_longitudes_boundaries({"data": lon})
        self.lat_bnds = self._get_coordinate_values({"data": lat}, "Y", bounds=True)
        self.lon_bnds = self._get_coordinate_values({"data": lon}, "X", bounds=True)

    def _project_bounds(self, x, y):
        """Transform the Cartesian product of explicit projected axis intervals.

        Parameters
        ----------
        x, y : dict
            Projected axis dictionary with one-dimensional ``data``, ``units`` and
            ``bounds_data`` of shape (n, 2).

        Returns
        -------
        latitude, longitude : numpy.ndarray
            Geographical corner arrays of shape (ny, nx, 4), in degrees, in counterclockwise
            projected order.

        Raises
        ------
        ValueError
            A projected corner falls outside the finite LAEA domain.
        """
        xb = x["bounds_data"] * _metres(x["units"])
        yb = y["bounds_data"] * _metres(y["units"])
        # Consistent counterclockwise corners, including descending axes.
        xl, xr = xb.min(axis=1), xb.max(axis=1)
        yl, yr = yb.min(axis=1), yb.max(axis=1)
        xx = np.stack(np.broadcast_arrays(xl[None, :], xr[None, :], xr[None, :], xl[None, :]))
        yy = np.stack(np.broadcast_arrays(yl[:, None], yl[:, None], yr[:, None], yr[:, None]))
        xx, yy = np.broadcast_arrays(xx, yy)
        lon, lat = self._to_geographical.transform(np.moveaxis(xx, 0, -1), np.moveaxis(yy, 0, -1))
        if not np.isfinite([lon, lat]).all():
            raise ValueError("LAEA cell corners fall outside the projection domain")
        return lat, lon

    def _refresh_projected_coordinates(self, full=False):
        """Recalculate geographical centres and corners after projected-axis changes.

        Parameters
        ----------
        full : bool, optional
            Update full coordinate dictionaries when True, otherwise rank-local dictionaries.
            Default is False.

        Raises
        ------
        ValueError
            A recalculated corner lies outside the finite projection domain.

        Notes
        -----
        This helper performs no communication. Its caller must restrict full-array updates to
        rank zero.
        """
        prefix = "_full_" if full else ""
        x, y = getattr(self, prefix + "x"), getattr(self, prefix + "y")
        xx, yy = np.meshgrid(x["data"] * _metres(x["units"]), y["data"] * _metres(y["units"]))
        lon, lat = self._to_geographical.transform(xx, yy)
        lat_bnds, lon_bnds = self._project_bounds(x, y)
        for name, values in (("lat", lat), ("lon", lon), ("lat_bnds", lat_bnds), ("lon_bnds", lon_bnds)):
            setattr(self, prefix + name, {"data": values})

    def _resize_projected_coordinates(self, n_cells, expand, full=False, is_first=True, is_last=True):
        """Resize projected axes and bounds, then recompute geographical geometry.

        Parameters
        ----------
        n_cells : int
            Number of cells added or removed at each selected edge.
        expand : bool
            Add cells if True, or remove them if False.
        full : bool, optional
            Resize full arrays on rank zero rather than local arrays. Default is False.
        is_first : bool, optional
            Whether this rank owns the first distributed partition. Default is True.
        is_last : bool, optional
            Whether this rank owns the last distributed partition. Default is True.

        Notes
        -----
        Only outer MPI partitions resize the corresponding distributed edge. Other spatial axes
        resize both edges. Expansion assumes uniform spacing, checked by expand before this
        helper is called.
        """
        if full and not self.master:
            return
        prefix = "_full_" if full else ""
        for axis in ("x", "y"):
            info = getattr(self, prefix + axis)
            distributed = not full and self.parallel_method == axis.upper()
            left, right = (is_first, is_last) if distributed else (True, True)
            if expand:
                # Uniform spacing was checked before entering the base expansion workflow.
                step = info["data"][1] - info["data"][0] if len(info["data"]) > 1 else (
                    info["bounds_data"][0, 1] - info["bounds_data"][0, 0])
                for key in ("data", "bounds_data"):
                    values = info[key]
                    shifts = np.arange(n_cells, 0, -1) * step
                    tail_shifts = np.arange(1, n_cells + 1) * step
                    if key == "bounds_data":
                        shifts, tail_shifts = shifts[:, None], tail_shifts[:, None]
                    parts = ([values[0] - shifts] if left else []) + [values]
                    parts += [values[-1] + tail_shifts] if right else []
                    info[key] = np.concatenate(parts)
            else:
                section = slice(n_cells if left else None, -n_cells if right and n_cells else None)
                for key in ("data", "bounds_data"):
                    info[key] = info[key][section]
        self._refresh_projected_coordinates(full=full)

    def expand(self, n_cells=1):
        """Expand a uniformly spaced LAEA grid while preserving projected cell geometry.

        Parameters
        ----------
        n_cells : int, optional
            Number of cells to add at each outer spatial edge. Default is one.

        Raises
        ------
        ValueError
            Projected centre spacing is nonuniform.

        Notes
        -----
        Collective operation. Base NES code expands scientific variables and updates rank
        limits; LAEA coordinates and corners are recomputed from the expanded projected axes.
        """
        for axis in (self.get_full_x(), self.get_full_y()):
            values = axis["data"]
            if len(values) > 2 and not np.allclose(np.diff(values), np.diff(values)[0]):
                raise ValueError("LAEA expansion requires uniformly spaced projected coordinates")
        super().expand(n_cells)

    def _expand_coordinates(self, n_cells, is_first, is_last):
        """Expand projected axes and regenerate geographical coordinates.

        Parameters
        ----------
        n_cells : int
            Number of cells added or removed at each selected edge.
        is_first : bool
            Whether this rank owns the first part of the distributed axis.
        is_last : bool
            Whether this rank owns the last part of the distributed axis.

        Notes
        -----
        Update rank-local geometry, modifying distributed edges only on the first/last
        partition.
        """
        self._resize_projected_coordinates(n_cells, True, is_first=is_first, is_last=is_last)

    def _contract_coordinates(self, n_cells, is_first, is_last):
        """Contract projected axes and regenerate geographical coordinates.

        Parameters
        ----------
        n_cells : int
            Number of cells added or removed at each selected edge.
        is_first : bool
            Whether this rank owns the first part of the distributed axis.
        is_last : bool
            Whether this rank owns the last part of the distributed axis.

        Notes
        -----
        Update rank-local geometry, modifying distributed edges only on the first/last
        partition.
        """
        self._resize_projected_coordinates(n_cells, False, is_first=is_first, is_last=is_last)

    def _expand_full_coordinates(self, n_cells):
        """Expand projected axes and regenerate geographical coordinates.

        Parameters
        ----------
        n_cells : int
            Number of cells added or removed at each selected edge.

        Notes
        -----
        Update full geometry only on rank zero; this helper performs no communication.
        """
        self._resize_projected_coordinates(n_cells, True, full=True)

    def _contract_full_coordinates(self, n_cells):
        """Contract projected axes and regenerate geographical coordinates.

        Parameters
        ----------
        n_cells : int
            Number of cells added or removed at each selected edge.

        Notes
        -----
        Update full geometry only on rank zero; this helper performs no communication.
        """
        self._resize_projected_coordinates(n_cells, False, full=True)

    def create_providentia_exp_grid_edge_coordinates(self):
        """Transform the closed perimeter of a contiguous projected grid.

        Returns
        -------
        grid_edge_lat : dict
            Closed clockwise latitude perimeter in degrees.
        grid_edge_lon : dict
            Closed clockwise longitude perimeter in degrees.

        Raises
        ------
        ValueError
            Projected bounds are not contiguous along either axis.

        Notes
        -----
        Use the local projected axis intervals, sorted spatially, so descending axes retain the
        correct outer outline.

        The closed outline follows the clockwise native-coordinate convention used by all structured grids,
        beginning at the native lower-left corner. Descending axes do not reverse the perimeter orientation.
        """
        return grid_perimeter(self, 'y', 'x')

    def _create_dimensions(self, netcdf):
        """Create native NetCDF dimensions for LAEA axes and geographical/projected vertices.

        Parameters
        ----------
        netcdf : netCDF4.Dataset
            Open output dataset. Every participating MPI rank must call this writer hook.
        """
        super()._create_dimensions(netcdf)
        for axis in ("x", "y"):
            netcdf.createDimension(axis, len(getattr(self, "get_full_" + axis)()["data"]))
        netcdf.createDimension("spatial_nv", 4)
        netcdf.createDimension("projected_nv", 2)

    def _create_dimension_variables(self, netcdf):
        """Write CF coordinate variables and explicit projected/geographical bounds.

        Parameters
        ----------
        netcdf : netCDF4.Dataset
            Open output dataset. Every participating MPI rank must call this writer hook.

        Notes
        -----
        Use double precision for spatial coordinates and a CF-compatible numeric level index.
        Projected axes carry X/Y annotations; two-dimensional auxiliary latitude/longitude
        variables do not duplicate those axis annotations.
        """
        super()._create_dimension_variables_64(netcdf)
        lev = netcdf.variables["lev"]
        lev.long_name = "model level"
        lev.axis = "Z"
        if not getattr(lev, "positive", ""):
            lev.positive = "up"
        if not getattr(lev, "units", ""):
            lev.units = "1"
        full_levels = self.get_full_levels()
        for key in ("standard_name", "long_name", "formula_terms"):
            if key in full_levels:
                lev.setncattr(key, full_levels[key])
        for name in ("lat", "lon"):
            if "axis" in netcdf.variables[name].ncattrs():
                netcdf.variables[name].delncattr("axis")
        for axis in ("x", "y"):
            info = getattr(self, "get_full_" + axis)()
            variable = netcdf.createVariable(axis, "f8", (axis,))
            variable.standard_name = "projection_{}_coordinate".format(axis)
            variable.long_name = "{} coordinate of projection".format(axis)
            variable.units = info["units"]
            variable.axis = axis.upper()
            variable.bounds = axis + "_bnds"
            bounds = netcdf.createVariable(axis + "_bnds", "f8", (axis, "projected_nv"))
            if self.size > 1:
                variable.set_collective(True)
                bounds.set_collective(True)
            variable[:] = info["data"]
            bounds[:] = info["bounds_data"]

    @staticmethod
    def _set_var_crs(var):
        """Set CF LAEA mapping and geographical auxiliary-coordinate references.

        Parameters
        ----------
        var : netCDF4.Variable
            Output scientific variable whose grid_mapping and coordinates attributes are
            updated.
        """
        var.grid_mapping = "laea"
        var.coordinates = "lat lon"

    def _create_metadata(self, netcdf):
        """Write the scalar LAEA mapping variable and retained CF projection attributes.

        Parameters
        ----------
        netcdf : netCDF4.Dataset
            Open output dataset. Every participating MPI rank must call this writer hook.
        """
        mapping = netcdf.createVariable("laea", "i4")
        mapping.setncatts(self.projection_data)

    def _create_variables(self, netcdf, chunking=False):
        """Write scientific variables and supply missing descriptive names.

        Parameters
        ----------
        netcdf : netCDF4.Dataset
            Open output dataset. Every participating MPI rank must call this writer hook.
        chunking : bool, optional
            Use base NES chunked output when True. Default is False.

        Notes
        -----
        Keep supplied standard_name/long_name attributes. Otherwise use the variable name as
        long_name.
        """
        super()._create_variables(netcdf, chunking=chunking)
        for name in self.variables:
            variable = netcdf.variables[name]
            if not getattr(variable, "standard_name", "") and not getattr(variable, "long_name", ""):
                variable.long_name = name

    def _filter_coordinates_selection(self):
        """Apply a spatial selection to projected axes and their explicit bounds.

        Notes
        -----
        Collective operation. Slice local axes using original read limits before filtering full
        arrays, then let the base class filter geographical/time/level coordinates. This
        preserves input-file read indices while selected arrays use the new domain.
        """
        idx = self._get_idx_intervals()
        for axis in ("x", "y"):
            info = getattr(self, "get_full_" + axis)()
            # Local slices must use the original file indices before full arrays are reduced to the selection.
            setattr(self, axis, self._get_coordinate_values(info, axis.upper()))
            section = slice(idx["idx_{}_min".format(axis)], idx["idx_{}_max".format(axis)])
            info["data"] = info["data"][section]
            info["bounds_data"] = info["bounds_data"][section]
            getattr(self, "set_full_" + axis)(info)
        super()._filter_coordinates_selection()

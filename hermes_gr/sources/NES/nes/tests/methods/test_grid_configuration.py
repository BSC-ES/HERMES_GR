"""Pure canonical configuration and actual NES method tests without native imports."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Condition

import numpy as np


NES = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('nes_grid_configuration_under_test', NES / 'grid_configuration.py')
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


def nes_class():
    """
    Load the actual NES configuration API without native module imports.

    Returns
    -------
    type
        Isolated class containing the actual NES configuration API methods.
    """
    tree = ast.parse((NES / 'default_nes.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Nes')
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in
                ['get_grid_configuration', 'validate_grid_configuration', '_update_grid_configuration_attribute']]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              cls], type_ignores=[])
    namespace = dict(json=json, GRID_CONFIGURATION_ATTRIBUTE=configuration.ATTRIBUTE,
                     PROJECTION_FIELDS=configuration.PROJECTION_FIELDS,
                     describe_grid_configuration=configuration.describe_grid_configuration,
                     configuration_from_parameters=configuration.configuration_from_parameters,
                     compare_grid_configurations=configuration.compare_grid_configurations)
    exec(compile(ast.fix_missing_locations(module), str(NES / 'default_nes.py'), 'exec'), namespace)
    return namespace['Nes']


class BroadcastGroup:
    def __init__(self):
        """Initialize isolated collective state for the regression fixture."""
        self.condition, self.values, self.arrivals = Condition(), {}, {}

    def communicator(self, rank):
        """
        Create a broadcast communicator for one simulated rank.

        Parameters
        ----------
        rank : int
            Zero-based simulated rank.

        Returns
        -------
        object
            Rank-local broadcast stand-in sharing the group synchronization state.
        """
        group = self

        class Comm:
            calls = 0

            def bcast(self, value, root=0):
                """
                Return the root payload after simulated collective participation.

                Parameters
                ----------
                value : object
                    Payload contributed to the simulated collective.
                root : int, optional
                    Rank supplying the broadcast payload.

                Returns
                -------
                object
                    Payload contributed by the root rank.
                """
                call = self.calls
                self.calls += 1
                with group.condition:
                    group.arrivals[call] = group.arrivals.get(call, 0) + 1
                    if rank == root:
                        group.values[call] = value
                    group.condition.notify_all()
                    if not group.condition.wait_for(lambda: group.arrivals[call] == 2, timeout=3):
                        raise RuntimeError('Missing collective participant')
                    return group.values[call]
        return Comm()


class GridConfigurationTests(unittest.TestCase):
    def setUp(self):
        """Prepare the isolated fixtures required by each regression."""
        self.Nes = nes_class()

    def parameters(self, projection='regular'):
        """
        Build geographical creation arguments for a small grid.

        Parameters
        ----------
        projection : str, optional
            Supported projection name for the configuration fixture.

        Returns
        -------
        dict
            Geographical creation arguments for the requested projection.
        """
        values = dict(lat_orig=0., lon_orig=0., inc_lat=1., inc_lon=1., n_lat=2, n_lon=2)
        if projection == 'global':
            values = dict(inc_lat=90., inc_lon=180.)
        if projection == 'global_monarch':
            values = dict(inc_lat=90., inc_lon=180.)
        if projection == 'rotated':
            values = dict(centre_lat=40., centre_lon=0., south_boundary=-.5, west_boundary=-.5,
                          inc_rlat=1., inc_rlon=1.)
        if projection in ['lcc', 'mercator']:
            values = dict(nx=2, ny=2, x_0=0., y_0=0., inc_x=1000., inc_y=1000., lon_0=0.)
            values.update(dict(lat_1=30., lat_2=60., lat_0=40.) if projection == 'lcc' else dict(lat_ts=40.))
        return values

    def grid(self, definition):
        """
        Build an isolated NES API instance from a canonical definition.

        Parameters
        ----------
        definition : dict
            Canonical geographical definition used to build controlled native axes.

        Returns
        -------
        object
            Isolated NES API instance with controlled global metadata and native axes.
        """
        grid = self.Nes.__new__(self.Nes)
        grid.master, grid.comm = True, SimpleNamespace(bcast=lambda value, root=0: value)
        grid.earth_radius, grid.global_attrs = {1: 6371000.}, {}
        grid.projection_data = dict(definition['projection'])
        if any(axis['layout'] == 'global_monarch' for axis in definition['axes'].values()):
            grid.projection_data['nes_projection'] = 'global_monarch'
        for name, axis in definition['axes'].items():
            values = axis['first'] + np.arange(axis['size']) * axis['step']
            if axis['layout'] == 'global_monarch':
                values[1:] -= axis['step'] / 4
                values[-1] -= axis['step'] / 4
            setattr(grid, '_full_' + name, {'data': values})
        if definition['projection']['grid_mapping_name'] != 'latitude_longitude':
            grid._full_lat = {'data': np.zeros(definition['shape'])}
            grid._full_lon = {'data': np.zeros(definition['shape'])}
        return grid

    def test_created_and_loaded_configurations_match_for_supported_projections(self):
        """Verify created and loaded configurations match for supported projections."""
        for kind in ['regular', 'global', 'global_monarch', 'rotated', 'lcc', 'mercator']:
            with self.subTest(kind=kind):
                expected = configuration.configuration_from_parameters(kind, self.parameters(kind))
                created = self.grid(expected)
                self.assertEqual(created.get_grid_configuration(), expected)
                created._update_grid_configuration_attribute()
                loaded = self.grid(expected)
                loaded.global_attrs = dict(created.global_attrs)
                self.assertEqual(loaded.get_grid_configuration(), expected)
                self.assertEqual(json.loads(loaded.global_attrs[configuration.ATTRIBUTE]), expected)
                self.assertTrue(loaded.validate_grid_configuration(projection=kind, **self.parameters(kind)))

    def test_scalar_cf_attributes_normalize_single_element_arrays(self):
        """Verify scalar CF attributes normalize single element arrays."""
        expected = configuration.configuration_from_parameters('regular', self.parameters())
        grid = self.grid(expected)
        grid.projection_data['semi_major_axis'] = np.array([6371000.])
        grid.projection_data['inverse_flattening'] = np.array([0.])
        self.assertEqual(grid.get_grid_configuration(), expected)

    def test_legacy_file_without_saved_attribute_uses_axes_and_cf(self):
        """Verify legacy file without saved attribute uses axes and CF."""
        expected = configuration.configuration_from_parameters('regular', self.parameters())
        grid = self.grid(expected)
        self.assertNotIn(configuration.ATTRIBUTE, grid.global_attrs)
        self.assertTrue(grid.validate_grid_configuration(expected))
        self.assertNotIn(configuration.ATTRIBUTE, grid.global_attrs)

    def test_origin_resolution_shape_and_projection_mismatches_are_explicit(self):
        """Verify origin resolution shape and projection mismatches are explicit."""
        original = self.parameters()
        grid = self.grid(configuration.configuration_from_parameters('regular', original))
        for field, value in [('lat_orig', 10.), ('inc_lon', 2.), ('n_lat', 3)]:
            with self.subTest(field=field):
                changed = dict(original, **{field: value})
                with self.assertRaisesRegex(ValueError, 'expected.*found'):
                    grid.validate_grid_configuration(projection='regular', **changed)
        with self.assertRaisesRegex(ValueError, 'projection'):
            grid.validate_grid_configuration(projection='mercator', **self.parameters('mercator'))

    def test_write_refreshes_stale_attribute_after_valid_grid_change(self):
        """Verify write refreshes stale attribute after valid grid change."""
        expected = configuration.configuration_from_parameters('regular', self.parameters())
        grid = self.grid(expected)
        grid._update_grid_configuration_attribute()
        grid._full_lat['data'] += 5.
        grid._update_grid_configuration_attribute()
        stored = json.loads(grid.global_attrs[configuration.ATTRIBUTE])
        self.assertEqual(stored['axes']['lat']['first'], 5.5)
        with self.assertRaises(ValueError):
            grid.validate_grid_configuration(expected)

    def test_time_paths_and_mpi_layout_are_not_configuration_fields(self):
        """Verify time paths and MPI layout are not configuration fields."""
        grid = self.grid(configuration.configuration_from_parameters('regular', self.parameters()))
        expected = grid.get_grid_configuration()
        grid.time, grid.parallel_method = ['different date'], 'X'
        grid.write_axis_limits, grid.path = {'x_min': 1}, '/somewhere/else.nc'
        grid.lat = {'data': np.array([99.])}
        self.assertEqual(grid.get_grid_configuration(), expected)

    def test_nonuniform_nonfinite_and_masked_axes_are_rejected(self):
        """Verify nonuniform nonfinite and masked axes are rejected."""
        expected = configuration.configuration_from_parameters('regular', dict(self.parameters(), n_lat=3))
        for values in [np.array([.5, 1.5, 3.5]), np.array([.5, np.nan, 2.5]),
                       np.ma.array([.5, 1.5, 2.5], mask=[0, 1, 0])]:
            with self.subTest(values=values):
                grid = self.grid(expected)
                grid._full_lat['data'] = values
                with self.assertRaises((ValueError, NotImplementedError)):
                    grid.get_grid_configuration()

    def test_legacy_float32_axes_allow_explicit_serialization_tolerance(self):
        """Verify legacy float32 axes allow explicit serialization tolerance."""
        parameters = dict(self.parameters(), lat_orig=40., inc_lat=.1, n_lat=100)
        expected = configuration.configuration_from_parameters('regular', parameters)
        grid = self.grid(expected)
        grid._full_lat['data'] = grid._full_lat['data'].astype(np.float32)
        self.assertTrue(grid.validate_grid_configuration(expected, angular_tolerance=1e-4))

    def test_descending_regular_axes_are_supported(self):
        """Verify descending regular axes are supported."""
        parameters = dict(self.parameters(), inc_lat=-1.)
        expected = configuration.configuration_from_parameters('regular', parameters)
        grid = self.grid(expected)
        self.assertEqual(grid.get_grid_configuration(), expected)
        self.assertTrue(grid.validate_grid_configuration(projection='regular', **parameters))

    def test_unsupported_nonuniform_grid_keeps_existing_output_behavior(self):
        """Verify unsupported nonuniform grid keeps existing output behavior."""
        expected = configuration.configuration_from_parameters('regular', dict(self.parameters(), n_lat=3))
        grid = self.grid(expected)
        grid._full_lat['data'] = np.array([.5, 1.5, 4.5])
        grid._update_grid_configuration_attribute()
        self.assertNotIn(configuration.ATTRIBUTE, grid.global_attrs)
        with self.assertRaises(NotImplementedError):
            grid.get_grid_configuration()

    def test_singleton_resolution_requires_metadata_or_bounds(self):
        """Verify singleton resolution requires metadata or bounds."""
        expected = configuration.configuration_from_parameters('regular', dict(self.parameters(), n_lat=1))
        grid = self.grid(expected)
        with self.assertRaisesRegex(ValueError, 'singleton resolution'):
            grid.get_grid_configuration()
        grid._full_lat_bnds = {'data': np.array([[0., 1.]])}
        self.assertEqual(grid.get_grid_configuration(), expected)
        del grid._full_lat_bnds
        grid.global_attrs[configuration.ATTRIBUTE] = json.dumps(expected)
        self.assertEqual(grid.get_grid_configuration(), expected)

    def test_nested_configuration_is_resolved_without_parent_path(self):
        """Verify nested configuration is resolved without parent path."""
        parent = dict(inc_rlat=.5, inc_rlon=.5, **{'1st_rlat': -1., '1st_rlon': 2.},
                      grid_north_pole_latitude=50., grid_north_pole_longitude=-180.)
        parameters = dict(n_rlat=3, n_rlon=4, parent_ratio=2, parent_grid_path='/irrelevant/path',
                          i_parent_start=2, j_parent_start=3)
        expected = configuration.configuration_from_parameters('rotated-nested', parameters, parent=parent)
        self.assertEqual(self.grid(expected).get_grid_configuration(), expected)
        self.assertNotIn('path', json.dumps(expected))
        self.assertEqual(expected['axes']['rlat']['first'], -1.)

    def test_unknown_schema_and_unsupported_grids_fail_explicitly(self):
        """Verify unknown schema and unsupported grids fail explicitly."""
        grid = self.grid(configuration.configuration_from_parameters('regular', self.parameters()))
        grid.global_attrs[configuration.ATTRIBUTE] = '{"schema_version":999}'
        with self.assertRaisesRegex(ValueError, 'schema'):
            grid.get_grid_configuration()
        grid.projection_data = {}
        with self.assertRaises(NotImplementedError):
            grid.get_grid_configuration()
        grid._update_grid_configuration_attribute()
        self.assertNotIn(configuration.ATTRIBUTE, grid.global_attrs)

    def test_absolute_tolerances_and_invalid_arguments(self):
        """Verify absolute tolerances and invalid arguments."""
        expected = configuration.configuration_from_parameters('regular', self.parameters())
        grid = self.grid(expected)
        grid._full_lat['data'] += 1e-7
        self.assertTrue(grid.validate_grid_configuration(expected))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            grid.validate_grid_configuration(expected, angular_tolerance=-1)
        with self.assertRaisesRegex(ValueError, 'either'):
            grid.validate_grid_configuration(expected, projection='regular')

    def test_rank_layout_does_not_change_results_or_mismatch_errors(self):
        """Verify rank layout does not change results or mismatch errors."""
        for incompatible in [False, True]:
            with self.subTest(incompatible=incompatible):
                actual = configuration.configuration_from_parameters('regular', self.parameters())
                group = BroadcastGroup()
                grids = [self.grid(actual), self.grid(actual)]
                for rank, grid in enumerate(grids):
                    grid.master, grid.comm = rank == 0, group.communicator(rank)
                del grids[1]._full_lat  # peers need no global or matching local arrays
                requested = dict(self.parameters(), lat_orig=10.) if incompatible else self.parameters()

                def validate(grid):
                    """
                    Run collective configuration validation and capture rank results.

                    Parameters
                    ----------
                    grid : object
                        Isolated NES API instance assigned to a simulated rank.

                    Returns
                    -------
                    bool or str
                        Validation result or shared mismatch error for this rank.
                    """
                    try:
                        return grid.validate_grid_configuration(projection='regular', **requested)
                    except ValueError as error:
                        return str(error)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(validate, grids))
                self.assertEqual(results[0], results[1])
                if incompatible:
                    self.assertIn('expected', results[0])
                    self.assertIn('found', results[0])
                else:
                    self.assertEqual(results, [True, True])

    def test_root_metadata_failure_is_reported_to_all_ranks(self):
        """Verify root metadata failure is reported to all ranks."""
        actual = configuration.configuration_from_parameters('regular', self.parameters())
        group = BroadcastGroup()
        grids = [self.grid(actual), self.grid(actual)]
        for rank, grid in enumerate(grids):
            grid.master, grid.comm = rank == 0, group.communicator(rank)
        grids[0]._full_lat['data'][0] = np.nan

        def read(grid):
            """
            Check that a rank receives the expected metadata-read error.

            Parameters
            ----------
            grid : object
                Isolated NES API instance assigned to a simulated rank.

            Returns
            -------
            None
            """
            with self.assertRaisesRegex(ValueError, 'Invalid full native grid axis lat'):
                grid.get_grid_configuration()
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(read, grids))


if __name__ == '__main__':
    unittest.main()

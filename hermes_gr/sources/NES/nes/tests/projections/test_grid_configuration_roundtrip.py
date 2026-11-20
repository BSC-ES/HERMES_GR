"""Native persistence/MPI regression tests for canonical structured-grid options."""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in
                ['mpi4py', 'netCDF4', 'nes', 'geopandas', 'pyproj'])


@unittest.skipUnless(AVAILABLE, 'Requires the native NES/MPI/NetCDF environment')
class GridConfigurationRoundtripTests(unittest.TestCase):
    def setUp(self):
        """Prepare the isolated fixtures required by each regression."""
        from mpi4py import MPI
        from nes import create_nes, open_netcdf
        self.comm = MPI.COMM_WORLD
        self.create_nes, self.open_netcdf = create_nes, open_netcdf
        directory = tempfile.mkdtemp(prefix='nes-configuration-') if self.comm.rank == 0 else None
        self.root = Path(self.comm.bcast(directory, root=0))
        self.handles = []

    def tearDown(self):
        """Close test handles and remove shared temporary fixture files."""
        for handle in self.handles:
            handle.close()
        self.comm.Barrier()
        if self.comm.rank == 0:
            shutil.rmtree(self.root)
        self.comm.Barrier()

    def parameters(self, kind):
        """
        Build geographical creation arguments for a small grid.

        Parameters
        ----------
        kind : str
            Projection/domain type used for the small native grid.

        Returns
        -------
        dict
            Geographical creation arguments for the requested projection.
        """
        if kind == 'regular':
            return dict(lat_orig=40., lon_orig=0., inc_lat=.1, inc_lon=.1, n_lat=4, n_lon=4)
        if kind == 'global':
            return dict(inc_lat=45., inc_lon=60.)
        if kind == 'global_monarch':
            return dict(inc_lat=45., inc_lon=60.)
        if kind == 'rotated':
            return dict(centre_lat=40., centre_lon=0., south_boundary=-2., west_boundary=-2.,
                        inc_rlat=1., inc_rlon=1.)
        result = dict(nx=4, ny=4, inc_x=1000., inc_y=1000., x_0=0., y_0=0., lon_0=0.)
        result.update(dict(lat_1=30., lat_2=60., lat_0=40.) if kind == 'lcc' else dict(lat_ts=40.))
        return result

    def roundtrip(self, kind, parameters):
        """
        Create, write, reopen and validate a native grid.

        Parameters
        ----------
        kind : str
            Projection/domain type used for the small native grid.
        parameters : dict
            Geographical keyword arguments forwarded to native NES creation.

        Returns
        -------
        tuple
            Created grid, reopened grid and temporary NetCDF path.
        """
        created = self.create_nes(comm=self.comm, projection=kind, balanced=True, **parameters)
        self.handles.append(created)
        expected = created.get_grid_configuration()
        filename = self.root / (kind + '.nc')
        created.to_netcdf(str(filename), serial=True)
        self.comm.Barrier()
        loaded = self.open_netcdf(str(filename), comm=self.comm, balanced=True)
        self.handles.append(loaded)
        self.assertEqual(loaded.get_grid_configuration(), expected)
        self.assertEqual(json.loads(loaded.global_attrs['nes_grid_configuration']), expected)
        self.assertTrue(loaded.validate_grid_configuration(projection=kind, **parameters))
        return created, loaded, filename

    def test_structured_projection_roundtrips(self):
        """Verify structured projection roundtrips."""
        for kind in ['regular', 'global', 'global_monarch', 'rotated', 'lcc', 'mercator']:
            with self.subTest(kind=kind):
                self.roundtrip(kind, self.parameters(kind))

    def test_nested_roundtrip_uses_resolved_geography(self):
        """Verify nested roundtrip uses resolved geography."""
        _, parent, filename = self.roundtrip('rotated', self.parameters('rotated'))
        parent.close()
        parameters = dict(parent_grid_path=str(filename), parent_ratio=2, i_parent_start=2,
                          j_parent_start=2, n_rlat=4, n_rlon=4)
        self.roundtrip('rotated-nested', parameters)

    def test_legacy_file_without_attribute_can_be_validated(self):
        """Verify legacy file without attribute can be validated."""
        from netCDF4 import Dataset
        created, loaded, filename = self.roundtrip('regular', self.parameters('regular'))
        expected = created.get_grid_configuration()
        loaded.close()
        self.comm.Barrier()
        if self.comm.rank == 0:
            with Dataset(filename, mode='r+') as dataset:
                dataset.delncattr('nes_grid_configuration')
        self.comm.Barrier()
        legacy = self.open_netcdf(str(filename), comm=self.comm, balanced=True)
        self.handles.append(legacy)
        self.assertNotIn('nes_grid_configuration', legacy.global_attrs)
        self.assertEqual(legacy.get_grid_configuration(), expected)
        self.assertTrue(legacy.validate_grid_configuration(expected))

    def test_parameter_mismatch_raises_same_error_on_every_rank(self):
        """Verify parameter mismatch raises same error on every rank."""
        _, loaded, _ = self.roundtrip('regular', self.parameters('regular'))
        requested = dict(self.parameters('regular'), lat_orig=41.)
        error = None
        try:
            loaded.validate_grid_configuration(projection='regular', **requested)
        except ValueError as exception:
            error = str(exception)
        self.assertIsNotNone(error)
        self.assertIn('expected', error)
        self.assertEqual(len(set(self.comm.allgather(error))), 1)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import xarray as xr

from clouddrift.kinematics import (
    inertial_oscillation_from_position,
    kinetic_energy,
    position_from_velocity,
    residual_position_from_displacement,
    spin,
    velocity_from_position,
)
from clouddrift.raggedarray import RaggedArray
from clouddrift.sphere import EARTH_RADIUS_METERS, coriolis_frequency

if __name__ == "__main__":
    unittest.main()


def sample_ragged_array() -> RaggedArray:
    drifter_id = [1, 3, 2]
    longitude = [[-121, -111, 51, 61, 71], [103, 113], [12, 22, 32, 42]]
    latitude = [[-90, -45, 45, 90, 0], [10, 20], [10, 20, 30, 40]]
    t = [[1, 2, 3, 4, 5], [4, 5], [2, 3, 4, 5]]
    test = [
        [True, True, True, False, False],
        [False, False],
        [True, False, False, False],
    ]
    rowsize = [len(x) for x in longitude]
    ids = [[d] * rowsize[i] for i, d in enumerate(drifter_id)]
    attrs_global = {
        "title": "test trajectories",
        "history": "version xyz",
    }
    variables_coords = ["ids", "time", "lon", "lat"]

    coords = {"lon": longitude, "lat": latitude, "ids": ids, "time": t}
    metadata = {"ID": drifter_id, "rowsize": rowsize}
    data = {"test": test}

    # append xr.Dataset to a list
    list_ds = []
    for i in range(0, len(rowsize)):
        xr_coords = {}
        for var in coords.keys():
            xr_coords[var] = (
                ["obs"],
                coords[var][i],
                {"long_name": f"variable {var}", "units": "-"},
            )

        xr_data = {}
        for var in metadata.keys():
            xr_data[var] = (
                ["traj"],
                [metadata[var][i]],
                {"long_name": f"variable {var}", "units": "-"},
            )

        for var in data.keys():
            xr_data[var] = (
                ["obs"],
                data[var][i],
                {"long_name": f"variable {var}", "units": "-"},
            )

        list_ds.append(
            xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=attrs_global)
        )

    ra = RaggedArray.from_files(
        [0, 1, 2],
        lambda i: list_ds[i],
        variables_coords,
        ["ID", "rowsize"],
        ["test"],
    )

    return ra


class kinetic_energy_tests(unittest.TestCase):
    def test_values_1d_velocity(self):
        u = np.random.random((100))
        self.assertTrue(np.all(kinetic_energy(u) == u**2 / 2))

    def test_1d_velocity_kwarg(self):
        u = np.random.random((100))
        self.assertTrue(
            np.all(kinetic_energy(u) == kinetic_energy(u, np.zeros_like(u)))
        )

    def test_values_2d_velocity(self):
        u = np.random.random((100))
        v = np.random.random((100))
        self.assertTrue(np.all(kinetic_energy(u, v) == (u**2 + v**2) / 2))


class inertial_oscillation_from_position_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 1440
        self.longitude = np.linspace(0, 45, self.INPUT_SIZE)
        self.latitude = np.linspace(40, 45, self.INPUT_SIZE)
        self.relative_vorticity = 0.1 * coriolis_frequency(self.latitude)

    def test_result_has_same_size_as_input(self):
        x, y = inertial_oscillation_from_position(
            self.longitude, self.latitude, relative_bandwidth=0.10, time_step=3600
        )
        self.assertTrue(np.all(self.longitude.shape == x.shape))
        self.assertTrue(np.all(self.longitude.shape == y.shape))

    def test_relative_vorticity(self):
        x, y = inertial_oscillation_from_position(
            self.longitude,
            self.latitude,
            relative_bandwidth=0.10,
            time_step=3600,
            relative_vorticity=self.relative_vorticity,
        )
        self.assertTrue(np.all(self.longitude.shape == x.shape))
        self.assertTrue(np.all(self.longitude.shape == y.shape))

    def test_time_step(self):
        x, y = inertial_oscillation_from_position(
            self.longitude,
            self.latitude,
            relative_bandwidth=0.10,
            time_step=60 * 20,
        )
        self.assertTrue(np.all(self.longitude.shape == x.shape))
        self.assertTrue(np.all(self.longitude.shape == y.shape))

    def test_simulation_case(self):
        lat0 = 30
        lon0 = 0
        t = np.arange(0, 60 - 1 / 24, 1 / 24)
        f = coriolis_frequency(lat0)
        uv = 0.5 * np.exp(-1j * (f * t * 86400 + 0.5 * np.pi))
        lon1, lat1 = position_from_velocity(
            np.real(uv),
            np.imag(uv),
            t * 86400,
            lon0,
            lat0,
            integration_scheme="forward",
            coord_system="spherical",
        )
        x_expected, y_expected = position_from_velocity(
            np.real(uv),
            np.imag(uv),
            t * 86400,
            0,
            0,
            integration_scheme="forward",
            coord_system="cartesian",
        )
        x_expected = x_expected - np.mean(x_expected)
        y_expected = y_expected - np.mean(y_expected)
        xhat1, yhat1 = inertial_oscillation_from_position(
            lon1, lat1, relative_bandwidth=0.10, time_step=3600
        )
        xhat1 = xhat1 - np.mean(xhat1)
        yhat1 = yhat1 - np.mean(yhat1)
        xhat2, yhat2 = inertial_oscillation_from_position(
            lon1, lat1, wavelet_duration=10, time_step=3600
        )
        xhat2 = xhat2 - np.mean(xhat2)
        yhat2 = yhat2 - np.mean(yhat2)
        m = 10
        self.assertTrue(
            np.allclose(xhat2[m * 24 : -m * 24], x_expected[m * 24 : -m * 24], atol=20)
        )
        self.assertTrue(
            np.allclose(yhat2[m * 24 : -m * 24], y_expected[m * 24 : -m * 24], atol=20)
        )
        self.assertTrue(
            np.allclose(xhat2[m * 24 : -m * 24], xhat1[m * 24 : -m * 24], atol=20)
        )
        self.assertTrue(
            np.allclose(yhat2[m * 24 : -m * 24], yhat1[m * 24 : -m * 24], atol=20)
        )


class residual_position_from_displacement_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 100
        self.lon = np.rad2deg(
            np.linspace(-np.pi, np.pi, self.INPUT_SIZE, endpoint=False)
        )
        self.lat = np.linspace(0, 45, self.INPUT_SIZE)
        self.x = np.random.rand(self.INPUT_SIZE)
        self.y = np.random.rand(self.INPUT_SIZE)

    def test_result_has_same_size_as_input(self):
        lon, lat = residual_position_from_displacement(
            self.lon, self.lat, self.x, self.y
        )
        self.assertTrue(np.all(self.lon.shape == lon.shape))
        self.assertTrue(np.all(self.lon.shape == lat.shape))

    def test_simple_case(self):
        lon, lat = residual_position_from_displacement(
            1, 0, 2 * np.pi * EARTH_RADIUS_METERS / 360, 0
        )
        self.assertTrue(np.isclose((np.mod(lon, 360), lat), (0, 0)).all())


class position_from_velocity_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 100
        self.lon = np.rad2deg(
            np.linspace(-np.pi, np.pi, self.INPUT_SIZE, endpoint=False)
        )
        self.lat = np.linspace(0, 45, self.INPUT_SIZE)
        self.time = np.linspace(0, 1e7, self.INPUT_SIZE)
        self.uf, self.vf = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="forward"
        )
        self.ub, self.vb = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="backward"
        )
        self.uc, self.vc = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="centered"
        )

    def test_result_has_same_size_as_input(self):
        lon, lat = position_from_velocity(
            self.uf,
            self.vf,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.all(self.uf.shape == lon.shape))
        self.assertTrue(np.all(self.uf.shape == lat.shape))

    def test_velocity_position_roundtrip_forward(self):
        lon, lat = position_from_velocity(
            self.uf,
            self.vf,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_velocity_position_roundtrip_backward(self):
        lon, lat = position_from_velocity(
            self.ub,
            self.vb,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="backward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_velocity_position_roundtrip_centered(self):
        lon, lat = position_from_velocity(
            self.uc,
            self.vc,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="centered",
        )
        # Centered scheme damps the 2dx waves so we need a looser tolerance.
        self.assertTrue(np.allclose(lon, self.lon, atol=1e-2))
        self.assertTrue(np.allclose(lat, self.lat, atol=1e-2))

    def test_works_with_xarray(self):
        lon, lat = position_from_velocity(
            xr.DataArray(data=self.uf),
            xr.DataArray(data=self.vf),
            xr.DataArray(data=self.time),
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_works_with_2d_array(self):
        uf = np.reshape(np.tile(self.uf, 4), (4, self.uf.size))
        vf = np.reshape(np.tile(self.vf, 4), (4, self.vf.size))
        time = np.reshape(np.tile(self.time, 4), (4, self.time.size))
        expected_lon = np.reshape(np.tile(self.lon, 4), (4, self.lon.size))
        expected_lat = np.reshape(np.tile(self.lat, 4), (4, self.lat.size))
        lon, lat = position_from_velocity(
            uf,
            vf,
            time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, expected_lon))
        self.assertTrue(np.allclose(lat, expected_lat))
        self.assertTrue(np.allclose(lon.shape, expected_lon.shape))
        self.assertTrue(np.allclose(lon.shape, expected_lat.shape))

    def test_works_with_2d_uv_1d_time(self):
        uf = np.reshape(np.tile(self.uf, 4), (4, self.uf.size))
        vf = np.reshape(np.tile(self.vf, 4), (4, self.vf.size))
        expected_lon = np.reshape(np.tile(self.lon, 4), (4, self.lon.size))
        expected_lat = np.reshape(np.tile(self.lat, 4), (4, self.lat.size))
        lon, lat = position_from_velocity(
            uf,
            vf,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, expected_lon))
        self.assertTrue(np.allclose(lat, expected_lat))
        self.assertTrue(np.allclose(lon.shape, expected_lon.shape))
        self.assertTrue(np.allclose(lon.shape, expected_lat.shape))

    def test_time_axis(self):
        uf = np.reshape(np.tile(self.uf, 6), (2, 3, self.uf.size))
        vf = np.reshape(np.tile(self.vf, 6), (2, 3, self.vf.size))
        time = np.reshape(np.tile(self.time, 6), (2, 3, self.time.size))
        expected_lon = np.reshape(np.tile(self.lon, 6), (2, 3, self.lon.size))
        expected_lat = np.reshape(np.tile(self.lat, 6), (2, 3, self.lat.size))

        for time_axis in [0, 1, 2]:
            # Pass inputs with swapped axes and differentiate along that time
            # axis.
            lon, lat = position_from_velocity(
                np.swapaxes(uf, time_axis, -1),
                np.swapaxes(vf, time_axis, -1),
                np.swapaxes(time, time_axis, -1),
                self.lon[0],
                self.lat[0],
                integration_scheme="forward",
                time_axis=time_axis,
            )

            # Swap axes back to compare with the expected result.
            self.assertTrue(np.allclose(np.swapaxes(lon, time_axis, -1), expected_lon))
            self.assertTrue(np.allclose(np.swapaxes(lat, time_axis, -1), expected_lat))
            self.assertTrue(
                np.all(np.swapaxes(lon, time_axis, -1).shape == expected_lon.shape)
            )
            self.assertTrue(
                np.all(np.swapaxes(lat, time_axis, -1).shape == expected_lat.shape)
            )


class velocity_from_position_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 100
        self.lon = np.rad2deg(np.linspace(-np.pi, np.pi, self.INPUT_SIZE))
        self.lat = np.zeros(self.lon.shape)
        self.time = np.linspace(0, 1e7, self.INPUT_SIZE)
        self.uf, self.vf = velocity_from_position(self.lon, self.lat, self.time)
        self.ub, self.vb = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="backward"
        )
        self.uc, self.vc = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="centered"
        )

    def test_result_has_same_size_as_input(self):
        self.assertTrue(np.all(self.uf.shape == self.vf.shape == self.lon.shape))
        self.assertTrue(np.all(self.ub.shape == self.vb.shape == self.lon.shape))
        self.assertTrue(np.all(self.uc.shape == self.vc.shape == self.lon.shape))

    def test_schemes_are_self_consistent(self):
        self.assertTrue(np.all(self.uf[:-1] == self.ub[1:]))
        self.assertTrue(
            np.all(np.isclose((self.uf[1:-1] + self.ub[1:-1]) / 2, self.uc[1:-1]))
        )
        self.assertTrue(self.uc[0] == self.uf[0])
        self.assertTrue(self.uc[-1] == self.ub[-1])

    def test_result_value(self):
        u_expected = 2 * np.pi * EARTH_RADIUS_METERS / 1e7
        self.assertTrue(np.all(np.isclose(self.uf, u_expected)))
        self.assertTrue(np.all(np.isclose(self.ub, u_expected)))
        self.assertTrue(np.all(np.isclose(self.uc, u_expected)))

    def test_works_with_xarray(self):
        lon = xr.DataArray(data=self.lon, coords={"time": self.time})
        lat = xr.DataArray(data=self.lat, coords={"time": self.time})
        time = xr.DataArray(data=self.time, coords={"time": self.time})
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == self.uf))
        self.assertTrue(np.all(vf == self.vf))

    def test_works_with_2d_array(self):
        lon = np.reshape(np.tile(self.lon, 4), (4, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 4), (4, self.lat.size))
        time = np.reshape(np.tile(self.time, 4), (4, self.time.size))
        expected_uf = np.reshape(np.tile(self.uf, 4), (4, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 4), (4, self.vf.size))
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))

    def test_works_with_3d_array(self):
        lon = np.reshape(np.tile(self.lon, 4), (2, 2, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 4), (2, 2, self.lat.size))
        time = np.reshape(np.tile(self.time, 4), (2, 2, self.time.size))
        expected_uf = np.reshape(np.tile(self.uf, 4), (2, 2, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 4), (2, 2, self.vf.size))
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))

    def test_works_with_3d_positions_1d_time(self):
        lon = np.reshape(np.tile(self.lon, 4), (2, 2, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 4), (2, 2, self.lat.size))
        expected_uf = np.reshape(np.tile(self.uf, 4), (2, 2, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 4), (2, 2, self.vf.size))
        uf, vf = velocity_from_position(lon, lat, self.time)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))

    def test_time_axis(self):
        lon = np.reshape(np.tile(self.lon, 6), (2, 3, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 6), (2, 3, self.lat.size))
        time = np.reshape(np.tile(self.time, 6), (2, 3, self.time.size))
        expected_uf = np.reshape(np.tile(self.uf, 6), (2, 3, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 6), (2, 3, self.vf.size))

        for time_axis in [0, 1, 2]:
            # Pass inputs with swapped axes and differentiate along that time
            # axis.
            uf, vf = velocity_from_position(
                np.swapaxes(lon, time_axis, -1),
                np.swapaxes(lat, time_axis, -1),
                np.swapaxes(time, time_axis, -1),
                time_axis=time_axis,
            )

            # Swap axes back to compare with the expected result.
            self.assertTrue(np.all(np.swapaxes(uf, time_axis, -1) == expected_uf))
            self.assertTrue(np.all(np.swapaxes(vf, time_axis, -1) == expected_vf))
            self.assertTrue(
                np.all(np.swapaxes(uf, time_axis, -1).shape == expected_uf.shape)
            )
            self.assertTrue(
                np.all(np.swapaxes(vf, time_axis, -1).shape == expected_uf.shape)
            )


class spin_tests(unittest.TestCase):
    def test_simple_spin(self):
        u = np.array([0.16622662, 0.92834973, 0.83375613, 0.99353467])
        v = np.array([0.91568661, 0.71860701, 0.69412382, 0.78336003])
        time = np.arange(4.0)
        self.assertTrue(
            np.allclose(
                spin(u, v, time),
                np.array([-0.84356843, 0.03282956, -0.0310163, -0.02280464]),
            )
        )

    def test_known_zero_spin(self):
        u = np.ones((10))
        v = np.ones((10))
        time = np.arange(10.0)
        self.assertTrue(np.all(spin(u, v, time) == 0))

    def test_multidim_inputs(self):
        u = np.ones((4, 3))
        v = np.ones((4, 3))
        time = np.arange(3.0)
        self.assertTrue(spin(u, v, time).shape == (4, 3))
        self.assertTrue(np.all(spin(u, v, time) == 0))

    def test_raises(self):
        with self.assertRaises(AttributeError):
            spin(0, 0, 0)
        with self.assertRaises(AttributeError):
            spin([0], [0], [0])
        with self.assertRaises(IndexError):
            spin(np.array([0]), np.array([0]), np.array([0]))
        with self.assertRaises(ValueError):
            spin(np.ones((4)), np.ones((4)), np.arange(3.0))
        with self.assertRaises(ValueError):
            u = np.random.random((4, 3))
            v = np.random.random((4, 3))
            time = np.arange((4.0))
            spin(u, v, time)
        with self.assertRaises(ValueError):
            spin(np.ones((4)), np.ones((4)), np.arange(4.0), time_axis=2)

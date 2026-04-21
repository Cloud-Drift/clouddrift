import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from clouddrift.adapters import gdp6h
from tests.adapters.utils import MultiPatcher


class gdp6h_tests(unittest.TestCase):
    drifter_files: list[str]
    response_mock: Mock
    gdp_mock: Mock

    def setUp(self) -> None:
        super().setUp()
        self.drifter_files = [
            "drifter_6h_0.nc",
            "drifter_6h_1.nc",
            "drifter_6h_2.nc",
            "drifter_6h_3.nc",
            "drifter_6h_4.nc",
        ]
        mock_drifter_ids_str = "\n".join(self.drifter_files)
        data_mock = Mock()
        data_mock.decode = Mock(return_value=mock_drifter_ids_str)
        self.response_mock = Mock()
        self.response_mock.read = Mock(return_value=data_mock)
        self.gdp_mock = Mock()
        self.gdp_mock.order_by_date = Mock(side_effect=lambda _, y: y)

    def test_downloads_all_files_returned(self):
        """
        If neither drifter ids or n_random_id (randomized sample size) are provided as arguments,
        download all of the files.
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp6h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp6h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp6h.gdp", self.gdp_mock),
            ]
        ):
            ret_drifter_ids = gdp6h.download("some-url.com", "../some/path", None, None)
            assert len(ret_drifter_ids) == len(self.drifter_files)

    def test_downloads_subset_of_files_returned(self):
        """
        If n_random_id is provided, exactly `n_random_id` drifter files should be downloaded
        """

        gdp_mock = Mock()
        gdp_mock.gdp_order_by_date = lambda _, y: y
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp6h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp6h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp6h.gdp", self.gdp_mock),
            ]
        ):
            ret_drifter_ids = gdp6h.download("some-url.com", "../some/path", None, 2)
            assert len(ret_drifter_ids) == 2

    def test_downloads_selected_drifter_id_files(self):
        """
        If drifter_ids are provided, download only those drifter files
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp6h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp6h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp6h.gdp", self.gdp_mock),
            ]
        ) as mocks:
            drifter_ids = [0, 1, 2]
            ret_drifter_ids = gdp6h.download(
                "some-url.com", "../some/path", drifter_ids, None
            )
            assert len(ret_drifter_ids) == 3
            mocks[1].assert_called_with(
                [
                    (
                        f"some-url.com/netcdf_1_5000/drifter_6h_{did}.nc",
                        os.path.join("../some/path", f"drifter_6h_{did}.nc"),
                    )
                    for did in ret_drifter_ids
                ]
            )

    def test_skip_download_uses_local_files(self):
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.os.walk",
                    Mock(
                        return_value=[
                            ("../some/path", [], self.drifter_files),
                        ]
                    ),
                ),
                patch("clouddrift.adapters.gdp6h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp6h.urllib.request.urlopen", Mock()),
            ]
        ) as mocks:
            ret_drifter_ids = gdp6h.download(
                "some-url.com", "../some/path", None, None, skip_download=True
            )

            assert ret_drifter_ids == [0, 1, 2, 3, 4]
            mocks[1].assert_not_called()
            mocks[2].assert_not_called()

    def test_skip_download_prints_local_mode_message(self):
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.os.walk",
                    Mock(return_value=[("../some/path", [], self.drifter_files)]),
                ),
                patch("clouddrift.adapters.gdp6h.print", Mock()),
            ]
        ) as mocks:
            gdp6h.download(
                "some-url.com", "../some/path", None, None, skip_download=True
            )

            mocks[1].assert_called_once_with(
                "Using local GDP 6-hourly files from ../some/path..."
            )

    def test_skip_download_requires_requested_local_files(self):
        with patch(
            "clouddrift.adapters.gdp6h.os.walk",
            Mock(return_value=[("../some/path", [], self.drifter_files[:2])]),
        ):
            with self.assertRaises(FileNotFoundError):
                gdp6h.download(
                    "some-url.com",
                    "../some/path",
                    [0, 1, 2],
                    None,
                    skip_download=True,
                )

    def test_skip_download_uses_nested_directories(self):
        with patch(
            "clouddrift.adapters.gdp6h.os.walk",
            Mock(
                return_value=[
                    ("../some/path", ["netcdf_1_5000", "netcdf_5001_10000"], []),
                    (
                        "../some/path/netcdf_1_5000",
                        [],
                        ["drifter_6h_0.nc", "drifter_6h_1.nc"],
                    ),
                    ("../some/path/netcdf_5001_10000", [], ["drifter_6h_5001.nc"]),
                ]
            ),
        ):
            ret_drifter_ids = gdp6h.download(
                "some-url.com", "../some/path", None, None, skip_download=True
            )

            assert ret_drifter_ids == [0, 1, 5001]

    def test_resolve_drifter_path_uses_local_file_map(self):
        file_path = gdp6h._resolve_drifter_path(
            5001,
            tmp_path="../some/path",
            filename_pattern="drifter_6h_{id}.nc",
            local_file_map={5001: "../some/path/netcdf_5001_10000/drifter_6h_5001.nc"},
        )

        assert file_path == "../some/path/netcdf_5001_10000/drifter_6h_5001.nc"

    def test_skip_download_orders_by_local_metadata_start_date(self):
        metadata_df = pd.DataFrame(
            {
                "ID": [2, 0, 1],
                "Start_date": pd.to_datetime(
                    [
                        "2020-01-01 00:00",
                        "2020-01-02 00:00",
                        "2020-01-03 00:00",
                    ]
                ),
            }
        )

        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp6h.os.walk",
                    Mock(return_value=[("../some/path", [], self.drifter_files[:3])]),
                ),
                patch(
                    "clouddrift.adapters.gdp6h._get_local_gdp_metadata",
                    Mock(return_value=metadata_df),
                ),
            ]
        ):
            ret_drifter_ids = gdp6h.download(
                "some-url.com", "../some/path", None, None, skip_download=True
            )

            assert ret_drifter_ids == [2, 0, 1]

    def test_get_local_file_map_warns_on_duplicate_ids(self):
        with patch(
            "clouddrift.adapters.gdp6h.os.walk",
            Mock(
                return_value=[
                    ("../some/path", ["a", "b"], []),
                    ("../some/path/a", [], ["drifter_6h_0.nc"]),
                    ("../some/path/b", [], ["drifter_6h_0.nc"]),
                ]
            ),
        ):
            with self.assertWarns(UserWarning):
                file_map = gdp6h._get_local_file_map("../some/path")

            assert os.path.normpath(file_map[0]) == os.path.normpath(
                os.path.join("../some/path", "a", "drifter_6h_0.nc")
            )

    def test_skip_download_orders_by_real_local_metadata_file(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            # local drifter files
            for did in [10, 20, 30]:
                open(os.path.join(tmp_path, f"drifter_6h_{did}.nc"), "w").close()

            # minimal dirfl file with 15 columns expected by parser logic
            with open(os.path.join(tmp_path, "dirfl_1_current.dat"), "w") as f:
                f.write(
                    "10 0 0 0 2020/01/03 00:00 0 0 2020/01/04 00:00 0 0 2020/01/05 00:00 0\n"
                )
                f.write(
                    "20 0 0 0 2020/01/01 00:00 0 0 2020/01/02 00:00 0 0 2020/01/03 00:00 0\n"
                )
                f.write(
                    "30 0 0 0 2020/01/02 00:00 0 0 2020/01/03 00:00 0 0 2020/01/04 00:00 0\n"
                )

            ret_drifter_ids = gdp6h.download(
                "some-url.com", tmp_path, None, None, skip_download=True
            )

            assert ret_drifter_ids == [20, 30, 10]

    def test_skip_download_warns_and_appends_ids_missing_in_local_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            for did in [10, 20]:
                open(os.path.join(tmp_path, f"drifter_6h_{did}.nc"), "w").close()

            with open(os.path.join(tmp_path, "dirfl_1_current.dat"), "w") as f:
                f.write(
                    "20 0 0 0 2020/01/01 00:00 0 0 2020/01/02 00:00 0 0 2020/01/03 00:00 0\n"
                )

            with self.assertWarns(UserWarning):
                ret_drifter_ids = gdp6h.download(
                    "some-url.com", tmp_path, None, None, skip_download=True
                )

            assert ret_drifter_ids == [20, 10]

    def test_skip_download_skips_malformed_local_metadata_files(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            for did in [10, 20]:
                open(os.path.join(tmp_path, f"drifter_6h_{did}.nc"), "w").close()

            with open(os.path.join(tmp_path, "dirfl_1_5000.dat"), "w") as f:
                f.write(
                    "20 0 0 0 2020/01/01 00:00 0 0 2020/01/02 00:00 0 0 2020/01/03 00:00 0\n"
                )

            with open(os.path.join(tmp_path, "dirfl_5001_10000.dat"), "w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<title>404</title>\n")

            with self.assertWarns(UserWarning):
                ret_drifter_ids = gdp6h.download(
                    "some-url.com", tmp_path, None, None, skip_download=True
                )

            assert ret_drifter_ids == [20, 10]

    def test_resolve_drifter_path_raises_for_missing_id_in_local_map(self):
        with self.assertRaises(FileNotFoundError):
            gdp6h._resolve_drifter_path(
                99,
                tmp_path="../some/path",
                filename_pattern="drifter_6h_{id}.nc",
                local_file_map={1: "../some/path/drifter_6h_1.nc"},
            )

    def test_rowsize_returns_zero_when_dataset_open_fails(self):
        with patch("clouddrift.adapters.gdp6h.xr.open_dataset", side_effect=OSError):
            rowsize = gdp6h._rowsize(
                1,
                tmp_path="../some/path",
                filename_pattern="drifter_6h_{id}.nc",
                local_file_map={1: "../some/path/drifter_6h_1.nc"},
            )

            assert rowsize == 0

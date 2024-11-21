import os
import unittest
from unittest.mock import Mock, patch

from clouddrift.adapters import gdp1h
from tests.adapters.utils import MultiPatcher


class gdp1h_tests(unittest.TestCase):
    drifter_files: list[str]
    response_mock: Mock
    gdp_mock: Mock

    def setUp(self) -> None:
        super().setUp()
        self.drifter_files = [
            "drifter_hourly_0.nc",
            "drifter_hourly_1.nc",
            "drifter_hourly_2.nc",
            "drifter_hourly_3.nc",
            "drifter_hourly_4.nc",
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
                    "clouddrift.adapters.gdp1h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp1h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp1h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp1h.gdp", self.gdp_mock),
            ]
        ):
            ret_drifter_ids = gdp1h.download("some-url.com", "../some/path", None, None)
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
                    "clouddrift.adapters.gdp1h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp1h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp1h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp1h.gdp", self.gdp_mock),
            ]
        ):
            ret_drifter_ids = gdp1h.download("some-url.com", "../some/path", None, 2)
            assert len(ret_drifter_ids) == 2

    def test_downloads_selected_drifter_id_files(self):
        """
        If drifter_ids are provided, download only those drifter files
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp1h.urllib.request.urlopen",
                    Mock(return_value=self.response_mock),
                ),
                patch("clouddrift.adapters.gdp1h.download_with_progress", Mock()),
                patch("clouddrift.adapters.gdp1h.os.makedirs", Mock()),
                patch("clouddrift.adapters.gdp1h.gdp", self.gdp_mock),
            ]
        ) as mocks:
            drifter_ids = [0, 1, 2]
            ret_drifter_ids = gdp1h.download(
                "some-url.com", "../some/path", drifter_ids, None
            )
            assert len(ret_drifter_ids) == 3
            mocks[1].assert_called_with(
                [
                    (
                        f"some-url.com/drifter_hourly_{did}.nc",
                        os.path.join("../some/path", f"drifter_hourly_{did}.nc"),
                    )
                    for did in drifter_ids
                ]
            )

    def test_verify_identifies_zero_byte_files(self):
        """
        Test that the verify function correctly identifies files with 0 bytes
        within a given directory.
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp1h.os.listdir",
                    Mock(return_value=self.drifter_files),
                ),
                patch(
                    "clouddrift.adapters.gdp1h.os.path.getsize",
                    side_effect=lambda x: 0 if "2.nc" in x else 100,
                ),
            ]
        ):
            zero_byte_files = gdp1h.verify("../some/path")
            assert zero_byte_files == [2]

    def test_fix_redownloads_zero_byte_files(self):
        """
        Test that the fix function redownloads files identified as 0 bytes by the verify function.
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp1h.verify", Mock(side_effect=[[2, 3], []])
                ),
                patch("clouddrift.adapters.gdp1h.download", Mock()),
            ]
        ) as mocks:
            result = gdp1h.fix("some-url.com", "../some/path")
            mocks[1].assert_called_with("some-url.com", "../some/path", [2, 3], None)
            assert result == 0

    def test_fix_returns_files_still_needing_fix(self):
        """
        Test that the fix function returns the files that still need fixing if redownloading fails.
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.gdp1h.verify",
                    Mock(side_effect=[[2, 3], [2], [2]]),
                ),
                patch("clouddrift.adapters.gdp1h.download", Mock()),
            ]
        ) as mocks:
            result = gdp1h.fix("some-url.com", "../some/path")
            mocks[1].assert_any_call("some-url.com", "../some/path", [2, 3], None)
            mocks[1].assert_any_call("some-url.com", "../some/path", [2], None)
            assert result == 1  # Expecting 1 files still needing fix

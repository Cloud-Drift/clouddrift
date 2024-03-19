import os
import unittest
from unittest.mock import Mock, patch

import clouddrift.adapters.gdp6h as gdp6h
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
                    "clouddrift.adapters.gdp1h.urllib.request.urlopen",
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
                        None,
                    )
                    for did in ret_drifter_ids
                ]
            )

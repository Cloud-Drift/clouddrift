import os
import unittest
from unittest.mock import Mock, patch

from clouddrift.adapters import gdp6h


class gdp6h_tests(unittest.TestCase):
    drifter_files: list[str]
    response_mock: Mock

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

    def test_downloads_all_files_returned(self):
        """
        If neither drifter ids or n_random_id (randomized sample size) are provided as arguments,
        download all of the files.
        """
        with patch(
            "clouddrift.adapters.gdp6h.urllib.request.urlopen",
            Mock(return_value=self.response_mock),
        ):
            download_requests = gdp6h._get_download_requests(
                "some-url.com", "../some/path", None, None
            )
            assert len(download_requests) == len(self.drifter_files)

    def test_downloads_subset_of_files_returned(self):
        """
        If n_random_id is provided, exactly `n_random_id` drifter files should be downloaded
        """

        with patch(
            "clouddrift.adapters.gdp6h.urllib.request.urlopen",
            Mock(return_value=self.response_mock),
        ):
            download_requests = gdp6h._get_download_requests(
                "some-url.com", "../some/path", None, 2
            )
            assert len(download_requests) == 2

    def test_downloads_selected_drifter_id_files(self):
        """
        If drifter_ids are provided, download only those drifter files
        """
        with patch(
            "clouddrift.adapters.gdp6h.urllib.request.urlopen",
            Mock(return_value=self.response_mock),
        ):
            drifter_ids = [0, 1, 2]
            valid_urls = [
                f"some-url.com/netcdf_1_5000/drifter_6h_{did}.nc" for did in drifter_ids
            ]
            valid_paths = [
                os.path.join("../some/path", f"drifter_6h_{did}.nc")
                for did in drifter_ids
            ]
            download_requests = gdp6h._get_download_requests(
                "some-url.com", "../some/path", drifter_ids, None
            )
            assert len(download_requests) == 3

            for url, path, _ in download_requests:
                assert url in valid_urls
                assert path in valid_paths

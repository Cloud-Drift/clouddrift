import unittest

from clouddrift.adapters import hurdat2


class hurdat2_tests(unittest.TestCase):
    def test_basin_selection(self) -> None:
        with self.subTest("Select Atlantic Basin"):
            download_requests = hurdat2._get_download_requests("atlantic", "some_path")
            url, _, _ = download_requests[0]

            assert len(download_requests) == 1
            assert url == hurdat2._ATLANTIC_BASIN_URL

        with self.subTest("Select Pacific Basin"):
            download_requests = hurdat2._get_download_requests("pacific", "some_path")
            url, _, _ = download_requests[0]

            assert len(download_requests) == 1
            assert url == hurdat2._PACIFIC_BASIN_URL

        with self.subTest("Select Both Option"):
            download_requests = hurdat2._get_download_requests("both", "some_path")
            atlantic_url, _, _ = download_requests[0]
            pacific_url, _, _ = download_requests[1]

            assert len(download_requests) == 2
            assert atlantic_url == hurdat2._ATLANTIC_BASIN_URL
            assert pacific_url == hurdat2._PACIFIC_BASIN_URL

import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

import clouddrift.adapters.gdp as gdp


class gdp_common_tests(unittest.TestCase):
    def test_list_gdp_directory_files_from_index(self):
        html = "\n".join(
            [
                '<a href="dirfl_10001_15000.dat">dirfl_10001_15000.dat</a>',
                '<a href="dirfl_1_5000.dat">dirfl_1_5000.dat</a>',
                '<a href="dirfl_5001_10000.dat">dirfl_5001_10000.dat</a>',
                '<a href="dirfl_15001_current.dat">dirfl_15001_current.dat</a>',
            ]
        )
        response = MagicMock()
        response.read = Mock(return_value=html.encode("utf-8"))
        response.__enter__.return_value = response
        response.__exit__.return_value = None

        with patch(
            "clouddrift.adapters.gdp.urllib.request.urlopen",
            Mock(return_value=response),
        ):
            files = gdp._list_gdp_directory_files()

        assert files == [
            "dirfl_1_5000.dat",
            "dirfl_5001_10000.dat",
            "dirfl_10001_15000.dat",
            "dirfl_15001_current.dat",
        ]

    def test_get_gdp_metadata_uses_discovered_files(self):
        with (
            patch(
                "clouddrift.adapters.gdp._list_gdp_directory_files",
                Mock(return_value=["dirfl_1_5000.dat", "dirfl_15001_current.dat"]),
            ),
            patch(
                "clouddrift.adapters.gdp.parse_directory_file",
                Mock(
                    side_effect=[
                        pd.DataFrame(
                            {"ID": [1], "Start_date": [pd.Timestamp("2020-01-02")]}
                        ),
                        pd.DataFrame(
                            {"ID": [2], "Start_date": [pd.Timestamp("2020-01-01")]}
                        ),
                    ]
                ),
            ) as parse_mock,
        ):
            df = gdp.get_gdp_metadata(tmp_path="/tmp/some-path")

        assert parse_mock.call_count == 2
        assert [c.args[0] for c in parse_mock.call_args_list] == [
            "dirfl_1_5000.dat",
            "dirfl_15001_current.dat",
        ]
        assert df.ID.tolist() == [2, 1]

    def test_get_gdp_metadata_skip_download_uses_local_files(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_path:
            for name in ["dirfl_1_5000.dat", "dirfl_5001_current.dat"]:
                open(os.path.join(tmp_path, name), "w").close()

            with (
                patch(
                    "clouddrift.adapters.gdp._list_gdp_directory_files",
                ) as list_mock,
                patch(
                    "clouddrift.adapters.gdp.parse_directory_file",
                    Mock(
                        side_effect=[
                            pd.DataFrame(
                                {"ID": [1], "Start_date": [pd.Timestamp("2020-01-02")]}
                            ),
                            pd.DataFrame(
                                {"ID": [2], "Start_date": [pd.Timestamp("2020-01-01")]}
                            ),
                        ]
                    ),
                ) as parse_mock,
            ):
                df = gdp.get_gdp_metadata(tmp_path=tmp_path, skip_download=True)

        list_mock.assert_not_called()
        assert parse_mock.call_count == 2
        assert df.ID.tolist() == [2, 1]

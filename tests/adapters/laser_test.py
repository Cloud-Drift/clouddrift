import os
import tempfile
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from clouddrift.adapters import laser


def _build_test_archive() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, mode="w") as archive:
        archive.writestr(
            "ReadMe.txt",
            "LASER sample archive",
        )
        archive.writestr(
            laser._DATA_FILENAME,
            "\n".join(
                [
                    "% header",
                    "% header",
                    "L_0002 2016-01-21 19:00:00.000000 29.1 -87.6 10.0 0.10 0.20 0.03",
                    "L_0001 2016-01-21 18:45:00.000000 29.0 -87.7 11.0 0.01 0.02 0.04",
                    "L_0001 2016-01-21 19:00:00.000000 29.2 -87.5 12.0 0.03 0.04 0.05",
                ]
            ),
        )
    return payload.getvalue()


class laser_tests(TestCase):
    def _mock_download(self, download_map, **_kwargs):
        _, output = download_map[0]
        if isinstance(output, str):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "wb") as f:
                f.write(_build_test_archive())
        else:
            output.write(_build_test_archive())

    def test_get_dataframe_parses_archive_and_sorts_rows(self):
        tmp_path = tempfile.mkdtemp()
        with patch(
            "clouddrift.adapters.laser.download_with_progress",
            side_effect=self._mock_download,
        ):
            df = laser.get_dataframe(tmp_path=tmp_path)

        self.assertEqual(
            list(df.columns),
            [
                "id",
                "latitude",
                "longitude",
                "position_error",
                "u",
                "v",
                "velocity_error",
                "obs",
            ],
        )
        self.assertEqual(df["id"].tolist(), ["L_0001", "L_0001", "L_0002"])
        self.assertEqual(df["obs"].dt.year.tolist(), [2016, 2016, 2016])

    def test_to_xarray_returns_ragged_dataset(self):
        tmp_path = tempfile.mkdtemp()
        with patch(
            "clouddrift.adapters.laser.download_with_progress",
            side_effect=self._mock_download,
        ):
            ds = laser.to_xarray(tmp_path=tmp_path)

        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 3)
        self.assertIn("time", ds.coords)
        self.assertIn("id", ds.coords)
        self.assertTrue(np.array_equal(ds["rowsize"].values, np.array([2, 1])))
        self.assertEqual(ds["latitude"].dtype, np.float32)
        self.assertEqual(ds["longitude"].dtype, np.float32)
        self.assertEqual(ds.attrs["title"], laser._DATASET_TITLE)

    def test_get_dataframe_forwards_skip_download(self):
        tmp_path = tempfile.mkdtemp()

        with patch(
            "clouddrift.adapters.laser.download_with_progress",
            side_effect=self._mock_download,
        ) as download_mock:
            laser.get_dataframe(tmp_path=tmp_path, skip_download=True)

        self.assertTrue(download_mock.call_args.kwargs["skip_download"])

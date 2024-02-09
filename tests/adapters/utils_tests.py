import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, mock_open, patch

from clouddrift.adapters import utils
from tests.adapters.utils import MultiPatcher


class utils_tests(unittest.TestCase):
    head_response_mock: Mock
    get_response_mock: Mock
    requests_mock: Mock
    open_mock: Mock
    bar_mock: Mock

    def setUp(self) -> None:
        self.head_response_mock = Mock()
        self.head_response_mock.headers = {"Last-Modified": "some-date-str"}
        self.head_response_mock.__enter__ = Mock(return_value=self.head_response_mock)
        self.head_response_mock.__exit__ = Mock()

        self.get_response_mock = Mock()
        self.get_response_mock.headers = dict()
        self.get_response_mock.iter_content = Mock(return_value=["a", "b", "c"])

        self.requests_mock = Mock()
        self.requests_mock.head = Mock(return_value=self.head_response_mock)
        self.requests_mock.get = Mock(return_value=self.get_response_mock)

        self.open_mock = mock_open()

        self.bar_mock = Mock()
        self.bar_mock.update = Mock()

    def test_forgo_download_no_update(self):
        """
        Don't download from the server if we have the latest version of the file.
        """

        now = datetime.now()
        datetime_mock = Mock()
        datetime_mock.strptime = Mock(return_value=now)

        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.datetime", datetime_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.getmtime",
                    Mock(return_value=now.timestamp()),
                ),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=True)
                ),
            ]
        ) as _:
            utils._download_with_progress(
                "some.url.com", "./some/path/existing-file.nc", 0, False, lambda x: x
            )
            self.requests_mock.get.assert_not_called()

    def test_download_new_update(self):
        """
        Download the file from the server if there is an update to it remotely.
        """
        now = datetime.now()
        datetime_mock = Mock()
        datetime_mock.strptime = Mock(return_value=now)
        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.datetime", datetime_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=True)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getmtime",
                    Mock(return_value=(now - timedelta(days=1)).timestamp()),
                ),
            ]
        ) as _:
            utils._download_with_progress(
                "some.url.com", "./some/path/existing-file.nc", 0, False, lambda x: x
            )
            self.requests_mock.get.assert_called()

    def test_progress_mechanism_enabled(self):
        """
        Ensure we show progress when user enables the feature
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.tqdm", Mock(return_value=self.bar_mock)
                ),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
            ]
        ) as mock:
            utils._download_with_progress(
                "some.url.com", "./some/path/existing-file.nc", 0, True, lambda x: x
            )
            mock[0].assert_called()
            self.bar_mock.update.assert_called()

    def test_progress_mechanism_disabled(self):
        """
        Ensure we don't show progress when user disables the feature
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.tqdm", Mock(return_value=self.bar_mock)
                ),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
            ]
        ) as mock:
            utils._download_with_progress(
                "some.url.com", "./some/path/existing-file.nc", 0, False, lambda x: x
            )
            mock[0].assert_not_called()
            self.bar_mock.update.assert_not_called()

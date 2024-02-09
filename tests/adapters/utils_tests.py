import concurrent.futures
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

    def test_progress_mechanism_disabled_files(self):
        """
        Ensure we don't show progress for the list of files when number of files is less than 20 and user 
        does not specify whether the progress feature is enabled/disabled.
        """
        mocked_futures_list = list()
        def gen_future_mock():
            nonlocal mocked_futures_list
            fut = concurrent.futures.Future()
            mocked_futures_list.append(fut)
            return fut

        tpe_mock = Mock()
        tpe_mock.__enter__ = Mock(return_value=tpe_mock)
        tpe_mock.__exit__ = Mock()
        tpe_mock.submit = Mock(side_effect=lambda *_, **__: gen_future_mock())
        futures_mock = Mock()
        futures_mock.ThreadPoolExecutor = Mock(return_value=tpe_mock)
        futures_mock.as_completed = Mock(return_value=mocked_futures_list)
        download_requests = [
            ("src0", "dst", None),
            ("src1", "dst", None),
            ("src2", "dst", None)
        ]

        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.tqdm", Mock(return_value=self.bar_mock)
                ),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
            ]
        ) as mock:
            utils.download_with_progress(download_requests)
            mock[0].assert_not_called()
            self.bar_mock.update.assert_not_called()

    def test_progress_mechanism_enabled_files(self):
        """
        Ensure we do show progress for the list of files when number of files is greater than 20 and user 
        does not specify whether the progress feature is enabled/disabled.
        """
        mocked_futures_list = list()
        def gen_future_mock():
            nonlocal mocked_futures_list
            fut = Mock()
            fut.exception = Mock(return_value=None)
            mocked_futures_list.append(fut)
            return fut

        tpe_mock = Mock()
        tpe_mock.__enter__ = Mock(return_value=tpe_mock)
        tpe_mock.__exit__ = Mock()
        tpe_mock.submit = Mock(side_effect=lambda *_, **__: gen_future_mock())
        futures_mock = Mock()
        futures_mock.ThreadPoolExecutor = Mock(return_value=tpe_mock)
        futures_mock.as_completed = Mock(return_value=mocked_futures_list)
        download_requests = [
            ("src0", "dst", None) for _ in range(0, 21)
        ]

        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.tqdm", Mock(return_value=self.bar_mock)
                ),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
            ]
        ) as mock:
            utils.download_with_progress(download_requests)
            mock[0].assert_called()
            self.bar_mock.update.assert_called()

    def test_progress_mechanism_enabled_file(self):
        """
        Ensure we do show progress per file when user enables the feature
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

    def test_progress_mechanism_disabled_file(self):
        """
        Ensure we don't show progress per file when user disables the feature
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

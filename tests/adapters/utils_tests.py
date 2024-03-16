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
                "some.url.com", "./some/path/existing-file.nc", 0, False
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
                "some.url.com", "./some/path/existing-file.nc", 0, False
            )
            self.requests_mock.get.assert_called()

    def test_progress_mechanism_disabled_files(self):
        """
        Ensure we don't show progress for the list of files when number of files is less than 20 and user
        does not specify whether the progress feature is enabled/disabled.
        """
        mocked_futures = [self.gen_future_mock() for _ in range(0, 3)]
        download_requests = [
            ("src0", "dst", None),
            ("src1", "dst", None),
            ("src2", "dst", None),
        ]

        tpe_mock = Mock()
        tpe_mock.__enter__ = Mock(return_value=tpe_mock)
        tpe_mock.__exit__ = Mock()
        tpe_mock.submit = Mock(side_effect=mocked_futures)

        futures_mock = Mock()
        futures_mock.ThreadPoolExecutor = Mock(return_value=tpe_mock)
        futures_mock.as_completed = Mock(return_value=mocked_futures)

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

        mocked_futures = [self.gen_future_mock() for _ in range(0, 21)]
        download_requests = [("src0", "dst", None) for _ in range(0, 21)]

        tpe_mock = Mock()
        tpe_mock.__enter__ = Mock(return_value=tpe_mock)
        tpe_mock.__exit__ = Mock()
        tpe_mock.submit = Mock(side_effect=mocked_futures)

        futures_mock = Mock()
        futures_mock.ThreadPoolExecutor = Mock(return_value=tpe_mock)
        futures_mock.as_completed = Mock(return_value=mocked_futures)

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
                "some.url.com", "./some/path/existing-file.nc", 0, True
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
                "some.url.com", "./some/path/existing-file.nc", 0, False
            )
            mock[0].assert_not_called()
            self.bar_mock.update.assert_not_called()

    def test_raises_on_any_exception_and_cleanup(self):
        """
        Ensure that any download job exception is propogated up and all download jobs are cancelled and cleaned up.
        """

        tpe_mock = Mock()
        tpe_mock.shutdown = Mock()
        tpe_mock.__enter__ = Mock(return_value=tpe_mock)
        tpe_mock.__exit__ = Mock()

        mocked_futures = [
            self.gen_future_mock(),
            self.gen_future_mock(),
            self.gen_future_mock(Exception("just a test exception that is expected")),
            self.gen_future_mock(),
        ]

        futures_mock = Mock()
        futures_mock.ThreadPoolExecutor = Mock(return_value=tpe_mock)
        futures_mock.as_completed = Mock(return_value=mocked_futures)
        tpe_mock.submit = Mock(side_effect=mocked_futures)

        os_mock = Mock()
        os_mock.remove = Mock()

        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.tqdm", Mock(return_value=self.bar_mock)
                ),
                patch("clouddrift.adapters.utils.open", self.open_mock),
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
                patch("clouddrift.adapters.utils.requests", self.requests_mock),
                patch("clouddrift.adapters.utils.os", os_mock),
            ]
        ) as _:
            self.assertRaises(
                Exception,
                utils.download_with_progress,
                [("src", "dst", None) for _ in range(0, len(mocked_futures))],
                show_list_progress=True,
            )
            assert tpe_mock.submit.call_count == len(mocked_futures)
            assert self.bar_mock.update.call_count == 2
            assert os_mock.remove.call_count == len(mocked_futures)
            tpe_mock.shutdown.assert_called_once()
            [fut_mock.cancel.assert_called_once() for fut_mock in mocked_futures]

    def gen_future_mock(self, ex=None):
        future = Mock()
        future.exception = Mock(return_value=ex)
        future.cancel = Mock()
        return future

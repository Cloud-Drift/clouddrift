import unittest
from datetime import datetime, timedelta
from io import BufferedIOBase
from unittest.mock import Mock, call, mock_open, patch

from requests import RequestException

from clouddrift.adapters import utils
from tests.adapters.utils import MultiPatcher


class utils_tests(unittest.TestCase):
    head_response_mock: Mock
    get_response_mock: Mock
    requests_mock: Mock
    open_mock: Mock
    bar_mock: Mock

    def setUp(self) -> None:
        """
        Set up the mocks for the tests.
        """

        # Mock responses for requests.head and requests.get
        self.head_response_mock = Mock()
        self.head_response_mock.headers = {"Last-Modified": "some-date-str"}
        self.head_response_mock.__enter__ = Mock(return_value=self.head_response_mock)
        self.head_response_mock.__exit__ = Mock()

        self.get_response_mock = Mock()
        self.get_response_mock.headers = {"Content-Length": "4"}
        self.get_response_mock.iter_content = Mock(
            return_value=[b"a", b"b", b"c", b"d"]
        )
        self.get_response_mock.__enter__ = Mock(return_value=self.get_response_mock)
        self.get_response_mock.__exit__ = Mock()

        # Create a mock requests module
        self.requests_mock = Mock()
        self.requests_mock.head = Mock(return_value=self.head_response_mock)
        self.requests_mock.get = Mock(return_value=self.get_response_mock)
        self.requests_mock.RequestException = RequestException  # Assign real exception

        # Mock open
        self.open_mock = mock_open()

        # Mock tqdm progress bar
        self.bar_mock = Mock()
        self.bar_mock.update = Mock()

        # Patch 'requests' in 'clouddrift.adapters.utils' with 'self.requests_mock'
        self.requests_patcher = patch(
            "clouddrift.adapters.utils.requests", self.requests_mock
        )
        self.requests_patcher.start()

        # Patch 'open' in 'clouddrift.adapters.utils' with 'self.open_mock'
        self.open_patcher = patch("clouddrift.adapters.utils.open", self.open_mock)
        self.open_patcher.start()

        # Ensure patches are stopped after tests
        self.addCleanup(self.requests_patcher.stop)
        self.addCleanup(self.open_patcher.stop)

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

    def test_download_new_update_buffer(self):
        """
        Download the file from the server if there is an update to it remotely (BufferedIOBase).
        """

        now = datetime.now()
        datetime_mock = Mock()
        datetime_mock.strptime = Mock(return_value=now)
        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.datetime", datetime_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=True)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getmtime",
                    Mock(return_value=(now - timedelta(days=1)).timestamp()),
                ),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
            ]
        ):
            buffer = Mock(spec=BufferedIOBase)
            utils._download_with_progress("some.url.com", buffer, 0, False)

            # Assertions to verify the behavior
            self.requests_mock.get.assert_called_with(
                "some.url.com", timeout=10, stream=True
            )
            buffer.write.assert_has_calls(
                [call(b"a"), call(b"b"), call(b"c"), call(b"d")]
            )

    def test_download_new_update_file(self):
        """
        Download the file from the server if there is an update to it remotely (file path).
        """

        now = datetime.now()
        datetime_mock = Mock()
        datetime_mock.strptime = Mock(return_value=now)
        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.datetime", datetime_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=True)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getmtime",
                    Mock(return_value=(now - timedelta(days=1)).timestamp()),
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getsize", Mock(return_value=4)
                ),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
            ]
        ):
            output_file = "output.file"
            utils._download_with_progress("some.url.com", output_file, 0, False)

            # Assertions to verify the behavior
            self.requests_mock.get.assert_called_with(
                "some.url.com", timeout=10, stream=True
            )
            self.open_mock.assert_called_with(output_file + ".part", "wb")
            handle = self.open_mock()
            handle.write.assert_has_calls(
                [call(b"a"), call(b"b"), call(b"c"), call(b"d")]
            )

    def test_progress_mechanism_disabled_files(self):
        """
        Ensure we don't show progress for the list of files when number of files is less than 20 and user
        does not specify whether the progress feature is enabled/disabled.
        """

        mocked_futures = [self.gen_future_mock() for _ in range(0, 3)]
        download_requests = [
            ("src0", "dst"),
            ("src1", "dst"),
            ("src2", "dst"),
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
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
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
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
            ]
        ) as mock:
            utils.download_with_progress(download_requests)
            mock[0].assert_called()
            self.bar_mock.update.assert_called()

    def test_progress_mechanism_enabled_file(self):
        """
        Ensure we do show progress per file when the user enables the feature.
        """

        tqdm_mock = Mock(return_value=self.bar_mock)
        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.tqdm", tqdm_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=False)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getsize", Mock(return_value=4)
                ),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
            ]
        ):
            output_file = "output.file"
            utils._download_with_progress("some.url.com", output_file, 0, True)

            # Assertions
            tqdm_mock.assert_called_once()
            self.bar_mock.update.assert_has_calls([call(1), call(1), call(1), call(1)])
            self.requests_mock.get.assert_called_with(
                "some.url.com", timeout=10, stream=True
            )
            self.open_mock.assert_called_with(output_file + ".part", "wb")

    def test_progress_mechanism_disabled_file(self):
        """
        Ensure we don't show progress per file when the user disables the feature.
        """

        tqdm_mock = Mock(return_value=self.bar_mock)
        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.tqdm", tqdm_mock),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=False)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getsize", Mock(return_value=4)
                ),
                patch("clouddrift.adapters.utils.os.remove", Mock()),
                patch("clouddrift.adapters.utils.os.rename", Mock()),
            ]
        ):
            output_file = "output.file"
            utils._download_with_progress("some.url.com", output_file, 0, False)

            # Assertions
            tqdm_mock.assert_not_called()
            self.bar_mock.update.assert_not_called()
            self.requests_mock.get.assert_called_with(
                "some.url.com", timeout=10, stream=True
            )
            self.open_mock.assert_called_with(output_file + ".part", "wb")

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
            self.gen_future_mock(complete=True),
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
                patch("clouddrift.adapters.utils.concurrent.futures", futures_mock),
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
            [fut_mock.cancel.assert_called_once() for fut_mock in mocked_futures[:-1]]
            mocked_futures[-1].cancel.assert_not_called()

    def gen_future_mock(self, ex=None, complete=False):
        future = Mock()
        future.exception = Mock(return_value=ex)
        future.done = Mock(return_value=complete)
        future.cancel = Mock()
        return future

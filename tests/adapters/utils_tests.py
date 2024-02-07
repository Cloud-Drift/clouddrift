import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from requests import Timeout
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from clouddrift.adapters import utils
from tests.adapters.utils import MultiPatcher


class utils_tests(unittest.TestCase):
    def test_only_download_when_updated(self):
        """
        Only download files from remote server if local file doesn't exist or the file has been updated.
        """
        now = datetime.now().timestamp()
        response_mock = Mock()
        response_mock.headers = Mock(return_value={"Last-Modified": now})

        with MultiPatcher(
            [
                patch("clouddrift.adapters.utils.requests.get", Mock()),
                patch(
                    "clouddrift.adapters.utils.requests.head",
                    Mock(return_value=response_mock),
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=True)
                ),
                patch(
                    "clouddrift.adapters.utils.os.path.getmtime", Mock(return_value=now)
                ),
            ]
        ) as mocks:
            utils.download_with_progress(
                [("some.url.com", "./some/path/existing-file.nc", None)]
            )
            mocks[0].assert_not_called()
            mocks[1].assert_called()

    def test_retry_mechanism(self):
        """
        Ensure retry mechanism works.
        """
        with MultiPatcher(
            [
                patch(
                    "clouddrift.adapters.utils.os.path.exists", Mock(return_value=False)
                ),
                patch(
                    "clouddrift.adapters.utils.requests.get",
                    Mock(side_effect=Timeout()),
                ),
            ]
        ) as mocks:
            utils.download_with_progress(
                [("some.url.com", "./some/path/existing-file.nc", None)],
                custom_retry_protocol=retry(
                    wait=wait_fixed(timedelta(seconds=1)),
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception(lambda ex: isinstance(ex, (Timeout,))),
                ),
            )
            assert mocks[1].call_count == 3

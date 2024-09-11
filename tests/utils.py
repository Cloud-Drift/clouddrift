import unittest

import clouddrift.raggedarray as ragged
from clouddrift.adapters import utils


class DisableProgressTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        utils._DISABLE_SHOW_PROGRESS = True
        ragged._DISABLE_SHOW_PROGRESS = True

    def tearDown(self) -> None:
        super().tearDown()
        utils._DISABLE_SHOW_PROGRESS = False
        ragged._DISABLE_SHOW_PROGRESS = False

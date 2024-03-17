import doctest

import clouddrift.ragged as ragged


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocTestSuite(
            ragged,
            {
                "optionflag": doctest.ELLIPSIS | doctest.FAIL_FAST,
                "globs": ragged.__dict__,
            },
        )
    )
    return tests

import doctest

import clouddrift.ragged as ragged


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocTestSuite(
            ragged,
            optionflags=doctest.ELLIPSIS | doctest.FAIL_FAST,
        )
    )
    return tests

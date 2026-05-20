import pytest

# Exception message fragments that indicate a transient remote-data failure
# rather than a code bug. Keep in sync with datasets_test._is_transient_remote_failure.
_NETWORK_FAILURE_SIGNATURES = (
    "SSLEOFError",
    "Max retries exceeded",
    "RetryError",
    "EOF occurred in violation of protocol",
    "ConnectTimeout",
    "ConnectionError",
    "EmptyDataError",
    "ParserError",
    "Name contains illegal characters",
    "<?xml version=",
)


def _is_network_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(sig in msg for sig in _NETWORK_FAILURE_SIGNATURES)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.when == "call" and call.excinfo is not None:
        if _is_network_error(call.excinfo.value):
            rep.outcome = "skipped"
            rep.longrepr = (
                str(item.fspath),
                getattr(item, "lineno", None) or 0,
                f"Skipped: remote data source unavailable ({call.excinfo.value!r})",
            )

import logging
from importlib.metadata import PackageNotFoundError, distribution

_logger = logging.getLogger(__name__)

try:
    version = distribution("clouddrift").version
except PackageNotFoundError:
    # Handle case when developing on a local copy of the library rather than using an
    # installed copy from a distribution.
    version = "9999"
    _logger.info(f"Cannot determine package version using {version}")

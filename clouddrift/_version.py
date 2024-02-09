import logging

import pkg_resources

_logger = logging.getLogger(__name__)

try:
    version = pkg_resources.get_distribution("clouddrift").version
except pkg_resources.DistributionNotFound:
    # Handle case when developing on a local copy of the library rather than using an
    # installed copy from a distribution.
    version = "9999"
    _logger.info(f"Cannot determine package version using {version}")

import logging
import os
from typing import Union

import click

# from memory_profiler import profile  # type: ignore
from clouddrift import adapters

_logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command(help="Download individual NetCDF files from the AOML server")
@click.option(
    "--drifter-ids",
    type=str,
    default=None,
    help="List of drifter to retrieve (Default: all)",
)
@click.option(
    "--sample",
    type=int,
    default=None,
    help="Randomly select n_random_id drifter IDs to download (Default: None)",
)
@click.option(
    "--url",
    type=str,
    default=adapters.gdp1h.GDP_DATA_URL,
    help=f"URL from which to download the data (Default: {adapters.gdp1h.GDP_DATA_URL})",
)
@click.option(
    "--path",
    type=click.Path(exists=False),
    default=adapters.gdp1h.GDP_TMP_PATH,
    help=f"Path to the directory where the individual NetCDF files are stored (Default: {adapters.gdp1h.GDP_TMP_PATH})",
)
@click.option(
    "--experimental/--no-experimental",
    type=bool,
    default=False,
    help=f"If true will instead use the experimental URL.This will override the value passed to the url \
    option(Default: False). Experimental URL ({adapters.gdp1h.GDP_DATA_URL_EXPERIMENTAL})",
)
@click.option(
    "--aggregate/--no-aggregate",
    type=bool,
    default=True,
    help="If true will aggregate all of the drifter data into one NETCDF file. (Default: True)",
)
@click.option(
    "--load-existing/--no-load-existing",
    type=bool,
    default=False,
    help="If true will only load the files which have been downloaded to the path. (Default: False)",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Log level to be printed to stdout. (Default: INFO)",
)
def gdp1h(*args, **kwargs):
    _gdp1h(*args, **kwargs)


# @profile
def _gdp1h(
    drifter_ids: Union[str, None],
    sample: int,
    url: str,
    path: str,
    experimental: bool,
    aggregate: bool,
    load_existing: bool,
    log_level: str,
):
    logging.basicConfig(
        level=log_level.upper(),
        format="[%(asctime)s][%(threadName)s][%(levelname)s][%(module)s][%(funcName)s][Ln %(lineno)s] - %(message)s",
    )

    if experimental:
        url = adapters.gdp1h.GDP_DATA_URL_EXPERIMENTAL

    if drifter_ids:
        selected_drifters = [int(did) for did in drifter_ids.split(",")]
    else:
        selected_drifters = None

    if load_existing and os.path.exists(path):
        filelist = [os.path.join(path, f) for f in os.listdir(path)]
        downloaded_drifter_ids = [int(f.split("_")[-1][:-3]) for f in filelist]
        if len(downloaded_drifter_ids) > 0:
            selected_drifters = downloaded_drifter_ids
        else:
            _logger.warning("Cannot load existing files as there are none")
    else:
        _logger.warning(
            "Cannot load existing files as the path provided has not been created"
        )

    labels = map(
        lambda kv: f"({kv[0]}: {kv[1]})",
        [
            ("drifter_ids", drifter_ids),
            ("sample", sample),
            ("url", url),
            ("path", path),
            ("experimental", experimental),
            ("aggregate", aggregate),
            ("load_existing", load_existing),
            ("log_level", log_level),
        ],
    )

    cli_inputs = ", ".join(labels)
    _logger.debug(f"Inputs: {cli_inputs}")
    _logger.debug(f"selected the following drifters: {selected_drifters}")

    ra = adapters.gdp1h.to_raggedarray(selected_drifters, sample, url, path)

    if aggregate:
        aggregate_path = f"{path}/aggregate"
        os.makedirs(aggregate_path, exist_ok=True)

        aggregate_file_path = f"{aggregate_path}/gdp1h.nc"
        _logger.info(
            f"Generating and storing aggregated gdp1h data to: ({aggregate_file_path})"
        )
        ra.to_netcdf(aggregate_file_path)


if __name__ == "__main__":
    cli()

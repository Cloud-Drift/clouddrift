import concurrent.futures
import logging
import os
from datetime import datetime
from io import BufferedIOBase
from typing import Callable, Sequence

import requests
from requests import Response
from tenacity import (
    RetryCallState,
    WrappedFn,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

_DISABLE_SHOW_PROGRESS = False  # purely to de-noise our test suite output, should never be used/configured outside of that.


def _before_call(rcs: RetryCallState):
    if rcs.attempt_number > 1:
        src = rcs.args[0]
        dst = "io-buffer" if isinstance(rcs.args[1], BufferedIOBase) else rcs.args[1]
        _logger.warn(
            f"retrying download request for (dst, src): {(src, dst)}, attempt: {rcs.attempt_number}"
        )


_CHUNK_SIZE = 1024
_logger = logging.getLogger(__name__)
_standard_retry_protocol: Callable[[WrappedFn], WrappedFn] = retry(
    retry=retry_if_exception(
        lambda ex: isinstance(ex, (requests.Timeout, requests.HTTPError))
    ),
    wait=wait_exponential_jitter(
        initial=1.25
    ),  # ~ 20-25 minutes total time before completely failing
    stop=stop_after_attempt(10),
    before=_before_call,
)


def download_with_progress(
    download_map: Sequence[tuple[str, BufferedIOBase | str, float | None]],
    show_list_progress: bool | None = None,
    desc: str = "Downloading files",
    custom_retry_protocol: Callable[[WrappedFn], WrappedFn] | None = None,
):
    if show_list_progress is None:
        show_list_progress = len(download_map) > 20
    if custom_retry_protocol is None:
        retry_protocol = _standard_retry_protocol
    else:
        retry_protocol = custom_retry_protocol  # type: ignore

    executor = concurrent.futures.ThreadPoolExecutor()
    futures: dict[concurrent.futures.Future, tuple[str, BufferedIOBase | str]] = dict()
    bar = None

    for src, dst, exp_size in download_map:
        futures[
            executor.submit(
                retry_protocol(_download_with_progress),
                src,
                dst,
                exp_size or 0,
                not show_list_progress,
            )
        ] = (src, dst)
    try:
        if show_list_progress:
            bar = tqdm(
                desc=desc,
                total=len(futures),
                unit="Files",
                disable=_DISABLE_SHOW_PROGRESS,
            )

        for fut in concurrent.futures.as_completed(futures):
            (src, dst) = futures[fut]
            ex = fut.exception(0)
            if ex is None:
                _logger.debug(f"Finished download job: ({src}, {dst})")
                if bar is not None:
                    bar.update(1)
            else:
                raise ex
    except Exception as e:
        _logger.error(
            f"Got the following exception: {str(e)}, cancelling all other jobs and cleaning up \
              any created resources."
        )
        for x in futures.keys():
            (src, dst) = futures[x]
            if isinstance(dst, (str,)) and os.path.exists(dst) and not x.done():
                os.remove(dst)
            if not x.done():
                x.cancel()
        raise e
    finally:
        executor.shutdown(True)
        if bar is not None:
            bar.close()


def _download_with_progress(
    url: str,
    output: BufferedIOBase | str,
    expected_size: float,
    show_progress: bool,
):
    if isinstance(output, str) and os.path.exists(output):
        _logger.debug(f"File exists {output} checking for updates...")
        local_last_modified = os.path.getmtime(output)

        # Get last modified time of the remote file
        with requests.head(url, timeout=5) as res:
            if "Last-Modified" in res.headers:
                remote_last_modified = datetime.strptime(
                    res.headers.get("Last-Modified", ""),
                    "%a, %d %b %Y %H:%M:%S %Z",
                )

                # compare with local modified time
                if local_last_modified >= remote_last_modified.timestamp():
                    _logger.debug(f"File: {output} is up to date; skip download.")
                    return
            else:
                _logger.warning(
                    "Cannot determine if the file has been updated on the remote source. "
                    + "'Last-Modified' header not present in server response."
                )
    _logger.debug(f"Downloading from {url} to {output}...")

    force_close = False
    response: Response | None = None
    buffer: BufferedIOBase | None = None
    bar = None

    try:
        response = requests.get(url, timeout=5, stream=True)

        if isinstance(output, str):
            buffer = open(output, "wb")
        else:
            buffer = output

        if show_progress:
            bar = tqdm(
                desc=url,
                total=float(response.headers.get("Content-Length", expected_size)),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                nrows=2,
                disable=_DISABLE_SHOW_PROGRESS,
            )

        for chunk in response.iter_content(_CHUNK_SIZE):
            if not chunk:
                break
            buffer.write(chunk)
            if bar is not None:
                bar.update(len(chunk))
    except Exception as e:
        force_close = True
        error_msg = f"Error downloading data file: {url} to: {output}, error: {e}"
        _logger.debug(error_msg)
        raise e
    finally:
        if response is not None:
            response.close()
        if buffer is not None and (
            not isinstance(output, BufferedIOBase) or force_close
        ):
            _logger.debug(f"closing buffer {buffer}")
            buffer.close()
        if bar is not None:
            bar.close()


__all__ = ["download_with_progress"]

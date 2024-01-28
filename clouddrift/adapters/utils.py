import concurrent.futures
import datetime
import os
import warnings
from io import BufferedIOBase
from typing import Callable, List, NamedTuple, Union

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

_CHUNK_SIZE = 1024


class _DownloadRequest(NamedTuple):
    src: str
    dst: Union[BufferedIOBase, str]
    exp_size: Union[float, None]


def download_with_progress(
    download_map: List[_DownloadRequest], prewrite_func=lambda x: x
):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = dict()
        for req in download_map:
            if len(req) > 2:
                src, dst, exp_size = req[0], req[1], req[2]
            elif len(req) <= 2:
                src, dst, exp_size = req[0], req[1], None

            futures[
                executor.submit(
                    _download_with_progress, src, dst, exp_size, prewrite_func
                )
            ] = (src, dst)

        for fut in concurrent.futures.as_completed(futures):
            (src, dst) = futures[fut]
            ex = fut.exception(0)
            if ex is not None:
                print(f"there was an issue downloading {src} to {dst}, exception details: {ex}")
            print(f"Finished downloading: {src}")


@retry(
    retry=retry_if_exception(
        lambda ex: isinstance(ex, [requests.Timeout, requests.HTTPError])
    ),
    wait=wait_exponential_jitter(initial=0.25),
    stop=stop_after_attempt(10),
)
def _download_with_progress(
    url: str,
    output: Union[BufferedIOBase, str],
    expected_size: Union[float, None],
    prewrite_func: Callable[[bytes], Union[str, bytes]],
):
    if isinstance(output, str) and os.path.exists(output):
        print(f"File exists {output} checking for updates...")
        local_last_modified = os.path.getmtime(output)

        # Get last modified time of the remote file
        with requests.head(url, timeout=5) as response:
            if "Last-Modified" in response.headers:
                remote_last_modified = datetime.strptime(
                    response.headers.get("Last-Modified"), "%a, %d %b %Y %H:%M:%S %Z"
                )

                # compare with local modified time
                if local_last_modified >= remote_last_modified.timestamp():
                    warnings.warn(
                        f"{output} already exists and is up to date; skip download."
                    )
                    return False
            else:
                warnings.warn(
                    "Cannot determine the file has been updated on the remote source. \
                              'Last-Modified' header not present."
                )
    print(f"Downloading from {url} to {output}...")

    force_close = False
    try:
        response = requests.get(url, timeout=5, stream=True)
        if isinstance(output, str):
            buffer = open(output, "wb")
        else:
            buffer = output
        bar = tqdm(
            desc=url,
            total=float(response.headers.get("Content-Length", expected_size)),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

        for chunk in response.iter_content(_CHUNK_SIZE):
            if not chunk:
                break
            buffer.write(prewrite_func(chunk))
            bar.update(len(chunk))
    except Exception as e:
        import traceback as tb

        force_close = True
        error_msg = f"Error downloading data file: {url} to: {output}, error: {e}"
        print(error_msg)
        tb.print_exc()
        raise RuntimeError(error_msg)
    finally:
        if response is not None:
            response.close()
        if buffer is not None and not isinstance(buffer, BufferedIOBase) or force_close:
            print(f"closing buffer {buffer}")
            buffer.close()
        if bar is not None:
            bar.close()
    return True

from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock, _patch


class MultiPatcher:
    _patches: Sequence[_patch[Any]]

    def __init__(self, patches: Sequence[_patch[Any]]):
        self._patches = patches

    def __enter__(self) -> Sequence[Mock]:
        return [p.start() for p in self._patches]

    def __exit__(self, *_):
        for p in self._patches:
            p.stop()

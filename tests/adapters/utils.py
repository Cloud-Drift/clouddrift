from collections.abc import Sequence
from unittest.mock import Mock, _patch


class MultiPatcher:
    _patches: Sequence[_patch]

    def __init__(self, patches: Sequence[_patch]):
        self._patches = patches

    def __enter__(self) -> Sequence[Mock]:
        return [p.start() for p in self._patches]

    def __exit__(self, *_):
        for p in self._patches:
            p.stop()

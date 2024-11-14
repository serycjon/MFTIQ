import logging
import warnings
from pathlib import Path
from typing import Any
logger = logging.getLogger(__name__)


class ConfigMissingWarning(UserWarning):
    pass


class Config():
    def __init__(self):
        self._missing_reported = set()

    def __getattr__(self, name) -> Any:
        # gets called on attempt to access not-existent attribute
        # C.foo.bar.baz does not fail if foo, foo.bar, or foo.bar.baz is not in config
        if name not in self._missing_reported:
            # We manually keep track of which names were already reported as missing, because of a bug in Python
            # https://github.com/python/cpython/issues/73858
            # https://github.com/python/cpython/pull/8232
            # https://github.com/python/cpython/blob/4765e1fa292007f8ddc59f33454b747312506a7a/Lib/warnings.py#L700
            warnings.warn(f"Value '{name}' not specified in config.", category=ConfigMissingWarning)
            self._missing_reported.add(name)

        return Config()

    def __bool__(self):
        # the truth value should be false to accomodate for inexistent config values
        # e.g. C.foo.bar.baz == False if foo, foo.bar, or foo.bar.baz is not in config
        return False

    def merge(self, other, update_dicts=False):
        other_dict = other.__dict__
        other_keys = other_dict.keys()
        our_keys = self.__dict__.keys()
        for key in other_keys:
            if key in our_keys:
                if update_dicts and isinstance(key, dict) and isinstance(getattr(self, key), dict):
                    getattr(self, key).update(other_dict[key])
                else:
                    logger.debug(f"Rewriting key [{key}] in config. ({getattr(self, key)} -> {getattr(other, key)})")
                    setattr(self, key, other_dict[key])
            else:
                setattr(self, key, other_dict[key])

    def __repr__(self):
        return repr(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


def load_config(path):
    """ https://stackoverflow.com/a/67692 """
    assert Path(path).exists(), f"config {path} does not exist!"
    import importlib.util
    spec = importlib.util.spec_from_file_location("tracker_config", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.get_config()

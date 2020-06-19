from .base import Ranker
from .trivial import Trivial


def __getattr__(name: str) -> Ranker:
    import importlib

    s = ""
    for i, c in enumerate(name):
        if c.isupper():
            if s: 
                s += "_"
            s += c.lower()
        else:
            s += c

    mod = importlib.import_module("onir.rankers.%s" % s)
    return getattr(mod, name)
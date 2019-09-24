# Functions derived from ocropy sl:
# https://github.com/tmbdev/ocropy/blob/d3e5cc60b64d070b60d606a16baeda6b436cc23b/ocrolib/sl.py
#
# ocropus itself can in some instances not not imported without executing it.
# (Missing if __name__ == "__main__.py") And ocropus is python2
# This also allows to use the function in python3 and and remove unneeded
# dependencies like matplotlib.
import numpy as np


def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop-s[0].start


def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop-s[1].start


def area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return np.prod([max(x.stop-x.start, 0) for x in a[:2]])


def width(s):
    return s[1].stop-s[1].start


def height(s):
    return s[0].stop-s[0].start

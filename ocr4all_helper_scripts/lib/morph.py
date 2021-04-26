# Functions derived from ocropy morph:
# https://github.com/tmbdev/ocropy/blob/6da681baa26718e7698368f07c7d72e0d6eadccf/ocrolib/morph.py
#
# ocropus itself can in some instances not not imported without executing it.
# (Missing if __name__ == "__main__.py") And ocropus is python2
# This also allows to use the function in python3 and and remove unneeded
# dependencies like matplotlib.

from scipy.ndimage import filters, measurements, morphology
from numpy import array, zeros, argsort, amax, amin, unique


def select_regions(binary, f, min=0, nbest=100000):
    """Given a scoring function f over slice tuples (as returned by
    find_objects), keeps at most nbest regions whose scores is higher
    than min."""
    labels, n = label(binary)
    objects = find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects)+1, 'i')
    if nbest > 0:
        for i in best[-nbest:]:
            if scores[i] <= min:
                continue
            keep[i+1] = 1
    return keep[labels]


def r_dilation(image, size, origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return filters.maximum_filter(image, size, origin=origin)


def r_erosion(image, size, origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return filters.minimum_filter(image, size, origin=origin)


def rb_dilation(image, size, origin=0):
    """Binary dilation using linear filters."""
    output = zeros(image.shape, 'f')
    filters.uniform_filter(image, size, output=output, origin=origin, mode='constant', cval=0)
    return array(output > 0, 'i')


def rb_erosion(image, size, origin=0):
    """Binary erosion using linear filters."""
    output = zeros(image.shape, 'f')
    filters.uniform_filter(image, size, output=output, origin=origin, mode='constant', cval=1)
    return array(output == 1, 'i')


def rb_opening(image, size, origin=0):
    """Binary opening using linear filters."""
    image = rb_erosion(image, size, origin=origin)
    return rb_dilation(image, size, origin=origin)


def find_objects(image, **kw):
    """Redefine the scipy.ndimage.measurements.find_objects function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try:
        return measurements.find_objects(image, **kw)
    except:
        pass
    types = ["int32", "uint32", "int64", "uint64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.find_objects(array(image, dtype=t), **kw)
        except:
            pass
    # let it raise the same exception as before
    return measurements.find_objects(image, **kw)


def label(image, **kw):
    """Redefine the scipy.ndimage.measurements.label function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try:
        return measurements.label(image, **kw)
    except:
        pass
    types = ["int32", "uint32", "int64", "uint64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.label(array(image, dtype=t), **kw)
        except:
            pass
    # let it raise the same exception as before
    return measurements.label(image, **kw)


def propagate_labels(image, labels, conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label.
    Assign the value `conflict` to any labels that have a conflict."""
    rlabels, _ = label(image)
    cors = correspondences(rlabels, labels)
    outputs = zeros(amax(rlabels)+1, 'i')
    oops = -(1 << 30)
    for o, i in cors.T:
        if outputs[o] != 0:
            outputs[o] = oops
        else:
            outputs[o] = i
    outputs[outputs == oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]


def correspondences(labels1, labels2):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images."""
    q = 100000
    assert amin(labels1) >= 0 and amin(labels2) >= 0
    assert amax(labels2) < q
    combo = labels1*q+labels2
    result = unique(combo)
    result = array([result//q, result % q])
    return result


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = morphology.distance_transform_edt(labels == 0, return_distances=1, return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances < maxdist)
    return spread

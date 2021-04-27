# Functions derived from ocropy ocropus-gpageseg and psegutils:
# https://github.com/tmbdev/ocropy/blob/6da681baa26718e7698368f07c7d72e0d6eadccf/ocropus-gpageseg
# https://github.com/tmbdev/ocropy/blob/6da681baa26718e7698368f07c7d72e0d6eadccf/ocrolib/psegutils.py
#
# ocropus itself can in some instances not not imported without executing it.
# (Missing if __name__ == "__main__.py") And ocropus is python2
# This also allows to use the function in python3 and and remove unneeded
# dependencies like matplotlib.
import numpy as np
from scipy.ndimage.filters import gaussian_filter, uniform_filter, maximum_filter
from ocr4all_helper_scripts.lib import morph, sl


def compute_colseps(binary, scale, max_blackseps, widen_blackseps, max_whiteseps, minheight_whiteseps):
    """Computes column separators either from vertical black lines or whitespace.
    """
    colseps = compute_colseps_conv(binary, scale,
                                   minheight_whiteseps=minheight_whiteseps,
                                   max_whiteseps=max_whiteseps)
    if max_blackseps > 0:
        seps = compute_separators_morph(binary, scale,
                                        max_blackseps=max_blackseps,
                                        widen_blackseps=widen_blackseps)
        colseps = np.maximum(colseps, seps)
        binary = np.minimum(binary, 1-seps)
    return colseps, binary


def compute_boxmap(binary, scale, threshold=(.5, 4), dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl.area)
    boxmap = np.zeros(binary.shape, dtype)
    for o in bysize:
        if sl.area(o)**.5 < threshold[0]*scale:
            continue
        if sl.area(o)**.5 > threshold[1]*scale:
            continue
        boxmap[o] = 1
    return boxmap


def compute_colseps_conv(binary, scale=1.0, minheight_whiteseps=10, max_whiteseps=3):
    """Find column separators by convolution and thresholding.
    """
    h, w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary, (scale, scale*0.5))
    smoothed = uniform_filter(smoothed, (5.0*scale, 1))
    thresh = (smoothed < np.amax(smoothed)*0.1)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary, (scale, scale*0.5), order=(0, 1))
    grad = uniform_filter(grad, (10.0*scale, 1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad > 0.5*np.amax(grad))
    # combine edges and whitespace
    seps = np.minimum(thresh, maximum_filter(grad, (int(scale), int(5*scale))))
    seps = maximum_filter(seps, (int(2*scale), 1))
    # select only the biggest column separators
    seps = morph.select_regions(seps, sl.dim0, min=minheight_whiteseps*scale, nbest=max_whiteseps)
    return seps


def compute_separators_morph(binary, scale, max_blackseps=0, widen_blackseps=10):
    """Finds vertical black lines corresponding to column separators.
    """
    d0 = int(max(5, scale/4))
    d1 = int(max(5, scale))+widen_blackseps
    thick = morph.r_dilation(binary, (d0, d1))
    vert = morph.rb_opening(thick, (10*scale, 1))
    vert = morph.r_erosion(vert, (d0//2, widen_blackseps))
    vert = morph.select_regions(vert, sl.dim1, min=3, nbest=2*max_blackseps)
    vert = morph.select_regions(vert, sl.dim0, min=20*scale, nbest=max_blackseps)
    return vert


def estimate_scale(binary):
    """Estimate the scale factor of a binary image
    """
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl.area)
    scalemap = np.zeros(binary.shape)
    for o in bysize:
        if np.amax(scalemap[o]) > 0:
            continue
        scalemap[o] = sl.area(o)**0.5
    scale = np.median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale


def binary_objects(binary):
    labels, n = morph.label(binary)
    objects = morph.find_objects(labels)
    return objects


def reading_order(lines):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = np.zeros((len(lines), len(lines)), 'B')

    def x_overlaps(u, v):
        return u[1].start < v[1].stop and u[1].stop > v[1].start

    def above(u, v):
        return u[0].start < v[0].start

    def left_of(u, v):
        return u[1].stop < v[1].start

    def separates(w, u, v):
        if w[0].stop < min(u[0].start, v[0].start):
            return 0
        if w[0].start > max(u[0].stop, v[0].stop):
            return 0
        if w[1].start < u[1].stop and w[1].stop > v[1].start:
            return 1

    for i, u in enumerate(lines):
        for j, v in enumerate(lines):
            if x_overlaps(u, v):
                if above(u, v):
                    order[i, j] = 1
            else:
                if [w for w in lines if separates(w,u,v)]==[]:
                    if left_of(u, v):
                        order[i, j] = 1
    return order


def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = np.zeros(n)
    L = []

    def visit(k):
        if visited[k]:
            return
        visited[k] = 1
        for l in find(order[:, k]):
            visit(l)
        L.append(k)

    for k in range(n):
        visit(k)

    return L  # [::-1]


def find(condition):
    """Return the indices where ravel(condition) is true
    """
    res, = np.nonzero(np.ravel(condition))
    return res


def compute_line_seeds(binary, bottom, top, colseps, scale, threshold=0.2, vscale=1.0):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = threshold
    vrange = int(vscale*scale)
    bmarked = maximum_filter(bottom == maximum_filter(bottom, (vrange, 0)), (2, 2))
    bmarked = bmarked*(bottom > t*np.amax(bottom)*t)*(1-colseps)
    tmarked = maximum_filter(top == maximum_filter(top, (vrange, 0)), (2, 2))
    tmarked = tmarked*(top > t*np.amax(top)*t/2)*(1-colseps)
    tmarked = maximum_filter(tmarked, (1, 20))
    seeds = np.zeros(binary.shape, 'i')
    delta = max(3, int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y, 1) for y in find(bmarked[:, x])]+[(y, 0) for y in find(tmarked[:, x])])[::-1]
        transitions += [(0, 0)]
        for l in range(len(transitions)-1):
            y0, s0 = transitions[l]
            if s0 == 0:
                continue
            seeds[y0-delta:y0, x] = 1
            y1, s1 = transitions[l+1]
            if s1 == 0 and (y0-y1) < 5*scale:
                seeds[y1:y0, x] = 1
    seeds = maximum_filter(seeds, (1, int(1+scale)))
    seeds = seeds*(1-colseps)
    seeds, _ = morph.label(seeds)
    return seeds


def remove_hlines(binary, scale, maxsize=10):
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i, b in enumerate(objects):
        if sl.width(b) > maxsize*scale:
            labels[b][labels[b] == i+1] = 0
    return np.array(labels != 0, 'B')

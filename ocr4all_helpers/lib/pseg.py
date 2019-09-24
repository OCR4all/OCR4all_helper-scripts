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
import lib.morph as morph
import lib.sl as sl


# Computes column separators either from vertical black lines or whitespace.
def compute_colseps(binary, scale, maxseps, blackseps):
    colseps = compute_colseps_conv(binary, scale)
    if blackseps and maxseps == 0:
        # simulate old behaviour of blackseps when the default value
        # for maxseps was 2, but only when the maxseps-value is still zero
        # and not set manually to a non-zero value
        maxseps = 2
    if maxseps > 0:
        seps = compute_separators_morph(binary, scale)
        colseps = np.maximum(colseps, seps)
        binary = np.minimum(binary, 1-seps)
    #binary, colseps = apply_mask(binary, colseps) # ignore apply mask as kraken does
    return colseps, binary


# Find column separators by convolution and thresholding.
def compute_colseps_conv(binary, scale=1.0, csminheight=10, maxcolseps=3):
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
    seps = morph.select_regions(seps, sl.dim0, min=csminheight*scale, nbest=maxcolseps)
    return seps


# Finds vertical black lines corresponding to column separators.
def compute_separators_morph(binary, scale, maxseps=0, sepwiden=10):
    d0 = int(max(5, scale/4))
    d1 = int(max(5, scale))+sepwiden
    thick = morph.r_dilation(binary, (d0, d1))
    vert = morph.rb_opening(thick, (10*scale, 1))
    vert = morph.r_erosion(vert, (d0//2, sepwiden))
    vert = morph.select_regions(vert, sl.dim1, min=3, nbest=2*maxseps)
    vert = morph.select_regions(vert, sl.dim0, min=20*scale, nbest=maxseps)
    return vert
# Functions derived from ocropy nlbin:
# https://github.com/tmbdev/ocropy/blob/8f354ad4facc19eb5c2d5099dc47df9343fd3602/ocropus-nlbin
#
# ocropus itself can in some instances not not imported without executing it.
# (Missing if __name__ == "__main__.py")
# This also allows to use the function in python3 and and remove unneeded
# dependencies like matplotlib.

import numpy as np
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
from PIL import Image


def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    """Estimate skew angle of a scanned line and rotate accordingly. Ported method from ocropy
    """
    d0, d1 = flat.size
    o0, o1 = int(bignore*d0), int(bignore*d1)  # border ignore
    flat = np.amax(flat)-flat
    flat -= np.amin(flat)
    est = Image.fromarray(flat[o0:d0-o0, o1:d1-o1])
    ma = maxskew
    ms = int(2*maxskew*skewsteps)
    return estimate_skew_angle(est, np.linspace(-ma, ma, ms+1))


def estimate_skew_angle(image, angles):
    """Estimate the angle of a skew of a scanned line. (Ported method from ocropy)
    """
    try:
        estimates = []
        for a in angles:
            rotated = image.rotate(a, expand=True)
            v = np.mean(np.array(rotated), axis=1)
            v = np.var(v)
            estimates.append((v, a))
        _, a = max(estimates)
        return a
    except IndexError:
        # Image is empty-ish
        return 0
        

def adaptive_binarize(image, threshold=0.5, zoom=0.5, perc=80, range=20):
    # check whether the image is already effectively binarized
    extreme = (np.sum(image < 0.05)+np.sum(image > 0.95))*1.0/np.prod(image.shape)
    if extreme > 0.95:
        flat = image.astype(np.float64)
    else:
        # if not, we need to flatten it by estimating the local whitelevel
        flat = estimate_local_whitelevel(image, zoom, perc, range).astype(np.float64)
        del image

    # estimate low and high thresholds
    lo, hi = estimate_thresholds(flat)
    # rescale the image to get the gray scale image
    flat -= lo
    flat /= (hi-lo)
    flat = np.clip(flat, 0, 1)
    binary = 1*(flat > threshold)
    del flat
    return binary


def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90):
    """ estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    """
    d0, d1 = flat.shape
    o0, o1 = int(bignore*d0), int(bignore*d1)
    est = flat[o0:d0-o0, o1:d1-o1]
    if escale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est-filters.gaussian_filter(est, e*20.0)
        v = filters.gaussian_filter(v**2, e*20.0)**0.5
        v = (v > 0.3*np.amax(v))
        v = morphology.binary_dilation(v, structure=np.ones((int(e*50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e*50))))
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    del est
    return lo, hi


def estimate_local_whitelevel(image, zoom=0.5, perc=80, range=20):
    """
    Flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    """
    m = interpolation.zoom(image, zoom)
    m = filters.percentile_filter(m, perc, size=(range, 2))
    m = filters.percentile_filter(m, perc, size=(2, range))
    m = interpolation.zoom(m, 1.0/zoom)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h]-m[:w, :h]+1, 0, 1)
    del m
    return flat

# -*- coding: utf-8 -*-
# Line segmentation script for images with PAGE xml.
# Derived from the `import_from_larex.py` script, parts of kraken and ocropy,
# with additional tweaks e.g. pre rotation of text regions
#
# nashi project:
#   https://github.com/andbue/nashi 
# ocropy:
#   https://github.com/tmbdev/ocropy/
# kraken:
#   https://github.com/mittagessen/kraken

import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import math

from lxml import etree
from PIL import Image, ImageDraw
from ocr4all_helpers.lib import imgmanipulate, morph, sl, pseg, nlbin

from multiprocessing.pool import ThreadPool
import json

import argparse

import os
import sys

# Add printing for every thread
from threading import Lock
s_print_lock = Lock()
def s_print(*a, **b):
    with s_print_lock:
        print(*a, **b)


def s_print_error(*objs):
    s_print("ERROR: ", *objs, file=sys.stderr)


class record(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


# Given a line segmentation map, computes a list
# of tuples consisting of 2D slices and masked images.
#
# Implementation derived from ocropy with changes to allow extracting
# the line coords/polygons
def compute_lines(segmentation, smear_strength, scale, growth, max_iterations, filter_strength=1.0):
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl.dim1(o) < 2*scale*filter_strength or sl.dim0(o) < scale*filter_strength:
            continue
        mask = (segmentation[o] == i+1)
        if np.amax(mask) == 0:
            continue

        result = record()
        result.label = i+1
        result.bounds = o
        polygon = []
        if ((segmentation[o] != 0) == (segmentation[o] != i+1)).any():
            ppoints = approximate_smear_polygon(mask, smear_strength, growth, max_iterations)
            ppoints = ppoints[1:] if ppoints else []
            polygon = [(o[1].start+x, o[0].start+y) for x, y in ppoints]
        if not polygon:
            polygon = [(o[1].start, o[0].start), (o[1].stop,  o[0].start),
                       (o[1].stop,  o[0].stop),  (o[1].start, o[0].stop)]
        result.polygon = polygon
        result.mask = mask
        lines.append(result)

    return lines


def compute_gradmaps(binary, scale, vscale=1.0, hscale=1.0, usegauss=False):
    # use gradient filtering to find baselines
    boxmap = pseg.compute_boxmap(binary, scale)
    cleaned = boxmap*binary
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned, (vscale*0.3*scale,
                                            hscale*6*scale), order=(1, 0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned, (max(4, vscale*0.3*scale),
                                            hscale*scale), order=(1, 0))
        grad = uniform_filter(grad, (vscale,hscale*6*scale))

    def norm_max(a):
        return a/np.amax(a)

    bottom = norm_max((grad<0)*(-grad))
    top = norm_max((grad>0)*grad)
    return bottom, top, boxmap


def boundary(contour):
    Xmin = np.min(contour[:, 0])
    Xmax = np.max(contour[:, 0])
    Ymin = np.min(contour[:, 1])
    Ymax = np.max(contour[:, 1])

    return [Xmin, Xmax, Ymin, Ymax]


# Approximate a single polygon around high pixels in a mask, via smearing
def approximate_smear_polygon(line_mask, smear_strength=(1, 2), growth=(1.1, 1.1), max_iterations=1000):
    padding = 1
    work_image = np.pad(np.copy(line_mask), pad_width=padding, mode='constant', constant_values=False)

    contours = find_contours(work_image, 0.5, fully_connected="low")

    if len(contours) > 0:
        iteration = 1
        while len(contours) > 1:
            # Get bounds with dimensions
            bounds = [boundary(contour) for contour in contours]
            widths = [b[1]-b[0] for b in bounds]
            heights = [b[3]-b[2] for b in bounds]

            # Calculate x and y median distances (or at least 1)
            width_median = sorted(widths)[int(len(widths) / 2)]
            height_median = sorted(heights)[int(len(heights) / 2)]

            # Calculate x and y smear distance
            smear_distance_x = math.ceil(width_median*smear_strength[0] * (iteration*growth[0]))
            smear_distance_y = math.ceil(height_median*smear_strength[1] * (iteration*growth[1]))

            # Smear image in x and y direction
            height, width = work_image.shape
            gaps_current_x = [float('Inf')]*height
            for x in range(width):
                gap_current_y = float('Inf')
                for y in range(height):
                    if work_image[y, x]:
                        # Entered Contour
                        gap_current_x = gaps_current_x[y]

                        if gap_current_y < smear_distance_y and gap_current_y > 0:
                            # Draw over
                            work_image[y-gap_current_y:y, x] = True
                        
                        if gap_current_x < smear_distance_x and gap_current_x > 0:
                            #Draw over
                            work_image[y, x-gap_current_x:x] = True

                        gap_current_y = 0
                        gaps_current_x[y] = 0
                    else:
                        # Entered/Still in Gap
                        gap_current_y += 1
                        gaps_current_x[y] += 1
            # Find contours of current smear
            contours = find_contours(work_image, 0.5, fully_connected="low")

            # Failsave if contours can't be smeared together after x iterations
            # Draw lines between the extreme points of each contour in order
            if iteration >= max_iterations and len(contours) > 1:
                s_print(("Start fail save, since precise line generation took"
                         " too many iterations ({}).").format(iteration))
                extreme_points = []
                for contour in contours:
                    sorted_x = sorted(contour, key=lambda c: c[0])
                    sorted_y = sorted(contour, key=lambda c: c[1])
                    extreme_points.append((tuple(sorted_x[0]), tuple(sorted_y[0]),
                                           tuple(sorted_x[-1]), tuple(sorted_y[-1])))
                
                sorted_extreme = sorted(extreme_points, key=lambda e: e)
                for c1, c2 in zip(sorted_extreme, sorted_extreme[1:]):
                    for p1 in c1:
                        nearest = None
                        nearest_dist = math.inf
                        for p2 in c2:
                            distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                            if distance < nearest_dist:
                                nearest = p2
                                nearest_dist = distance
                        if nearest:
                            # Draw line between nearest points
                            yy, xx, _ = line_aa(int(p1[0]), int(nearest[0]), int(p2[1]), int(nearest[1]))
                            # Remove border points
                            line_points = [(x, y) for x, y in zip(xx, yy) if 0 < x < width and 0 < y < height]
                            xx_filtered, yy_filtered = zip(*line_points)
                            # Paint
                            work_image[yy_filtered, xx_filtered] = True
                contours = find_contours(work_image, 0.5, fully_connected="low")

            iteration += 1
        return [(p[1]-padding, p[0]-padding) for p in approximate_polygon(contours[0], 0.1)]
    return []
    

def segment(im, scale=None,
            max_blackseps=0, widen_blackseps=10,
            max_whiteseps=3, minheight_whiteseps=10,
            smear_strength=(1, 2), growth=(1.1, 1.1), orientation=0,
            fail_save_iterations=1000, vscale=1.0, hscale=1.0,
            minscale=12.0, maxlines=300,
            threshold=0.2, usegauss=False):
    """
    Segments a page into text lines.
    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.
    """

    colors = im.getcolors(2)
    if im.mode != '1' and not (colors is not None and len(colors) == 2):
        raise ValueError('Image is not bi-level')

    # rotate input image for vertical lines
    im_rotated = im.rotate(-1*orientation, expand=True)

    a = np.array(im_rotated.convert('L')) if im_rotated.mode == '1' else np.array(im_rotated)

    binary = np.array(a > 0.5*(np.amin(a) + np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = pseg.estimate_scale(binary)
    s_print(scale, binary.shape, np.min(binary), np.max(binary))
    if scale < minscale:
        s_print_error("scale ({}) less than --minscale; skipping".format(scale))
        return

    binary = pseg.remove_hlines(binary, scale)
    # emptyish images will cause exceptions here.
    try:
        colseps, binary = pseg.compute_colseps(binary,
                                               scale,
                                               max_blackseps,
                                               widen_blackseps,
                                               max_whiteseps,
                                               minheight_whiteseps)
    except ValueError:
        return []

    bottom, top, boxmap = compute_gradmaps(binary, scale, vscale, hscale, usegauss)
    seeds = pseg.compute_line_seeds(binary, bottom, top, colseps, scale, threshold=threshold)
    llabels1 = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = np.where(llabels1 > 0, llabels1, spread*binary)
    segmentation = llabels*binary

    if np.amax(segmentation) > maxlines:
        s_print_error("too many lines {}".format(np.amax(segmentation)))
        return

    lines_and_polygons = compute_lines(segmentation,
                                       smear_strength,
                                       scale,
                                       growth,
                                       fail_save_iterations)

    # Translate each point back to original
    deltaX = (im_rotated.width - im.width) / 2
    deltaY = (im_rotated.height - im.height) / 2
    centerX = im_rotated.width / 2
    centerY = im_rotated.height / 2

    def translate_back(point):
        # rotate point around center
        orient_rad = -1*orientation * (math.pi / 180)
        rotatedX = ((point[0]-centerX) * math.cos(orient_rad)
                    - (point[1]-centerY) * math.sin(orient_rad)
                    + centerX)
        rotatedY = ((point[0]-centerX) * math.sin(orient_rad)
                    + (point[1]-centerY) * math.cos(orient_rad)
                    + centerY)
        # move point
        return (int(rotatedX-deltaX), int(rotatedY-deltaY))

    return [[translate_back(p) for p in record.polygon] for record in lines_and_polygons]


def pagexmllineseg(xmlfile, imgpath,
                   scale=None,
                   vscale=1.0,
                   hscale=1.0,
                   max_blackseps=0,
                   widen_blackseps=10,
                   max_whiteseps=-1,
                   minheight_whiteseps=10,
                   minscale=12,
                   maxlines=300,
                   smear_strength=(1, 2),
                   growth=(1.1, 1.1),
                   fail_save_iterations=100,
                   maxskew=2.0,
                   skewsteps=8,
                   usegauss=False,
                   remove_images=False):
    name = os.path.splitext(os.path.split(imgpath)[-1])[0]
    s_print("""Start process for '{}'
        |- Image: '{}'
        |- Annotations: '{}' """.format(name, imgpath, xmlfile))

    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    s_print("[{}] Retrieve TextRegions".format(name))

    # convert point notation from older pagexml versions
    for c in root.xpath("//ns:Coords[not(@points)]", namespaces=ns):
        cc = []
        for point in c.xpath("./ns:Point", namespaces=ns):
            cx = point.attrib["x"]
            cy = point.attrib["y"]
            c.remove(point)
            cc.append(cx+","+cy)
        c.attrib["points"] = " ".join(cc)

    coordmap = {}
    for r in root.xpath('//ns:TextRegion', namespaces=ns):
        rid = r.attrib["id"]
        coordmap[rid] = {"type": r.attrib["type"]}
        coordmap[rid]["coords"] = []
        for c in r.xpath("./ns:Coords", namespaces=ns) + r.xpath("./Coords"):
            coordmap[rid]["coordstring"] = c.attrib["points"]
            coordstrings = [x.split(",") for x in c.attrib["points"].split()]
            coordmap[rid]["coords"] += [[int(x[0]), int(x[1])]
                                        for x in coordstrings]
        if 'orientation' in r.attrib:
            coordmap[rid]["orientation"] = float(r.attrib["orientation"])

    s_print("[{}] Extract Textlines from TextRegions".format(name))
    im = Image.open(imgpath)

    if remove_images:
        # Draw white over ImageRegions
        white = {
            "1": 1, "L": 255, "P": 255,
            "RGB": (255, 255, 255), "RGBA": (255, 255, 255, 255),
            "CMYK": (0, 0, 0, 0), "YCbCr": (1, 0, 0),
            "Lab": (100, 0, 0), "HSV": (0, 0, 100)
        }[im.mode]
        draw = ImageDraw.Draw(im)
        for r in root.xpath('//ns:ImageRegion', namespaces=ns):
            for c in r.xpath("./ns:Coords", namespaces=ns) + r.xpath("./Coords"):
                coordstrings = [x.split(",") for x in c.attrib["points"].split()]
                poly = [(int(x[0]), int(x[1])) for x in coordstrings]
                draw.polygon(poly, fill=white)
        del draw

    for n, c in enumerate(sorted(coordmap)):
        coords = coordmap[c]['coords']

        if len(coords) < 3:
            continue
        cropped, [minX, minY, maxX, maxY] = imgmanipulate.cutout(im, coords)

        if 'orientation' in coordmap[c]:
            orientation = coordmap[c]['orientation']
        else:
            orientation = -1*nlbin.estimate_skew(cropped, 0, maxskew=maxskew,
                                                 skewsteps=skewsteps)
            s_print(("[{}] Skew estimate between +/-{} in {} steps."
                     " Estimated {}°").format(name, maxskew, skewsteps, orientation))

        if cropped is not None:
            colors = cropped.getcolors(2)
            if not (colors is not None and len(colors) == 2):
                cropped = Image.fromarray(nlbin.adaptive_binarize(np.array(cropped)))
            if coordmap[c]["type"] == "drop-capital":
                lines = [1]
            else:
                # if line in
                lines = segment(cropped, scale=scale,
                                max_blackseps=max_blackseps,
                                widen_blackseps=widen_blackseps,
                                max_whiteseps=max_whiteseps,
                                minheight_whiteseps=minheight_whiteseps,
                                smear_strength=smear_strength, growth=growth,
                                orientation=orientation,
                                fail_save_iterations=fail_save_iterations,
                                vscale=vscale, hscale=hscale,
                                minscale=minscale, maxlines=maxlines,
                                usegauss=usegauss)

        else:
            lines = []


        # Iterpret whole region as textline if no textline are found
        if not(lines) or len(lines) == 0:
            coordstrg = " ".join([str(x)+","+str(y) for x, y in coords])
            textregion = root.xpath('//ns:TextRegion[@id="'+c+'"]', namespaces=ns)[0]
            if orientation:
                textregion.set('orientation', str(orientation))
            linexml = etree.SubElement(textregion, "TextLine",
                                       attrib={"id": "{}_l{:03d}".format(c, n+1)})
            etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})
        else:
            for n, poly in enumerate(lines):
                if coordmap[c]["type"] == "drop-capital":
                    coordstrg = coordmap[c]["coordstring"]
                else:
                    coords = ((x+minX, y+minY) for x, y in poly)
                    coordstrg = " ".join([str(int(x))+","+str(int(y)) for x, y in coords])

                textregion = root.xpath('//ns:TextRegion[@id="'+c+'"]', namespaces=ns)[0]
                if orientation:
                    textregion.set('orientation', str(orientation))
                linexml = etree.SubElement(textregion, "TextLine",
                                           attrib={"id": "{}_l{:03d}".format(c, n+1)})
                etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})

    s_print("[{}] Generate new PAGE xml with textlines".format(name))
    xmlstring = etree.tounicode(root.getroottree()).replace(
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    no_lines_segm = int(root.xpath("count(//TextLine)"))
    return xmlstring, no_lines_segm


# Command line interface for the pagelineseg script
def cli():
    parser = argparse.ArgumentParser("""
    Line segmentation with regions read from a PAGE xml file
    """)
    # input
    g_in = parser.add_argument_group('input')
    g_in.add_argument('DATASET',
                      type=str,
                      help=('Path to the input dataset in json format with '
                            'a list of image path, pagexml path and optional'
                            ' output path. (Will overwrite pagexml if no '
                            'output path is given)')
                      )
    g_in.add_argument('--remove_images',
                      action='store_true',
                      help=('Remove ImageRegions from the image before '
                            'processing TextRegions for TextLines. Can be used'
                            ' if ImageRegions overlap with TextRegions'
                            'default: %(default)s')
                      )

    # limits
    g_limit = parser.add_argument_group('limit parameters')
    g_limit.add_argument('--minscale',
                         type=float,
                         default=12.0,
                         help='minimum scale permitted, default: %(default)s'
                         )
    g_limit.add_argument('--maxlines',
                         type=float,
                         default=300,
                         help='maximum # lines permitted, default: %(default)s'
                         )

    # line parameters
    g_line = parser.add_argument_group('line parameters')
    g_line.add_argument('--threshold',
                        type=float,
                        default=0.2,
                        help='baseline threshold, default: %(default)s'
                        )
    g_line.add_argument('--usegauss',
                        action='store_true',
                        help=('use gaussian instead of uniform, '
                              'default: %(default)s')
                        )

    # scale parameters
    g_scale = parser.add_argument_group('scale parameters')
    g_scale.add_argument('-s', '--scale',
                         type=float,
                         default=None,
                         help=('Scale of the input image used for the line'
                               'segmentation. Will be estimated if '
                               'not defined, 0 or smaller.')
                         )
    g_scale.add_argument('--hscale',
                         type=float,
                         default=1.0,
                         help=('Non-standard scaling of horizontal parameters.'
                               ' (default: %(default)s)')
                         )
    g_scale.add_argument('--vscale',
                         type=float,
                         default=1.0,
                         help=('non-standard scaling of vertical parameters. '
                               '(default: %(default)s)')
                         )
    g_scale.add_argument('--filter_strength',
                         type=float,
                         default=1.0,
                         help=('Strength individual characters are filtered out '
                               'when creating a textline, default: %(default)s')
                         )

    # region skew estimate
    g_skew = parser.add_argument_group('skew estimate parameters')
    g_skew.add_argument('-m', '--maxskew',
                        type=float,
                        default=2.0,
                        help=('Maximal estimated skew of an image.')
                        )
    g_skew.add_argument('--skewsteps',
                        type=int,
                        default=8,
                        help=('Steps between 0 and +maxskew/-maxskew to '
                              'estimate the possible skew of a region. Higher '
                              'values will be more precise but will also take '
                              'longer.')
                        )

    # line extraction
    g_ext = parser.add_argument_group('extraction parameters')
    g_ext.add_argument('-p', '--parallel',
                       type=int,
                       default=1,
                       help=('Number of threads parallely working on images. '
                             '(default:%(default)s)')
                       )
    g_ext.add_argument('-x', '--smearX',
                       type=float,
                       default=2,
                       help=('Smearing strength in X direction for the '
                             'algorithm calculating the textline polygon '
                             'wrapping all contents. (default:%(default)s)')
                       )
    g_ext.add_argument('-y', '--smearY',
                       type=float,
                       default=1,
                       help=('Smearing strength in Y direction for the '
                             'algorithm calculating the textline polygon '
                             'wrapping all contents. (default:%(default)s)')
                       )
    g_ext.add_argument('--growthX',
                       type=float,
                       default=1.1,
                       help=('Growth in X direction for every iteration of '
                             'the Textline polygon finding. Will speed up the '
                             'algorithm at the cost of precision. '
                             '(default: %(default)s)')
                       )
    g_ext.add_argument('--growthY',
                       type=float,
                       default=1.1,
                       help=('Growth in Y direction for every iteration of '
                             'the Textline polygon finding. Will speed up the '
                             'algorithm at the cost of precision. '
                             '(default: %(default)s)')
                       )
    g_ext.add_argument('--fail_save',
                       type=int,
                       default=1000,
                       help=('Fail save to counter infinite loops when '
                             'combining contours to a precise textlines. '
                             'Will connect remaining contours with lines. '
                             '(default: %(default)s)')
                       )

    # column parameters
    g_colb = parser.add_argument_group('Black column parameters')
    g_colb.add_argument('--max_blackseps','--maxseps',
                        # --maxseps to be consistent with ocropy
                        type=int,
                        default=0,
                        help=('Maximum # black column separators, '
                              'default: %(default)s')
                        )
    g_colb.add_argument('--widen_blackseps','--sepwiden',
                        # --sepwiden to be consistent with ocropy
                        type=int,
                        default=10,
                        help=('Widen black separators (to account for warping),'
                              ' default: %(default)s')
                        )
    g_colw = parser.add_argument_group('White column parameters')
    g_colw.add_argument('--max_whiteseps','--maxcolseps',
                        # --maxcolseps to be consistent with ocropy
                        type=int,
                        default=-1,
                        help=('Maximum # whitespace column separators. '
                              '(default: %(default)s)')
                        )
    g_colw.add_argument('--minheight_whiteseps', '--csminheight',
                        # --csminheight to be consistent with ocropy
                        type=float,
                        default=10,
                        help=('minimum column height (units=scale), '
                              'default: %(default)s')
                        )
                    
    args = parser.parse_args()
    s_print(args)

    with open(args.DATASET, 'r') as data_file:
        dataset = json.load(data_file)

    # Parallel processes for the pagexmllineseg cli
    def parallel(data):
        if len(data) == 3:
            image, pagexml, path_out = data
        elif len(data) == 2:
            image, pagexml = data
            path_out = pagexml
        else:
            raise ValueError("Invalid data line with length {} "
                             "instead of 2 or 3".format(len(data)))

        xml_output, _ = pagexmllineseg(pagexml, image,
                                       scale=args.scale,
                                       vscale=args.vscale,
                                       hscale=args.hscale,
                                       max_blackseps=args.max_blackseps,
                                       widen_blackseps=args.widen_blackseps,
                                       max_whiteseps=args.max_whiteseps,
                                       minheight_whiteseps=args.minheight_whiteseps,
                                       minscale=args.minscale,
                                       maxlines=args.maxlines,
                                       smear_strength=(args.smearX, args.smearY),
                                       growth=(args.growthX, args.growthY),
                                       fail_save_iterations=args.fail_save,
                                       maxskew=args.maxskew,
                                       skewsteps=args.skewsteps,
                                       usegauss=args.usegauss,
                                       remove_images=args.remove_images)
        with open(path_out, 'w+') as output_file:
            s_print("Save annotations into '{}'".format(path_out))
            output_file.write(xml_output)
    
    s_print("Process {} images, with {} in parallel".format(len(dataset), args.parallel))

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(args.parallel, len(dataset))) as pool:
        pool.map(parallel, dataset)


if __name__ == "__main__":
    cli()

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

from shapely.geometry import MultiPoint
from scipy.spatial import ConvexHull
import alphashape


from scipy.ndimage.filters import gaussian_filter, uniform_filter
import math
from typing import List, Tuple, Union

from lxml import etree
from PIL import Image, ImageDraw
from ocr4all_helpers.lib import imgmanipulate, morph, sl, pseg, nlbin

from multiprocessing.pool import ThreadPool
import json

import argparse
from matplotlib import pyplot as plt

import os
import sys

# Add printing for every thread
from threading import Lock
s_print_lock = Lock()


def s_print(*args, **kwargs):
    with s_print_lock:
        print(*args, **kwargs)


def s_print_error(*objs):
    s_print("ERROR: ", *objs, file=sys.stderr)


class Record(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def compute_lines(segmentation: np.ndarray, scale: int, filter_strength: float = 1.0) -> List[Record]:
    """Given a line segmentation map, computes a list of tuples consisting of 2D slices and masked images.

    Implementation derived from ocropy with changes to allow extracting the line coords/polygons
    """
    lobjects = morph.find_objects(segmentation)
    lines = []
    for idx, obj in enumerate(lobjects):
        if obj is None:
            continue
        if sl.dim1(obj) < 2*scale*filter_strength or sl.dim0(obj) < scale*filter_strength:
            continue
        mask = (segmentation[obj] == idx+1)
        if np.amax(mask) == 0:
            continue

        result = Record()
        result.label = idx+1
        result.bounds = obj
        polygon = []
        if ((segmentation[obj] != 0) == (segmentation[obj] != idx+1)).any():
            ppoints = shape_from_mask(mask, "bounding_box")
            polygon = [(obj[1].start+x, obj[0].start+y) for x, y in ppoints]
        if not polygon:
            polygon = [(obj[1].start, obj[0].start), (obj[1].stop,  obj[0].start),
                       (obj[1].stop,  obj[0].stop),  (obj[1].start, obj[0].stop)]
        result.polygon = polygon
        result.mask = mask
        lines.append(result)

    return lines


def compute_gradmaps(binary: np.array, scale: float, vscale: float = 1.0, hscale: float = 1.0,
                     usegauss: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses gradient filtering to find baselines
    """
    boxmap = pseg.compute_boxmap(binary, scale)
    cleaned = boxmap*binary
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned, (vscale*0.3*scale, hscale*6*scale), order=(1, 0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned, (max(4, vscale*0.3*scale), hscale*scale), order=(1, 0))
        grad = uniform_filter(grad, (vscale, hscale*6*scale))

    def norm_max(a):
        return a/np.amax(a)

    bottom = norm_max((grad<0)*(-grad))
    top = norm_max((grad > 0)*grad)
    return bottom, top, boxmap


def boundary(contour: np.ndarray) -> List[np.float64]:
    x_min = np.min(contour[:, 0])
    x_max = np.max(contour[:, 0])
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])

    return [x_min, x_max, y_min, y_max]


def shape_from_mask(mask: np.ndarray, shape: str) -> List[Tuple[np.float64, np.float64]]:
    dispatch_table = {
        "bounding_box": calc_bbox,
        "convex_hull": calc_convex_hull,
        "alphashape": calc_alphashape
    }

    return dispatch_table.get(shape)(mask)


def calc_bbox(mask: np.ndarray) -> List[Tuple[np.float64, np.float64]]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def calc_alphashape(mask: np.ndarray, alpha: float = 0.0):
    points = np.argwhere(mask).tolist()
    points = [(p[1], p[0]) for p in points]
    alpha_shape = alphashape.alphashape(points, alpha)

    return list(alpha_shape.exterior.coords)


def calc_convex_hull(mask: np.ndarray) -> List[Tuple[np.float64, np.float64]]:
    points = np.argwhere(mask).tolist()
    points = [(p[1], p[0]) for p in points]
    multi_points = MultiPoint(points)
    convex_hull = multi_points.convex_hull

    return list(convex_hull.exterior.coords)


def segment(im: Image, scale: float = None,
            max_blackseps: int = 0, widen_blackseps: int = 10,
            max_whiteseps: int = 3, minheight_whiteseps: int = 10,
            orientation: int = 0, vscale: float = 1.0, hscale: float = 1.0,
            minscale: float = 12.0, maxlines: int = 300,
            threshold: float = 0.2, usegauss: bool = False) -> Union[None, List[List[Tuple[int, int]]]]:
    """
    Segments a page into text lines.
    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.
    """
    colors = im.getcolors(2)
    if (im.mode not in ['1', "L"]) and not (colors is not None and len(colors) == 2):
        raise ValueError('Image is not bi-level')

    # rotate input image for vertical lines
    im_rotated = im.rotate(-1*orientation, expand=True)

    a = np.array(im_rotated.convert('L')) if im_rotated.mode == '1' else np.array(im_rotated)

    binary = np.array(a > 0.5*(np.amin(a) + np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = pseg.estimate_scale(binary)
    if scale < minscale:
        s_print_error(f"scale ({scale}) less than --minscale; skipping")
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
        s_print_error(f"too many lines {np.amax(segmentation)}")
        return

    lines_and_polygons = compute_lines(segmentation,
                                       scale)

    # Translate each point back to original
    delta_x = (im_rotated.width - im.width) / 2
    delta_y = (im_rotated.height - im.height) / 2
    center_x = im_rotated.width / 2
    center_y = im_rotated.height / 2

    def translate_back(point):
        # rotate point around center
        orient_rad = -1*orientation * (math.pi / 180)
        rotated_x = ((point[0]-center_x) * math.cos(orient_rad) - (point[1]-center_y) * math.sin(orient_rad) + center_x)
        rotated_y = ((point[0]-center_x) * math.sin(orient_rad) + (point[1]-center_y) * math.cos(orient_rad) + center_y)
        # move point
        return int(rotated_x-delta_x), int(rotated_y-delta_y)

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
                   maxskew=2.0,
                   skewsteps=8,
                   usegauss=False,
                   remove_images=False) -> Tuple[str, int]:
    name = os.path.splitext(os.path.split(imgpath)[-1])[0]
    s_print(f"""Start process for '{name}'
        |- Image: '{imgpath}'
        |- Annotations: '{xmlfile}' """)

    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    s_print(f"[{name}] Retrieve TextRegions")

    # convert point notation from older pagexml versions
    for coord in root.xpath("//ns:Coords[not(@points)]", namespaces=ns):
        cc = []
        for point in coord.xpath("./ns:Point", namespaces=ns):
            cx = point.attrib["x"]
            cy = point.attrib["y"]
            coord.remove(point)
            cc.append(f"{cx},{cy}")
        coord.attrib["points"] = " ".join(cc)

    coordmap = {}
    for text_region in root.xpath('//ns:TextRegion', namespaces=ns):
        rid = text_region.attrib["id"]
        coordmap[rid] = {"type": text_region.attrib.get("type", "TextRegion")}
        coordmap[rid]["coords"] = []
        for c in text_region.xpath("./ns:Coords", namespaces=ns) + text_region.xpath("./Coords"):
            coordmap[rid]["coordstring"] = c.attrib["points"]
            coordstrings = [x.split(",") for x in c.attrib["points"].split()]
            coordmap[rid]["coords"] += [[int(x[0]), int(x[1])]
                                        for x in coordstrings]
        if 'orientation' in text_region.attrib:
            coordmap[rid]["orientation"] = float(text_region.attrib["orientation"])

    s_print(f"[{name}] Extract Textlines from TextRegions")
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

    for idx, coord in enumerate(sorted(coordmap)):
        coords = coordmap[coord]['coords']

        if len(coords) < 3:
            continue
        cropped, [minX, minY, maxX, maxY] = imgmanipulate.cutout(im, coords)

        if 'orientation' in coordmap[coord]:
            orientation = coordmap[coord]['orientation']
        else:
            orientation = -1*nlbin.estimate_skew(cropped, 0, maxskew=maxskew,
                                                 skewsteps=skewsteps)
            s_print(f"[{name}] Skew estimate between +/-{maxskew} in {skewsteps} steps. Estimated {orientation}°")

        if cropped is not None:
            colors = cropped.getcolors(2)
            if not (colors is not None and len(colors) == 2):
                cropped = Image.fromarray(nlbin.adaptive_binarize(np.array(cropped)).astype(np.uint8))
            if coordmap[coord]["type"] == "drop-capital":
                lines = [1]
            else:
                # if line in
                lines = segment(cropped, scale=scale,
                                max_blackseps=max_blackseps,
                                widen_blackseps=widen_blackseps,
                                max_whiteseps=max_whiteseps,
                                minheight_whiteseps=minheight_whiteseps,
                                orientation=orientation,
                                vscale=vscale, hscale=hscale,
                                minscale=minscale, maxlines=maxlines,
                                usegauss=usegauss)

        else:
            lines = []

        _img = Image.open(imgpath)
        for poly in lines:
            ImageDraw.Draw(_img).polygon(poly)

        plt.figure(figsize=(20, 20))
        plt.imshow(_img)
        plt.show()

        # Interpret whole region as textline if no textlies are found
        if not lines or len(lines) == 0:
            coordstrg = " ".join([f"{str(int(x))},{str(int(y))}" for x, y in coords])
            textregion = root.xpath(f'//ns:TextRegion[@id="{coord}"]', namespaces=ns)[0]
            if orientation:
                textregion.set('orientation', str(orientation))
            linexml = etree.SubElement(textregion, "TextLine",
                                       attrib={"id": f"{coord}_l{str(idx+1).zfill(3)}"})
            etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})
        else:
            for idx, poly in enumerate(lines):
                if coordmap[coord]["type"] == "drop-capital":
                    coordstrg = coordmap[coord]["coordstring"]
                else:
                    coords = ((x+minX, y+minY) for x, y in poly)
                    coordstrg = " ".join([f"{str(int(x))},{str(int(y))}" for x, y in coords])

                textregion = root.xpath(f'//ns:TextRegion[@id="{coord}"]', namespaces=ns)[0]
                if orientation:
                    textregion.set('orientation', str(orientation))
                linexml = etree.SubElement(textregion, "TextLine",
                                           attrib={"id": f"{coord}_l{str(idx+1).zfill(3)}"})
                etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})

    s_print(f"[{name}] Generate new PAGE XML with text lines")
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
                        help='Maximal estimated skew of an image.'
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
    g_colb.add_argument('--max_blackseps', '--maxseps',
                        # --maxseps to be consistent with ocropy
                        type=int,
                        default=0,
                        help=('Maximum # black column separators, '
                              'default: %(default)s')
                        )
    g_colb.add_argument('--widen_blackseps', '--sepwiden',
                        # --sepwiden to be consistent with ocropy
                        type=int,
                        default=10,
                        help=('Widen black separators (to account for warping),'
                              ' default: %(default)s')
                        )
    g_colw = parser.add_argument_group('White column parameters')
    g_colw.add_argument('--max_whiteseps', '--maxcolseps',
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
            raise ValueError(f"Invalid data line with length {len(data)} "
                             "instead of 2 or 3")

        xml_output, _ = pagexmllineseg(pagexml, image,
                                       scale=args.scale,
                                       vscale=args.vscale,
                                       hscale=args.hscale,
                                       max_blackseps=args.max_blackseps,
                                       widen_blackseps=args.widen_blackseps,
                                       max_whiteseps=args.max_whiteseps,
                                       minheight_whiteseps=args.minheight_whiteseps,
                                       maxskew=args.maxskew,
                                       skewsteps=args.skewsteps,
                                       usegauss=args.usegauss,
                                       remove_images=args.remove_images)
        with open(path_out, 'w+') as output_file:
            s_print(f"Save annotations into '{path_out}'")
            output_file.write(xml_output)
    
    s_print(f"Process {len(dataset)} images, with {args.parallel} in parallel")

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(args.parallel, len(dataset))) as pool:
        pool.map(parallel, dataset)


if __name__ == "__main__":
    cli()

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

from ocr4all_helper_scripts.lib import imgmanipulate, morph, sl, pseg, nlbin
from ocr4all_helper_scripts.utils.datastructures import Record
from ocr4all_helper_scripts.utils import pageutils, imageutils

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import math

from lxml import etree
from PIL import Image
from shapely.geometry import Polygon
from shapely import affinity


# Add printing for every thread
from threading import Lock

s_print_lock = Lock()


def s_print(*args, **kwargs):
    with s_print_lock:
        print(*args, **kwargs)


def s_print_error(*objs):
    s_print("ERROR: ", *objs, file=sys.stderr)


def compute_lines(segmentation: np.ndarray,
                  smear_strength: Tuple[float, float],
                  scale: int,
                  growth: Tuple[float, float],
                  max_iterations: int,
                  filter_strength: float,
                  bounding_box: bool) -> List[Record]:
    """Given a line segmentation map, computes a list of tuples consisting of 2D slices and masked images.
    Implementation derived from ocropy with changes to allow extracting the line coords/polygons.
    """
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl.dim1(o) < 2 * scale * filter_strength or sl.dim0(o) < scale * filter_strength:
            s_print(f"Filter strength of {filter_strength} too high. Skipping detected line object…")
            continue
        mask = (segmentation[o] == i + 1)
        if np.amax(mask) == 0:
            continue

        result = Record()
        result.label = i + 1
        result.bounds = o
        polygon = []
        if ((segmentation[o] != 0) == (segmentation[o] != i + 1)).any() and not bounding_box:
            ppoints = approximate_smear_polygon(mask, smear_strength, growth, max_iterations)
            ppoints = ppoints[1:] if ppoints else []
            polygon = [(o[1].start + x, o[0].start + y) for x, y in ppoints]
        if not polygon:
            polygon = [(o[1].start, o[0].start), (o[1].stop, o[0].start),
                       (o[1].stop, o[0].stop), (o[1].start, o[0].stop)]
        result.polygon = polygon
        result.mask = mask
        lines.append(result)

    return lines


def compute_gradmaps(binary: np.array, scale: float, vscale: float = 1.0, hscale: float = 1.0,
                     usegauss: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use gradient filtering to find baselines
    """
    boxmap = pseg.compute_boxmap(binary, scale)
    cleaned = boxmap * binary
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0 * cleaned, (vscale * 0.3 * scale, hscale * 6 * scale), order=(1, 0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0 * cleaned, (max(4, vscale * 0.3 * scale), hscale * scale), order=(1, 0))
        grad = uniform_filter(grad, (vscale, hscale * 6 * scale))

    def norm_max(a):
        return a / np.amax(a)

    bottom = norm_max((grad < 0) * (-grad))
    top = norm_max((grad > 0) * grad)
    return bottom, top, boxmap


def boundary(contour: np.ndarray) -> List[np.float64]:
    """Calculates boundary of contour
    """
    x_min = np.min(contour[:, 0])
    x_max = np.max(contour[:, 0])
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])

    return [x_min, x_max, y_min, y_max]


def approximate_smear_polygon(line_mask: np.ndarray, smear_strength: Tuple[float, float] = (1.0, 2.0),
                              growth: Tuple[float, float] = (1.1, 1.1), max_iterations: int = 1000):
    """Approximate a single polygon around high pixels in a mask, via smearing
    """
    padding = 1
    work_image = np.pad(np.copy(line_mask), pad_width=padding, mode='constant', constant_values=False)

    contours = find_contours(work_image, 0.5, fully_connected="low")

    if len(contours) > 0:
        iteration = 1
        while len(contours) > 1:
            # Get bounds with dimensions
            bounds = [boundary(contour) for contour in contours]
            widths = [b[1] - b[0] for b in bounds]
            heights = [b[3] - b[2] for b in bounds]

            # Calculate x and y median distances (or at least 1)
            width_median = sorted(widths)[int(len(widths) / 2)]
            height_median = sorted(heights)[int(len(heights) / 2)]

            # Calculate x and y smear distance
            smear_distance_x = math.ceil(width_median * smear_strength[0] * (iteration * growth[0]))
            smear_distance_y = math.ceil(height_median * smear_strength[1] * (iteration * growth[1]))

            # Smear image in x and y direction
            height, width = work_image.shape
            gaps_current_x = [float('Inf')] * height
            for x in range(width):
                gap_current_y = float('Inf')
                for y in range(height):
                    if work_image[y, x]:
                        # Entered Contour
                        gap_current_x = gaps_current_x[y]

                        if gap_current_y < smear_distance_y and gap_current_y > 0:
                            # Draw over
                            work_image[y - gap_current_y:y, x] = True

                        if gap_current_x < smear_distance_x and gap_current_x > 0:
                            # Draw over
                            work_image[y, x - gap_current_x:x] = True

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
                s_print(f"Start fail save, since precise line generation took too many iterations ({iteration}).")
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
                            distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
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
        return [(p[1] - padding, p[0] - padding) for p in approximate_polygon(contours[0], 0.1)]
    return []


def segment(im: Image, scale: float = None, max_blackseps: int = 0, widen_blackseps: int = 10, max_whiteseps: int = 3,
            minheight_whiteseps: int = 10, filter_strength: float = 1.0,
            smear_strength: Tuple[float, float] = (1.0, 2.0), growth: Tuple[float, float] = (1.1, 1.1),
            orientation: int = 0, fail_save_iterations: int = 1000, vscale: float = 1.0, hscale: float = 1.0,
            minscale: float = 12.0, maxlines: int = 300, threshold: float = 0.2, usegauss: bool = False,
            bounding_box: bool = False):
    """
    Segments a page into text lines.
    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.
    """

    colors = im.getcolors(2)
    if (im.mode not in ['1', "L"]) and not (colors is not None and len(colors) == 2):
        raise ValueError('Image is not bi-level')

    # rotate input image for vertical lines
    im_rotated = im.rotate(-1 * orientation, expand=True)

    a = np.array(im_rotated.convert('L')) if im_rotated.mode == '1' else np.array(im_rotated)

    binary = np.array(a > 0.5 * (np.amin(a) + np.amax(a)), 'i')
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
    llabels = np.where(llabels1 > 0, llabels1, spread * binary)
    segmentation = llabels * binary

    if np.amax(segmentation) > maxlines:
        s_print_error(f"too many lines {np.amax(segmentation)}")
        return

    lines_and_polygons = compute_lines(segmentation,
                                       smear_strength,
                                       scale,
                                       growth,
                                       fail_save_iterations,
                                       filter_strength,
                                       bounding_box)

    # Translate each point back to original
    delta_x = (im_rotated.width - im.width) / 2
    delta_y = (im_rotated.height - im.height) / 2
    center_x = im_rotated.width / 2
    center_y = im_rotated.height / 2

    def translate_back(point):
        # rotate point around center
        orient_rad = -1 * orientation * (math.pi / 180)
        rotated_x = ((point[0] - center_x) * math.cos(orient_rad)
                     - (point[1] - center_y) * math.sin(orient_rad)
                     + center_x)
        rotated_y = ((point[0] - center_x) * math.sin(orient_rad)
                     + (point[1] - center_y) * math.cos(orient_rad)
                     + center_y)
        # move point
        return int(rotated_x - delta_x), int(rotated_y - delta_y)

    return [[translate_back(p) for p in record.polygon] for record in lines_and_polygons]


def pagelineseg(xmlfile: str,
                imgpath: str,
                scale: float = None,
                vscale: float = 1.0,
                hscale: float = 1.0,
                max_blackseps: int = 0,
                widen_blackseps: int = 10,
                max_whiteseps: int = -1,
                minheight_whiteseps: int = 10,
                minscale: float = 12.0,
                maxlines: int = 300,
                smear_strength: Tuple[float, float] = (1.0, 2.0),
                growth: Tuple[float, float] = (1.1, 1.1),
                filter_strength: float = 1.0,
                fail_save_iterations: int = 100,
                maxskew: float = 2.0,
                skewsteps: int = 8,
                usegauss: bool = False,
                remove_images: bool = False,
                bounding_box: bool = False):
    name = Path(imgpath).name.split(".")[0]
    s_print(f"""Start process for '{name}'
        |- Image: '{imgpath}'
        |- Annotations: '{xmlfile}' """)

    root = pageutils.get_root(xmlfile)

    s_print(f"[{name}] Retrieve TextRegions")

    pageutils.convert_point_notation(root)

    coordmap = pageutils.construct_coordmap(root)

    s_print(f"[{name}] Extract Textlines from TextRegions")

    im = Image.open(imgpath)
    width, height = im.size

    if remove_images:
        imageutils.remove_images(im, root)

    pageutils.remove_existing_textlines(root)

    for coord_idx, coord in enumerate(sorted(coordmap)):
        region_coords = coordmap[coord]['coords']

        if len(region_coords) < 3:
            continue

        cropped, [min_x, min_y, max_x, max_y] = imgmanipulate.cutout(im, region_coords)

        if coordmap[coord].get("orientation"):
            orientation = coordmap[coord]['orientation']
        else:
            orientation = -1 * nlbin.estimate_skew(cropped, 0, maxskew=maxskew,
                                                   skewsteps=skewsteps)
            s_print(f"[{name}] Skew estimate between +/-{maxskew} in {skewsteps} steps. Estimated {orientation}°")

        if cropped is not None:
            colors = cropped.getcolors(2)
            if not (colors is not None and len(colors) == 2):
                cropped = Image.fromarray(nlbin.adaptive_binarize(np.array(cropped)).astype(np.uint8))
            if coordmap[coord]["type"] == "drop-capital":
                lines = [1]
            else:
                lines = segment(cropped,
                                scale=scale,
                                max_blackseps=max_blackseps,
                                widen_blackseps=widen_blackseps,
                                max_whiteseps=max_whiteseps,
                                minheight_whiteseps=minheight_whiteseps,
                                filter_strength=filter_strength,
                                smear_strength=smear_strength,
                                growth=growth,
                                orientation=orientation,
                                fail_save_iterations=fail_save_iterations,
                                vscale=vscale,
                                hscale=hscale,
                                minscale=minscale,
                                maxlines=maxlines,
                                usegauss=usegauss,
                                bounding_box=bounding_box)

        else:
            lines = []

        # Interpret whole region as TextLine if no TextLines are found
        if not lines or len(lines) == 0:
            coord_str = " ".join([f"{x},{y}" for x, y in region_coords])
            textregion = root.find(f'.//{{*}}TextRegion[@id="{coord}"]')
            if orientation:
                textregion.set('orientation', str(orientation))
            linexml = etree.SubElement(textregion, "TextLine",
                                       attrib={"id": "{}_l{:03d}".format(coord, coord_idx + 1)})
            etree.SubElement(linexml, "Coords", attrib={"points": coord_str})
        else:
            for poly_idx, poly in enumerate(lines):
                if coordmap[coord]["type"] == "drop-capital":
                    coord_str = coordmap[coord]["coordstring"]
                else:
                    line_coords = Polygon([(x + min_x, y + min_y) for x, y in poly])
                    sanitized_coords = pageutils.sanitize(line_coords, Polygon(region_coords), width, height)
                    coord_str = " ".join([f"{int(x)},{int(y)}" for x, y in sanitized_coords])

                textregion = root.find(f'.//{{*}}TextRegion[@id="{coord}"]')
                if orientation:
                    textregion.set('orientation', str(orientation))
                linexml = etree.SubElement(textregion, "TextLine",
                                           attrib={"id": "{}_l{:03d}".format(coord, poly_idx + 1)})
                etree.SubElement(linexml, "Coords", attrib={"points": coord_str})

    s_print(f"[{name}] Generate new PAGE XML with text lines")
    xmlstring = etree.tounicode(root.getroottree()).replace(
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")
    return xmlstring

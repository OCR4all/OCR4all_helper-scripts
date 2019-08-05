# -*- coding: utf-8 -*-
# Line segmentation script for images with PAGE xml.
# Derived from the `import_from_larex.py` script from the nashi project:
# https://github.com/andbue/nashi

import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import binary_dilation
import math

from lxml import etree
from PIL import Image, ImageDraw

from kraken import pageseg, binarization
from kraken.lib import morph, sl
from kraken.lib.util import pil2array
from kraken.binarization import is_bitonal
from kraken.lib.exceptions import KrakenInputException

from multiprocessing.pool import ThreadPool
import json

import argparse

import os

# Add printing for every thread
from threading import Lock
s_print_lock = Lock()
def s_print(*a,**b):
    with s_print_lock:
        print(*a,**b)


def cutout(im, coords):
    """
        Cut out coords from image, crop and return new image.
    """
    coords = [tuple(t) for t in coords]
    if not coords:
        return None
    maskim = Image.new('1', im.size, 0)
    ImageDraw.Draw(maskim).polygon(coords, outline=1, fill=1)
    new = Image.new(im.mode, im.size, "white")
    masked = Image.composite(im, new, maskim)
    cropped = masked.crop([
            min([x[0] for x in coords]), min([x[1] for x in coords]),
            max([x[0] for x in coords]), max([x[1] for x in coords])])
    return cropped


class record(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def compute_lines(segmentation, spread, scale, tolerance):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl.dim1(o) < 2*scale or sl.dim0(o) < scale:
            continue
        mask = (segmentation[o] == i+1)
        if np.amax(mask) == 0:
            continue

        result = record()
        result.label = i+1
        result.bounds = o
        polygon = []
        if ((segmentation[o] != 0) == (segmentation[o] != i+1)).any():
            ppoints = draw_polygon(mask, smear_strength, growth)
            ppoints = ppoints[1:] if ppoints else []
            polygon = [(o[0].start+p[0], o[1].start+p[1]) for p in ppoints]
        if not polygon:
            polygon = [(o[0].start, o[1].start), (o[0].stop,  o[1].start),
                       (o[0].stop,  o[1].stop),  (o[0].start, o[1].stop)]
        result.polygon = polygon
        result.mask = mask
        lines.append(result)
    return lines


def draw_polygon(lspread, smear_strength, growth):
    """Draws a polygon around area of value lineno in array lspread."""
    
    cont = approximate_smear_polygon(lspread,smear_strength=smear_strength,growth=growth)
    if len(cont) == 1:
        polyg = approximate_polygon(cont[0], tolerance=tolerance).astype(int)
        return [(p[0]-1, p[1]-1) for p in polyg]
    else:
        return [(p[0]-1, p[1]-1) for p in cont]


def boundary(contour):
    Xmin = np.min(contour[:,0])
    Xmax = np.max(contour[:,0])
    Ymin = np.min(contour[:,1])
    Ymax = np.max(contour[:,1])

    return [Xmin, Xmax, Ymin, Ymax]


def approximate_smear_polygon(line_mask, smear_strength=(1,2), growth=(1.1, 1.1), maxIterations=10):
    work_image = np.copy(line_mask)

    contours = find_contours(np.pad(work_image, pad_width=1, mode='constant', constant_values=False), 0.5, fully_connected="low")

    if len(contours) > 0:
        iteration = 0
        while len(contours) > 1:
            # Get bounds sorted by x and y
            bounds = [boundary(contour) for contour in contours]
            sorted_x = sorted(bounds, key=lambda b: (b[0], b[2]))
            sorted_y = sorted(bounds, key=lambda b: (b[2], b[0]))

            # Calculate x and y distances between neighboring bounds
            distances_x = [c2[0]-c1[1] for c1, c2 in zip(sorted_x, sorted_x[1:]) if c2[0]-c1[1] > 0]
            distances_y = [c2[2]-c1[3] for c1, c2 in zip(sorted_y, sorted_y[1:]) if c2[2]-c1[3] > 0]

            # Calculate x and y median distances (or at least 1)
            dist_x_median = sorted(distances_x)[int(len(distances_x) / 2)] if len(distances_x) > 0 else 1
            dist_y_median = sorted(distances_y)[int(len(distances_y) / 2)] if len(distances_y) > 0 else 1

            # Calculate x and y smear distance 
            smear_distance_x = math.ceil(dist_x_median*smear_strength * (iteration*growth[0]))
            smear_distance_y = math.ceil(dist_y_median*smear_strength * (iteration*growth[1]))

            # Smear image in x and y direction
            width, height = work_image.shape
            gaps_current_x = [float('Inf')]*height
            for x in range(width):
                gap_current_y = float('Inf')
                for y in range(height):
                    if work_image[x, y]:
                        # Entered Contour
                        gap_current_x = gaps_current_x[y]

                        if gap_current_y < smear_distance_y and gap_current_y > 0:
                            # Draw over
                            work_image[x, y-gap_current_y:y] = True 
                        
                        if gap_current_x < smear_distance_x and gap_current_x > 0:
                            #Draw over
                            work_image[x-gap_current_x:x, y] = True 

                        gap_current_y = 0
                        gaps_current_x[y] = 0
                    else:
                        # Entered/Still in Gap
                        gap_current_y += 1
                        gaps_current_x[y] += 1

            # Find contours of current smear
            contours = find_contours(np.pad(work_image, pad_width=1, mode='constant', constant_values=False), 0.5, fully_connected="low")
            iteration += 1

        return contours[0]
    return []
    

def segment(im, scale=None, maxcolseps=2, black_colseps=False, tolerance=1):
    """
    Segments a page into text lines.
    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.
    Args:
        im (PIL.Image): A bi-level page of mode '1' or 'L'
        scale (float): Scale of the image
        maxcolseps (int): Maximum number of whitespace column separators
        black_colseps (bool): Whether column separators are assumed to be
                              vertical black lines or not
        tolerance (float): Tolerance for the polygons wrapping textlines
    Returns:
        {'boxes': [(x1, y1, x2, y2),...]}: A
        dictionary containing the text direction and a list of reading order
        sorted bounding boxes under the key 'boxes'.
    Raises:
        KrakenInputException if the input image is not binarized or the text
        direction is invalid.
    """

    if im.mode != '1' and not is_bitonal(im):
        raise KrakenInputException('Image is not bi-level')

    # rotate input image for vertical lines
    angle = 0
    offset = (0, 0)

    im = im.rotate(angle, expand=True)

    # honestly I've got no idea what's going on here. In theory a simple
    # np.array(im, 'i') should suffice here but for some reason the
    # tostring/fromstring magic in pil2array alters the array in a way that is
    # needed for the algorithm to work correctly.
    a = pil2array(im)
    binary = np.array(a > 0.5*(np.amin(a) + np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = pageseg.estimate_scale(binary)

    binary = pageseg.remove_hlines(binary, scale)
    # emptyish images will cause exceptions here.
    try:
        if black_colseps:
            colseps, binary = pageseg.compute_black_colseps(binary, scale,
                                                            maxcolseps)
        else:
            colseps = pageseg.compute_white_colseps(binary, scale, maxcolseps)
    except ValueError:
        return {'boxes':  []}

    bottom, top, boxmap = pageseg.compute_gradmaps(binary, scale)
    seeds = pageseg.compute_line_seeds(binary, bottom, top, colseps, scale)
    llabels1 = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = np.where(llabels1 > 0, llabels1, spread*binary)
    segmentation = llabels*binary

    lines_and_polygons = compute_lines(segmentation, spread, scale, tolerance)
    # TODO: rotate_lines for polygons
    order = pageseg.reading_order([l.bounds for l in lines_and_polygons])
    lsort = pageseg.topsort(order)
    lines = [lines_and_polygons[i].bounds for i in lsort]
    lines = [(s2.start, s1.start, s2.stop, s1.stop) for s1, s2 in lines]
    return {'boxes': pageseg.rotate_lines(lines, 360-angle, offset).tolist(),
            'lines': lines_and_polygons,
            'script_detection': False}


def pagexmllineseg(xmlfile, imgpath, scale=None, tolerance=1):
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
            # coordstrings = [x.split(",") for x in c.attrib["points"].split()]
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

    filename = root.xpath('//ns:Page', namespaces=ns)[0]\
        .attrib["imageFilename"]

    s_print("[{}] Extract Textlines from TextRegions".format(name))
    im = Image.open(imgpath)

    for n, c in enumerate(sorted(coordmap)):
        if type(scale) == dict:
            if coordmap[c]['type'] in scale:
                rscale = scale[coordmap[c]['type']]
            elif "other" in scale:
                rscale = scale["other"]
            else:
                rscale = None
        else:
            rscale = scale
        coords = coordmap[c]['coords']
        if len(coords) < 3:
            continue
        cropped = cutout(im, coords)
        offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
        if cropped is not None:
            if not binarization.is_bitonal(cropped):
                try:
                    cropped = binarization.nlbin(cropped)
                except SystemError:
                    continue
            if coordmap[c]["type"] == "drop-capital":
                lines = [1]
            else:
                # if line in
                lines = segment(cropped, scale=rscale, maxcolseps=-1, tolerance=tolerance)

                lines = lines["lines"] if "lines" in lines else []
        else:
            lines = []

        # Iterpret whole region as textline if no textline are found
        if not(lines) or len(lines) == 0:
            coordstrg = " ".join([str(x[0])+","+str(x[1]) for x in coords])
            textregion = root.xpath('//ns:TextRegion[@id="'+c+'"]', namespaces=ns)[0]
            linexml = etree.SubElement(textregion, "TextLine",
                                       attrib={"id": "{}_l{:03d}".format( c, n+1)})
            coordsxml = etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})

        else:
            for n, l in enumerate(lines):
                if coordmap[c]["type"] == "drop-capital":
                    coordstrg = coordmap[c]["coordstring"]
                else:
                    coords = ((x[1]+offset[0], x[0]+offset[1]) for x in l.polygon)
                    coordstrg = " ".join([str(int(x[0]))+","+str(int(x[1])) for x in coords])
                textregion = root.xpath('//ns:TextRegion[@id="'+c+'"]', namespaces=ns)[0]
                linexml = etree.SubElement(textregion, "TextLine",
                                           attrib={"id": "{}_l{:03d}".format( c, n+1)})
                coordsxml = etree.SubElement(linexml, "Coords", attrib={"points": coordstrg})

    s_print("[{}] Generate new PAGE xml with textlines".format(name))
    xmlstring = etree.tounicode(root.getroottree()).replace(
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    no_lines_segm = int(root.xpath("count(//TextLine)"))
    return xmlstring, no_lines_segm

def main():
    parser = argparse.ArgumentParser("""
    Line segmentation with regions read from a PAGE xml file
    """)
    parser.add_argument('DATASET',type=str,help='Path to the input dataset in json format with a list of image path, pagexml path and optional output path. (Will overwrite pagexml if no output path is given)') 
    parser.add_argument('-s','--scale', type=float, default=None, help='Scale of the input image used for the line segmentation. Will be estimated if not defined.')
    parser.add_argument('-p','--parallel', type=int, default=1, help='Number of threads parallely working on images. (default:1)')
    parser.add_argument('-t','--tolerance', type=float, default=1, help='Tolerance for the polygons wrapping textlines (default:1)')
                    
    args = parser.parse_args()

    with open(args.DATASET, 'r') as data_file:
        dataset = json.load(data_file)

    # Parallel processes for the pagexmllineseg
    def parallel(data):
        image,pagexml = data[:2]
        pagexml_out = data[2] if (len(data) > 2 and data[2] is not None) else pagexml

        xml_output, number_lines = pagexmllineseg(pagexml, image, scale=args.scale, tolerance=args.tolerance)
        with open(pagexml_out, 'w+') as output_file:
            s_print("Save annotations into '{}'".format(pagexml_out))
            output_file.write(xml_output)
    
    s_print("Process {} images, with {} in parallel".format(len(dataset), args.parallel))

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(args.parallel,len(dataset))) as pool:
        output = pool.map(parallel,dataset)
    

if __name__ == "__main__":
    main()

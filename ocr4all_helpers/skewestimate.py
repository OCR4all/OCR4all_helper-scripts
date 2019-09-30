# -*- coding: utf-8 -*-
# Skew estimate script for regions of images with PAGE xml.

import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import math

from lxml import etree
from PIL import Image
from ocr4all_helpers.lib import imgmanipulate, morph, sl, pseg, nlbin

from multiprocessing.pool import ThreadPool
import json

import argparse

import os


# Add printing for every thread
from threading import Lock
s_print_lock = Lock()
def s_print(*a, **b):
    with s_print_lock:
        print(*a, **b)


def pagexmlskewestimate(xmlfile, imgpath, from_scratch=False):
    name = os.path.splitext(os.path.split(imgpath)[-1])[0]
    s_print("""Start process for '{}'
        |- Image: '{}'
        |- Annotations: '{}' """.format(name, imgpath, xmlfile))

    im = Image.open(imgpath)
    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    regions = root.xpath('//ns:TextRegion', namespaces=ns)
    for n, region in enumerate(regions):
        s_print("[{}] Calculate skew of {}/{}".format(name, n, len(regions)))

        # Read coords
        for c in region.xpath("./ns:Coords", namespaces=ns) + region.xpath("./Coords"):
            coords = []
            if "points" in c.attrib:
                coordstrings = [x.split(",") for x in c.attrib["points"].split()]
                coords += [[int(x[0]), int(x[1])] for x in coordstrings]
            else:
                for point in c.xpath("./ns:Point", namespaces=ns):
                    cx = point.attrib["x"]
                    cy = point.attrib["y"]
                    coords.append([int(cx), int(cy)])

        # Read orientation
        if len(coords) > 2 and ('orientation' not in region.attrib or from_scratch):
            cropped, _ = imgmanipulate.cutout(im, coords)
            orientation = -1*nlbin.estimate_skew(cropped)
            region.set('orientation', str(orientation))

    s_print("[{}] Add all orientations in annotation file".format(name))
    xmlstring = etree.tounicode(root.getroottree())
    no_lines_segm = int(root.xpath("count(//TextLine)"))
    return xmlstring, no_lines_segm


def main():
    parser = argparse.ArgumentParser("""
    Calculate skew angles for regions read from a PAGE xml file
    """)
    parser.add_argument('DATASET', type=str, help=
            'Path to the input dataset in json format with a list of image path,'+
            'pagexml path and optional output path. (Will overwrite pagexml if no output path is given)') 
    parser.add_argument('-s', '--from_scratch', action='store_true', help=
            'Overwrite existing orientation angles, by calculating them from scratch.')

    args = parser.parse_args()

    with open(args.DATASET, 'r') as data_file:
        dataset = json.load(data_file)

    # Parallel processes for the pagexmllineseg
    def parallel(data):
        image, pagexml = data[:2]
        pagexml_out = data[2] if (len(data) > 2 and data[2] is not None) else pagexml

        xml_output, number_lines = pagexmlskewestimate(pagexml, image, args.from_scratch)
        with open(pagexml_out, 'w+') as output_file:
            s_print("Save annotations into '{}'".format(pagexml_out))
            output_file.write(xml_output)
    
    s_print("Process {} images, with {} in parallel".format(len(dataset), args.parallel))

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(args.parallel, len(dataset))) as pool:
        output = pool.map(parallel, dataset)
    

if __name__ == "__main__":
    main()

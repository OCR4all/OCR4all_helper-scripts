# -*- coding: utf-8 -*-
# Skew estimate script for regions of images with PAGE xml.
from lxml import etree
from PIL import Image
from ocr4all_helper_scripts.lib import imgmanipulate, nlbin

import os


# Add printing for every thread
from threading import Lock
s_print_lock = Lock()


def s_print(*args, **kwargs):
    with s_print_lock:
        print(*args, **kwargs)


def pagexmlskewestimate(xmlfile: str, imgpath: str, from_scratch: bool = False, maxskew: int = 2, skewsteps: int = 8):
    name = os.path.splitext(os.path.split(imgpath)[-1])[0]
    s_print(f"""Start process for '{name}'
        |- Image: '{imgpath}'
        |- Annotations: '{xmlfile}' """)

    im = Image.open(imgpath)
    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    regions = root.xpath('//ns:TextRegion', namespaces=ns)
    for n, region in enumerate(regions):
        s_print(f"[{name}] Calculate skew of {n}/{len(regions)}")

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
            orientation = -1*nlbin.estimate_skew(cropped, 0, maxskew=maxskew,
                                                 skewsteps=skewsteps)
            region.set('orientation', str(orientation))

    s_print(f"[{name}] Add all orientations in annotation file")
    xmlstring = etree.tounicode(root.getroottree())
    no_lines_segm = int(root.xpath("count(//TextLine)"))
    return xmlstring, no_lines_segm

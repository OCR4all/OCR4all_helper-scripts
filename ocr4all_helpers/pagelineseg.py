from skimage import draw, transform, measure
import numpy as np
import math
import json
from lxml import etree

import argparse
from multiprocessing.pool import ThreadPool
from threading import Lock
from collections import namedtuple
from kraken import pageseg, binarization
from kraken.lib import morph


# Named tuples
Region = namedtuple("Region", "coords type lines orientation")
Rectangle = namedtuple("Rectangle", "left right top bottom")


# Thread save print
s_print_lock = Lock()
def s_print(*a, **b):
    with s_print_lock:
        print(*a, **b)

### Command line interface ###
def cli():
    parser = argparse.ArgumentParser(""" Line segmentation with regions read from a PAGE xml file """)
    parser.add_argument('DATASET',type=str,help='Path to the input dataset in json format with a list of image path, pagexml path and optional output path. (Will overwrite pagexml if no output path is given)') 
    parser.add_argument('-s','--scale', type=float, default=None, help='Scale of the input image used for the line segmentation. Will be estimated if not defined.')
    parser.add_argument('-p','--parallel', type=int, default=1, help='Number of threads parallely working on images. (default:%(default)s)')
    parser.add_argument('-x','--smearX', type=float, default=2, help='Smearing strength in X direction for the algorithm calculating the textline polygon wrapping all contents. (default:%(default)s)')
    parser.add_argument('-y','--smearY', type=float, default=1, help='Smearing strength in Y direction for the algorithm calculating the textline polygon wrapping all contents. (default:%(default)s)')
    parser.add_argument('--growthX', type=float, default=1.1, help='Growth in X direction for every iteration of the Textline polygon finding. Will speed up the algorithm at the cost of precision. (default: %(default)s)')
    parser.add_argument('--growthY', type=float, default=1.1, help='Growth in Y direction for every iteration of the Textline polygon finding. Will speed up the algorithm at the cost of precision. (default: %(default)s)')
    parser.add_argument('--maxcolseps', type=int, default=-1, help='Maximum # whitespace column separators, (default: %(default)s)')
    parser.add_argument('--fail_save', type=int, default=1000, help='Fail save to counter infinite loops when combining contours to a precise textlines. Will connect remaining contours with lines. (default: %(default)s)')

    args = parser.parse_args()

    # Parallel processes for the pagexmllineseg
    def cli_process(data):
        image,pagexml = data[:2]
        pagexml_out = data[2] if (len(data) > 2 and data[2] is not None) else pagexml

        pagexml_tree = etree.parse(pagexml).getroot()

        regions = extract_regions(pagexml_tree)

        xml_output, number_lines = pagexmllineseg(pagexml, image, 
                                                    scale=args.scale,
                                                    maxcolseps=args.maxcolseps, 
                                                    smear_strength=(args.smearX, args.smearY), 
                                                    growth=(args.growthX,args.growthY),
                                                    fail_save_iterations=args.fail_save)
        

        xml_output = update_pagexml(pagexml_tree, segmented_regions)

        with open(pagexml_out, 'w+') as output_file:
            s_print("Save annotations into '{}'".format(pagexml_out))
            output_file.write(xml_output)

    # Load dataset from json
    with open(args.DATASET, 'r') as data_file:
        dataset = json.load(data_file)

    s_print("Process {} images, with {} in parallel".format(len(dataset), args.parallel))

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(args.parallel,len(dataset))) as pool:
        output = pool.map(cli_process,dataset)


### Segmentation ###
# Segment an image into text lines
def segment_region(region, region_cutout, scale, max_colseps, black_colseps):
    binary = 1 - np.array(region_cutout > 0.5*(np.amin(region_cutout) + np.amax(region_cutout)), 'i')
    width, height = binary.shape

    if not region.orientation:
        region.orientation = estimate_skew(binary)

    if region.orientation != 0:
        binary = transform.rotate(binary, region.orientation,
                                resize=True, center=(width/2, height/2))

    if not scale:
        scale = pageseg.estimate_scale(binary)

    binary = pageseg.remove_hlines(binary, scale)

    try:
        if black_colseps:
            colseps, binary = pageseg.compute_black_colseps(binary, scale, max_colseps)
        else:
            colseps = pageseg.compute_white_colseps(binary, scale, max_colseps)
    except ValueError:
        # Caused by empty(-ish) regions
        return region

    ## Line extraction
    # Segment lines
    bottom, top, boxmap = pageseg.compute_gradmaps(binary, scale)
    seeds = pageseg.compute_line_seeds(binary, bottom, top, colseps, scale)
    llabels1 = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = np.where(llabels1 > 0, llabels1, spread*binary)
    segmentation = llabels*binary

    # Filter lines
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if o[1].stop-o[1].start < 2*scale or o[0].stop-o[0].start < scale:
            continue
        mask = (segmentation[o] == i+1)
        if np.amax(mask) == 0:
            continue

        polygon = []
        if ((segmentation[o] != 0) == (segmentation[o] != i+1)).any():
            ppoints = approximate_smear_polygon(mask, smear_strength, growth, max_iterations)
            ppoints = ppoints[1:] if ppoints else []
            polygon = [(o[0].start+p[0], o[1].start+p[1]) for p in ppoints]
        if not polygon:
            polygon = [(o[0].start, o[1].start), (o[0].stop,  o[1].start),
                       (o[0].stop,  o[1].stop),  (o[0].start, o[1].stop)]
        lines.append(polygon)




compute_lines()





# Estimate the skew of an image
def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    ''' estimate skew angle '''
    width, height = flat.size
    border_x, border_y = int(bignore*width), int(bignore*height)

    flat = np.amax(flat)-flat
    flat -= np.amin(flat)

    est = flat[border_x:width-border_x, border_y:height-border_y]

    angles = np.linspace(-maxskew, maxskew, int(2*maxskew*skewsteps)+1)

    # Find best angle via varianz
    estimates = []
    for angle in angles:
        rotated = transform.rotate(est, angle, resize=True)
        v = np.mean(np.array(rotated),axis=1)
        v = np.var(v)
        estimates.append((v,angle))
    _, angle = max(estimates)
    return angle

### Image manipulation ###
def cutout_region(gray, region):
    if len(region.coods) > 0:
        mask = np.zeros(gray.shape(), dtype=bool)
        xl, yl = zip(region.coords)
        mask[draw.polygon(xl, yl)] = True

        left = min(xl)
        right = max(xl)
        top = min(yl)
        bottom = max(yl)

        cutout = (gray[mask])[left:right, top:bottom]
        return (cutout, Rectangle(left, right, top, bottom))
    else:
        raise TypeError("Can't cut region from image. Region coords are empty")


### Page XML IO ###
# Extract all regions of a pagexml file (Existing TextLines will be ignored)
def extract_regions(xmltree):
    ns = {"ns": xmltree.nsmap[None]}

    regions = {}
    for r in xmltree.xpath('//ns:TextRegion', namespaces=ns):
        region_id = r.attrib["id"]
        region_type = {"type": r.attrib["type"]}

        for c in r.xpath("./ns:Coords", namespaces=ns) + r.xpath("./Coords"):
            if "points" in c.attrib:
                def point(point_str):
                    xs, ys = point_str.split(",")
                    return (int(xs), int(ys))

                coords = [point(ps) for ps in c.attrib["points"].split(" ")]
            else:
                coords = []
                for point in c.xpath("./ns:Point", namespaces=ns):
                    x = point.attrib["x"]
                    y = point.attrib["y"]
                    coords.append((x, y))

        if 'orientation' in r.attrib:
            orientation = float(r.attrib["orientation"])
        else:
            orientation = None

        regions[region_id] = Region(coords, region_type, [], orientation)

    return regions

# Include updated regions and save pagexml as 2017-07-15 version
def update_pagexml(xmltree, regions):
    ns = {"ns": xmltree.nsmap[None]}

    def coords(points):
        return {"points": " ".join(str(p[0])+","+str(p[1]) for p in points)}

    for n, (r_id, region) in enumerate(regions):
        textregion = xmltree.xpath('//ns:TextRegion[@id="'+r_id+'"]', namespaces=ns)[0]
        if region.orientation:
            textregion.attrib['orientation'] = region.orientation
        textregion.attrib["Coords"] = coords(region.coords)

        def l_id(index):
            l_id = "{}_l{:03d}".format(r_id, index)

        # TextLines
        if len(region.lines) == 0:
            # Iterpret whole region as textline if no textline are found
            linexml = etree.SubElement(textregion, "TextLine", {"id": l_id(n+1)})
            etree.SubElement(linexml, "Coords", coords(region.coords))
        else:
            for ln, l in enumerate(region.lines):
                linexml = etree.SubElement(textregion, "TextLine", {"id": l_id(ln+1)})
                etree.SubElement(linexml, "Coords", coords(l))

    # Set PageXML Version to 2017-07-15
    xmlstring = etree.tounicode(xmltree.getroottree()).replace(
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    return xmlstring


if __name__ == "__main__":
    cli()

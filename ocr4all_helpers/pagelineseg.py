from skimage import io, draw, transform
from scipy.ndimage import filters, interpolation

import numpy as np
import json
from lxml import etree

import argparse
from multiprocessing.pool import ThreadPool
from threading import Lock
from collections import namedtuple
from kraken import pageseg
from kraken.lib import morph


# Named tuples
Region = namedtuple("Region", "coords type lines orientation")
Rectangle = namedtuple("Rectangle", "left right top bottom")
LineDescriptor = namedtuple("LineDescriptor", "label bounds mask")

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

    # (Parallel) processes for the line segmentation
    def cli_process(data):
        image_path, pagexml = data[:2]
        image = io.imread(image_path, as_gray=True)
        pagexml_out = data[2] if (len(data) > 2 and data[2] is not None) else pagexml

        pagexml_tree = etree.parse(pagexml).getroot()

        regions = extract_regions(pagexml_tree)
        #TODO add more params
        region_cutouts = {r_id: cutout_region(region, image) for r_id, region in regions.items()}
        segmented_regions = {r_id: segment_region(region, region_cutouts[r_id][0]) for r_id, region in regions.items()}

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
def segment_region(region, region_cutout, max_colseps=1, scale=None, black_colseps=False, pad=3, expand=3, noise=8):
    binary = 1 - (region_cutout > 0.5*(np.amin(region_cutout) + np.amax(region_cutout)))

    width, height = binary.shape

    region_orientation = region.orientation

    if not region_orientation:
        region_orientation = estimate_skew(binary)

    if region_orientation != 0:
        binary = transform.rotate(binary, region_orientation,
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

    # Filter and extract line segments from segmentations
    lobjects = morph.find_objects(segmentation)
    linedescriptors = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        dim0 = o[0].stop-o[0].start
        dim1 = o[1].stop-o[1].start
        if dim1 < 2*scale or dim0 < scale: continue
        mask = (segmentation[o]==i+1)
        if np.amax(mask) == 0: continue
        linedescriptors.append(LineDescriptor(i+1, o, mask))
    
    # Reading Order?
    
    # Extract poligons of line segments
    lines = []
    for linedesc in linedescriptors:
        bounds = linedesc.bound

        # padding
        if pad > 0:
            mask = pad_image(linedesc.mask, pad, cval=0)
        else:
            mask = linedesc.mask

        mask = remove_noise(mask, noise)
        line = extract(mask,
                    int(bounds[0].start) - pad,
                    int(bounds[1].start) - pad,
                    int(bounds[0].stop) + pad,
                    int(bounds[1].stop) + pad)
        if expand > 0:
            mask = filters.maximum_filter(mask, (expand, expand))
        line_mask = np.where(mask, line, np.amax(line))
        #TODO calc polys from line
        #line = … line_mask …
        #region.lines.append(line)

    #TODO rotate back, offset etc.
    return Region(region.coords, region.type, lines, region_orientation)


# Estimate the skew of an image
def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    ''' estimate skew angle '''
    width, height = flat.shape
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
# Cut a region out of a gray image
def cutout_region(region, gray):
    if len(region.coords) > 0:
        mask = np.zeros(gray.shape, dtype=bool)
        xl, yl = zip(*region.coords)
        xp, yp = draw.polygon(xl, yl)

        mask[yp, xp] = True

        left = min(xl)
        right = max(xl)
        top = min(yl)
        bottom = max(yl)
        
        cutout = np.ones_like(gray)*255
        cutout[mask] = gray[mask]
        return (cutout[top:bottom, left:right], Rectangle(left, right, top, bottom))
    else:
        raise TypeError("Can't cut region from image. Region coords are empty")


# TODO replace (ocropy)
def pad_image(image,d,cval=np.inf):
    result = np.ones(np.array(image.shape)+2*d)
    result[:,:] = np.amax(image) if cval==np.inf else cval
    result[d:-d,d:-d] = image
    return result

# TODO replace (ocropy)
def extract(image,y0,x0,y1,x1,mode='nearest',cval=0):
    h,w = image.shape
    ch,cw = y1-y0,x1-x0
    y,x = np.clip(y0,0,max(h-ch,0)),np.clip(x0,0,max(w-cw, 0))
    sub = image[y:y+ch,x:x+cw]
    try:
        r = interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        if cw > w or ch > h:
            pady0, padx0 = max(-y0, 0), max(-x0, 0)
            r = interpolation.affine_transform(r, np.eye(2), offset=(pady0, padx0), cval=1, output_shape=(ch, cw))
        return r

    except RuntimeError:
        # workaround for platform differences between 32bit and 64bit
        # scipy.ndimage
        dtype = sub.dtype
        sub = np.array(sub,dtype='float64')
        sub = interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
        sub = np.array(sub,dtype=dtype)
        return sub

# TODO replace (ocropy)
def extract_masked(image,linedesc,pad=5,expand=0):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0,x0,y1,x1 = [int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start, \
                  linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
    if pad>0:
        mask = pad_image(linedesc.mask,pad,cval=0)
    else:
        mask = linedesc.mask
    line = extract(image,y0-pad,x0-pad,y1+pad,x1+pad)
    if expand>0:
        mask = filters.maximum_filter(mask,(expand,expand))
    line = np.where(mask,line,np.amax(line))
    return line

# TODO replace (ocropy)
def remove_noise(line,minsize=8):
    """Remove small pixels from an image."""
    if minsize==0: return line
    bin = (line>0.5*amax(line))
    labels,n = morph.label(bin)
    sums = measurements.sum(bin,labels,range(n+1))
    sums = sums[labels]
    good = minimum(bin,1-(sums>0)*(sums<minsize))
    return good


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

    for n, (r_id, region) in enumerate(regions.items()):
        textregion = xmltree.xpath('//ns:TextRegion[@id="{}"]'.format(r_id), namespaces=ns)[0]
        if region.orientation:
            textregion.attrib['orientation'] = region.orientation

        for coords_elem in textregion.xpath("./ns:Coords", namespaces=ns):
            textregion.remove(coords_elem)
        etree.SubElement(textregion, "Coords", coords(region.coords))

        def l_id(index):
            return "{}_l{:03d}".format(r_id, index)

        # TextLines
        if len(region.lines) == 0:
            # Iterpret whole region as textline if no textline are found
            s_print({"id": l_id(n+1)})
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

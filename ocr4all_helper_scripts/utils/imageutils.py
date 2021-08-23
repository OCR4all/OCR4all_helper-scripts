from lxml import etree
from PIL import Image, ImageDraw


def remove_images(image: Image, tree: etree.Element):
    """Draw white over ImageRegions
    """
    white = {
        "1": 1, "L": 255, "P": 255,
        "RGB": (255, 255, 255), "RGBA": (255, 255, 255, 255),
        "CMYK": (0, 0, 0, 0), "YCbCr": (1, 0, 0),
        "Lab": (100, 0, 0), "HSV": (0, 0, 100)
    }[image.mode]
    draw = ImageDraw.Draw(image)
    for image_region in tree.xpath('//{*}ImageRegion'):
        for coords in image_region.xpath("./{*}Coords"):
            coordstrings = [x.split(",") for x in coords.attrib["points"].split()]
            poly = [(int(x[0]), int(x[1])) for x in coordstrings]
            draw.polygon(poly, fill=white)
    del draw

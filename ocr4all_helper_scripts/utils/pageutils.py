from lxml import etree
from shapely.geometry import Polygon, GeometryCollection


def sanitize(polygon: Polygon,
             parent: Polygon,
             page_width: int,
             page_height: int):
    sanitized_polygon = parent.intersection(polygon)
    # If intersection leads to more than one element just use the element with the largest area as all others are
    # most likely nice. TODO: check if this can't be done more elegantely
    if isinstance(sanitized_polygon, GeometryCollection):
        sanitized_polygon = max(sanitized_polygon, key=lambda a: a.area)
    sanitized_polygon = [(min(page_width, max(0, x)),
                          min(page_height, max(0, y))) for x, y in sanitized_polygon.exterior.coords]
    return sanitized_polygon


def get_root(xmlfile: str) -> etree.Element:
    try:
        return etree.parse(xmlfile).getroot()
    except etree.ParseError as e:
        raise e


def convert_point_notation(tree: etree.Element):
    """Converts point notation from older PAGE XML versions to latest coords attribute notation

    """
    for coord in [c for c in tree.findall(".//{*}Coords") if not c.attrib.get("points")]:
        cc = []
        for point in coord.find("./{*}Point"):
            cx = point.attrib["x"]
            cy = point.attrib["y"]
            coord.remove(point)
            cc.append(f"{cx},{cy}")
        coord.attrib["points"] = " ".join(cc)


def construct_coordmap(tree: etree.Element) -> dict:
    """Construct coordmap from PAGE XML tree which holds coordinate and orientation information for every
    TextRegion element"""
    coordmap = {}

    for text_region in tree.findall('.//{*}TextRegion'):
        region_id = text_region.attrib["id"]
        coordmap[region_id] = {"type": text_region.attrib.get("type", "TextRegion")}
        coordmap[region_id]["coords"] = []

        for coord in text_region.findall("./{*}Coords"):
            coordmap[region_id]["coordstring"] = coord.attrib["points"]
            coordstrings = [x.split(",") for x in coord.attrib["points"].split()]
            coordmap[region_id]["coords"] += [[int(x[0]), int(x[1])] for x in coordstrings]
        if text_region.attrib.get("orientation"):
            coordmap[region_id]["orientation"] = float(text_region.attrib["orientation"])

    return coordmap


def remove_existing_textlines(tree: etree.Element):
    for textline in tree.findall(".//{*}TextLine"):
        textline.getparent().remove(textline)

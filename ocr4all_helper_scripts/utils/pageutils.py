from typing import Dict

from lxml import etree


def get_root(xmlfile: str) -> etree.Element:
    try:
        return etree.parse(xmlfile).getroot()
    except etree.ParseError as e:
        raise e


def autoextract_namespace(tree: etree.Element) -> Dict[str, str]:
    extracted_ns = tree.xpath("naemsapce-uri(.)")

    if extracted_ns.startswith("http://schema.primaresearch.org/PAGE/gts/pagecontent/"):
        return {"page": extracted_ns}
    else:
        return {}


def convert_point_notation(tree: etree.Element, ns_map: Dict[str, str]):
    """Converts point notation from older PAGE XML versions to latest coords attribute notation

    """
    for coord in tree.xpath("//page:Coords[not(@points)]", namespaces=ns_map):
        cc = []
        for point in coord.xpath("./page:Point", namespaces=ns_map):
            cx = point.attrib["x"]
            cy = point.attrib["y"]
            coord.remove(point)
            cc.append(f"{cx},{cy}")
        coord.attrib["points"] = " ".join(cc)


def construct_coordmap(tree: etree.Element, ns_map: Dict[str, str]) -> dict:
    """Construct coordmap from PAGE XML tree which holds coordinate and orientation information for every
    TextRegion element"""
    coordmap = {}

    for text_region in tree.xpath('.//page:TextRegion', namespaces=ns_map):
        region_id = text_region.attrib["id"]
        coordmap[region_id] = {"type": text_region.attrib.get("type", "TextRegion")}
        coordmap[region_id]["coords"] = []

        for c in text_region.xpath("./page:Coords", namespaces=ns_map) + text_region.xpath("./Coords"):
            coordmap[region_id]["coordstring"] = c.attrib["points"]
            coordstrings = [x.split(",") for x in c.attrib["points"].split()]
            coordmap[region_id]["coords"] += [[int(x[0]), int(x[1])] for x in coordstrings]
        if text_region.attrib.get("orientation"):
            coordmap[region_id]["orientation"] = float(text_region.attrib["orientation"])

    return coordmap

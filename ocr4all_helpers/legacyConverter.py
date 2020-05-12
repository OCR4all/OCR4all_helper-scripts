from pathlib import Path
import argparse
from typing import Dict, List, Tuple

from lxml import etree


def get_namespace(tree: etree.Element) -> Dict[str, str]:
    """Automatically extracts Page XML namespace

    :param tree: Base Page XML file representing the page.
    :return: Namespace dictionary.
    """
    return {"p": tree.nsmap[None]}


def convert_page(xml: Path) -> etree.Element:
    """Converts a page in a legacy OCR4all project to latest by getting and processing the information in the affiliated
    directories and writing them into the pages Page XML file.

    :param xml: Base Page XML file representing the page.
    :return: XML tree enriched with all necessary information from the legacy directories.
    """
    tree = etree.parse(str(xml)).getroot()
    ns = get_namespace(tree)
    page_dir = Path(xml.parent, xml.stem)

    line_counter = 1

    regions_data = get_regions_data(page_dir)

    for region in regions_data:
        _region = region["name"]
        region["line_coords"], region["pred"], region["gt"], region["bin"], region["nrm"] = process_lines(
            Path(page_dir, _region), region["offset"])

    for order, region in enumerate(tree.xpath(".//p:TextRegion", namespaces=ns)):
        region_data = regions_data[order]

        region_text_equiv = region.xpath("./p:TextEquiv", namespaces=ns)

        if region_text_equiv:
            region.remove(region_text_equiv[0])

        for line_number, line in enumerate(region_data["line_coords"]):
            line_elem = etree.Element("TextLine")
            line_elem.attrib["id"] = f"l{line_counter}"
            line_counter += 1

            line_coord_elem = etree.Element("Coords")
            line_coord_elem.attrib["points"] = calc_bbox(region_data, line_number)
            line_elem.append(line_coord_elem)

            if region_data["pred"][line_number]:
                prediction_elem = etree.Element("TextEquiv")
                prediction_elem.attrib["index"] = "1"

                prediction_unicode = etree.Element("Unicode")
                prediction_unicode.text = region_data["pred"][line_number]
                prediction_elem.append(prediction_unicode)

                line_elem.append(prediction_elem)

            if region_data["gt"][line_number]:
                gt_elem = etree.Element("TextEquiv")
                gt_elem.attrib["index"] = "0"

                gt_unicode = etree.Element("Unicode")
                gt_unicode.text = region_data["gt"][line_number]
                gt_elem.append(gt_unicode)

                line_elem.append(gt_elem)

            region.append(line_elem)

    return tree


def get_regions_data(path: Path) -> List[dict]:
    """Gets base data for a region in a page.

    :param path: Path to the region directory.
    :return: List of dictionaries holding base region data.
    """
    regions = list()

    for region in sorted(path.glob("./*.offset")):
        region_dict = dict()

        with region.open("r") as file:
            (x, y) = file.read().split(",")
            region_dict["name"] = str(region.stem)
            region_dict["offset"] = (int(x), int(y))

        regions.append(region_dict)

    return regions


def process_lines(path: Path, offset: Tuple[int, int]) -> Tuple[list, list, list, list, list]:
    """Collects image and text information about each line in region.

    :param path: Path to the region directory.
    :param offset: Region offset for calculating actual coordinates of the lines.
    :return: Information about line coordinates, prediction and ground truth text and binary and greyscale images.
    """
    line_coords = []
    prediction = []
    gt = []
    bin_imgs = []
    nrm_imgs = []

    for line_coord in sorted(path.glob("./*.coords")):
        prediction_file = Path(line_coord.parent, line_coord.stem).with_suffix(".pred.txt")
        gt_file = Path(line_coord.parent, line_coord.stem).with_suffix(".gt.txt")

        bin_imgs.append(Path(line_coord.parent, line_coord.stem).with_suffix(".bin.png"))
        nrm_imgs.append(Path(line_coord.parent, line_coord.stem).with_suffix(".nrm.png"))

        with line_coord.open("r") as coord_file:
            (y_min, x_min, y_max, x_max) = coord_file.read().split(",")
            line_coords.append((int(y_min) + offset[1], int(x_min) + offset[0], int(y_max) + offset[1],
                                int(x_max) + offset[0]))

        if prediction_file.is_file():
            with prediction_file.open("r") as pred_file:
                prediction.append("".join(pred_file.read()))
        else:
            prediction.append(None)

        if gt_file.is_file():
            with gt_file.open("r") as pred_file:
                gt.append("".join(pred_file.read()))
        else:
            gt.append(None)

    return line_coords, prediction, gt, bin_imgs, nrm_imgs


def calc_bbox(region: dict, linenumber: int) -> str:
    """Calculates bounding from line coordinate files.

    :param region: Region data dictionary.
    :param linenumber: Number of the line for which the bounding box shall be calculated.
    :return: String representation of the bounding box.
    """
    coords = region["line_coords"][linenumber]
    return f"{coords[1]},{coords[2]} {coords[1]},{coords[0]} {coords[3]},{coords[0]} {coords[3]},{coords[2]}"


def write_xml(file: Path, tree: etree.Element):
    """Writes the enriched XML tree to file.

    :param file: Output file.
    :param tree: Enriched XML tree.
    """
    with file.open("w") as outfile:
        outfile.write(etree.tostring(tree, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser("""
    Convert legacy OCR4all projects to latest.
    """)
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the OCR4all project.')

    args = parser.parse_args()

    for xml in Path(args.path).glob("*.xml"):
        updated_page = convert_page(xml)
        write_xml(xml, updated_page)


if __name__ == "__main__":
    main()



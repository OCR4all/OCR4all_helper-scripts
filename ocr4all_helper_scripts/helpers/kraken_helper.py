from pathlib import Path
import subprocess
from typing import List
import sys

from lxml import etree


class KrakenHelper:
    def __init__(self, files):
        self.files = [Path(file) for file in files]

    def run(self):
        files_args = []
        for file in self.files:
            files_args.append("-i")
            files_args.append(str(file))
            files_args.append(str(Path(file.parent, f"{file.name.split('.')[0]}.xml")))

        command = ["kraken", "-x", "-v"]
        command.extend(files_args)
        command.append("segment")
        command.append("-bl")
        subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)

    def postprocess(self):
        """Fixes several non-valid PAGE XML entries produced by kraken in the currently used version.

        """
        for file in self.files:
            xml = Path(file.parent, f"{file.name.split('.')[0]}.xml")
            root = etree.parse(str(xml)).getroot()

            text_regions = root.findall(".//{*}TextRegion")
            ro = list()

            for idx, text_region in enumerate(text_regions):
                coords = text_region.find("./{*}Coords")
                points = coords.get("points")
                if "-" in points:
                    points = points.replace("-", "")
                    coords.set("points", points)

                if text_region.get("context") is None and coords.get("points").startswith("0,0"):
                    self.shrink_full_page_region(text_region)

                new_id = f"r_{str(idx).zfill(4)}"
                text_region.set("id", new_id)
                ro.append(f"r_{str(idx).zfill(4)}")

                for line in text_region.findall("./{*}TextLine"):
                    line_coords = line.find("./{*}Coords")
                    line_points = line_coords.get("points")
                    if "-" in line_points:
                        line_points = line_points.replace("-", "")
                        line_coords.set("points", line_points)

            self.create_reading_order(root, ro)
            with xml.open("w") as outfile:
                outfile.write(etree.tostring(root, encoding="unicode", pretty_print=True))

    @staticmethod
    def shrink_full_page_region(text_region: etree.Element):
        """Shrinks full page TextRegion which only contain one TextLine element to the coords of the TextLine element.
        These elements are produced by kraken when a baseline and its TextLine element can't be assigned to an existing
        region.

        """
        region_coord = text_region.find("./{*}Coords")

        textline = text_region.find("./{*}TextLine")
        textline_coord = textline.find("./{*}Coords")

        region_coord.set("points", textline_coord.get("points"))

    @staticmethod
    def create_reading_order(root: etree.Element, reading_order: List[str]):
        """Creates ReadingOrder element from existing order in XML tree as kraken doesn't create this itself.

        """
        page_elem = root.find("./{*}Page")

        reading_order_element = etree.Element("ReadingOrder")
        ordered_group_element = etree.SubElement(reading_order_element, "OrderedGroup")
        ordered_group_element.set("id", "g0")

        for idx, elem in enumerate(reading_order):
            region_ref_index_elem = etree.SubElement(ordered_group_element, "RegionRefIndexed")
            region_ref_index_elem.set("index", str(idx))
            region_ref_index_elem.set("regionRef", elem)

        page_elem.insert(0, reading_order_element)

from pathlib import Path
import subprocess

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

        command = f"kraken -x -v {' '.join(files_args)} segment -bl"
        subprocess.call(command, shell=True)

    def postprocess(self):
        for file in self.files:
            xml = Path(file.parent, f"{file.name.split('.')[0]}.xml")
            root = etree.parse(str(xml)).getroot()
            text_regions = root.findall(".//{*}TextRegion")
            for idx, text_region in enumerate(text_regions):
                text_region.set("id", f"r_{str(idx).zfill(4)}")
            with xml.open("w") as outfile:
                outfile.write(etree.tostring(root, encoding="unicode", pretty_print=True))

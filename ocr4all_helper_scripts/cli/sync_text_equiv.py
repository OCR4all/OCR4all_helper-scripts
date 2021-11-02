from typing import Tuple
from pathlib import Path

import click
from lxml import etree


@click.command("sync-text-equiv", help="Sync region text equiv elements with textline text equiv content.")
@click.argument("FILES", nargs=-1, required=True, type=str)
def sync_text_equiv_cli(files):
    for file in files:
        root = etree.parse(str(file)).getroot()
        ns = {"ns": root.nsmap[None]}

        text_regions = root.xpath(".//ns:TextRegion", namespaces=ns)
        for text_region in text_regions:
            gt_text, rec_text = get_text_content(text_region)
            if gt_text:
                set_text(ns, text_region, gt_text, "0")
            if rec_text:
                set_text(ns, text_region, rec_text, "1")
        with Path(file).open("w") as outfile:
            outfile.write(etree.tostring(root, encoding="unicode", pretty_print=True))


def set_text(ns: dict, text_region: etree.Element, content: str, index: str):
    existing_equiv = text_region.xpath(f"./ns:TextEquiv[@index='{index}']", namespaces=ns)
    if len(existing_equiv) == 1:
        existing_equiv[0].find("./{*}Unicode").text = content
    else:
        equiv = etree.SubElement(text_region, "TextEquiv")
        equiv.set("index", index)
        unicode = etree.SubElement(equiv, "Unicode")
        unicode.text = content


def get_text_content(text_region: etree.Element) -> Tuple[str, str]:
    lines = text_region.findall(".//{*}TextLine")
    gt_text, rec_text = [], []
    for line in lines:
        equiv = line.find("./{*}TextEquiv")
        if equiv is not None:
            content = "".join(equiv.find("./{*}Unicode").itertext())
            if content and equiv.get("index") == "0":
                gt_text.append(content)
            elif content and equiv.get("index") == "1":
                rec_text.append(content)
    return "\n".join(gt_text), "\n".join(rec_text)


if __name__ == "__main__":
    sync_text_equiv_cli()

from ocr4all_helper_scripts.helpers import legacyconvert_helper

from pathlib import Path
import click


@click.command("legacy-convert", help="Convert legacy OCR4all projects to latest.")
@click.option("-p", "--path", type=str, required=True, help="Path to the OCR4all project.")
def legacyconvert_cli(path):
    for xml in sorted(list(Path(path).glob("*.xml"))):
        updated_page = legacyconvert_helper.convert_page(xml)
        legacyconvert_helper.write_xml(xml, updated_page)


if __name__ == "__main__":
    legacyconvert_cli()

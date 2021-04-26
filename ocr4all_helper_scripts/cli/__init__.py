from ocr4all_helper_scripts.cli.legacyconvert import legacyconvert_cli
from ocr4all_helper_scripts.cli.pagelineseg import pagelineseg_cli
from ocr4all_helper_scripts.cli.skewestimate import skewestimate_cli

import click


@click.group()
@click.version_option()
def cli(**kwargs):
    """
    CLI entrypoint for OCR4all-helper-scripts
    """


cli.add_command(legacyconvert_cli)
cli.add_command(pagelineseg_cli)
cli.add_command(skewestimate_cli)

from ocr4all_helper_scripts.cli.legacyconvert import legacyconvert_cli
from ocr4all_helper_scripts.cli.pagelineseg import pagelineseg_cli
from ocr4all_helper_scripts.cli.skewestimate import skewestimate_cli
from ocr4all_helper_scripts.cli.sync_text_equiv import sync_text_equiv_cli
from ocr4all_helper_scripts.cli.kraken import kraken_cli
from ocr4all_helper_scripts.cli.calamari_eval_wrapper import calamari_eval_cli

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
cli.add_command(sync_text_equiv_cli)
cli.add_command(kraken_cli)
cli.add_command(calamari_eval_cli)

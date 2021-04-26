import click

@click.group()
@click.version_option()
def cli(**kwargs):
    """
    CLI entrypoint for OCR4all-helper-scripts
    """

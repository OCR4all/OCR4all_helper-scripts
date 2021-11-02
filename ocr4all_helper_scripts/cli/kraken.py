from ocr4all_helper_scripts.helpers import kraken_helper


import click


@click.command("sync-text-equiv", help="Sync region text equiv elements with textline text equiv content.")
@click.argument("FILES", nargs=-1, required=True, type=str)
def kraken_cli(files):
    helper = kraken_helper.KrakenHelper(files)
    helper.run()
    helper.postprocess()


if __name__ == "__main__":
    kraken_cli()
